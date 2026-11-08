"""Run the official LightGBM native API with the existing complete architecture."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import urllib.request
import lightgbm as lgb

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.dart_dropout.run_experiment import (
    REPO, WINDOWS, FINAL_CUTOFF, TARGETS, PROCESSING, RESIDUAL,
    load_designs, archived_baseline, validate_provenance, input_files,
    source_files as dependency_sources, sha, now, log, write_json,
    score_predictions)
from experiments.data_processing_round1.features import ProcessingFeatures, assemble, text_features
from experiments.data_processing_round1.run_experiment import load_data
from experiments.plan_a.features import HistoryFeatures
from experiments.plan_a.run_experiment import metrics
from experiments.property_groups.models import LOT_COLUMNS
from .models import PARAMS, ROUNDS, VERSION, OfficialLightGBM

HERE = Path(__file__).resolve().parent
CANDIDATE = 'official_lightgbm_native'
BASELINE = 'current_best_all_three'


def fit_window(raw, test, cutoff, end, out):
    homes, designs = load_designs(raw, test, cutoff, end)
    folder = out/'windows'/cutoff
    folder.mkdir(parents=True, exist_ok=False)
    control = archived_baseline(homes, cutoff, final=end is None)
    control.to_csv(folder/'baseline_predictions.csv', index=False)
    prediction = control.copy()
    audits = []
    for kind, (train, valid, xt, xv) in designs.items():
        if not (train.date_sold < pd.Timestamp(cutoff)).all() or set(train.ml_num) & set(valid.ml_num):
            raise ValueError('Temporal split or training/validation IDs overlap')
        started = time.monotonic()
        log(f'训练 官方 LightGBM {cutoff}/{kind} n={len(train)} features={len(xt.columns)}')
        model = OfficialLightGBM().fit(xt, train.sold_price)
        price = model.predict(xv)
        expected = control.set_index('ml_num').predicted_price.reindex(valid.ml_num).to_numpy()
        np.testing.assert_array_equal(price, expected)
        model_dir = folder/kind/CANDIDATE
        model.save(model_dir, cutoff_exclusive=cutoff, training_n=len(train),
                   max_train_sale=str(train.date_sold.max()), target_type=kind,
                   training_types=sorted(train.house_type_name.unique()))
        np.testing.assert_array_equal(price, OfficialLightGBM.load(model_dir).predict(xv))
        if model.booster.current_iteration() != ROUNDS:
            raise ValueError('Unexpected boosting round count')
        pd.DataFrame({'ml_num': valid.ml_num, 'predicted_price': price}).to_csv(
            model_dir/'predictions_unscored.csv', index=False)
        pd.DataFrame({'feature': model.feature_names,
                      'total_gain': model.booster.feature_importance(importance_type='gain')}).to_csv(
            model_dir/'feature_importance.csv', index=False)
        update = pd.Series(price, index=valid.ml_num)
        mask = prediction.house_type_name.eq(kind)
        prediction.loc[mask, 'predicted_price'] = update.reindex(prediction.loc[mask, 'ml_num']).to_numpy()
        audits.append(dict(target=kind, training_n=len(train), features=len(xt.columns),
                           reload_exact=True, baseline_exact=True,
                           seconds=time.monotonic()-started))
        log(f'完成 {cutoff}/{kind}，预测未评分')
    unchanged = ~homes.house_type_name.isin(TARGETS).to_numpy()
    np.testing.assert_array_equal(prediction.predicted_price.to_numpy()[unchanged],
                                  control.predicted_price.to_numpy()[unchanged])
    parts = []
    for name, rows in [(BASELINE, control), (CANDIDATE, prediction)]:
        rows = rows.copy()
        rows['candidate'], rows['cutoff_exclusive'] = name, cutoff
        rows['protocol'] = 'final' if end is None else ('check' if cutoff == '2026-02-01' else 'development')
        parts.append(rows)
    result = pd.concat(parts, ignore_index=True)
    result.to_csv(folder/'predictions_unscored.csv', index=False)
    write_json(folder/'audit.json', dict(models=audits, unchanged_other_branches=True))
    return result


def fresh_final_predict(raw, test, run):
    reference = raw.sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    hidden = data.hide_answers(test, 'off_market').reset_index(drop=True)
    base_provider = HistoryFeatures(reference, FINAL_CUTOFF)
    base = base_provider.transform(hidden)
    target = hidden.loc[hidden.house_type_name.isin(TARGETS[:2])].reset_index(drop=True)
    provider = ProcessingFeatures(reference, FINAL_CUTOFF)
    processed = assemble(base_provider.transform(target), provider.history(target),
                         provider.land(target), text_features(target), 'all_three')
    result = archived_baseline(test, FINAL_CUTOFF, final=True).set_index('ml_num')
    for kind in TARGETS:
        homes, design = (target, processed) if kind in TARGETS[:2] else (hidden, base)
        mask = homes.house_type_name.eq(kind)
        x = design.loc[mask]
        if kind == 'Condo Townhouse':
            x = x.drop(columns=list(LOT_COLUMNS))
        model = OfficialLightGBM.load(Path(run)/'windows'/FINAL_CUTOFF/kind/CANDIDATE)
        result.loc[homes.loc[mask, 'ml_num'], 'predicted_price'] = model.predict(x)
    return result.reindex(test.ml_num).reset_index()[['ml_num', 'predicted_price']]


def comparisons(scored):
    rows = []
    dimensions = [('overall', ['protocol']), ('type', ['protocol', 'house_type_name']),
                  ('window', ['protocol', 'cutoff_exclusive']),
                  ('type_window', ['protocol', 'cutoff_exclusive', 'house_type_name']),
                  ('month', ['protocol', 'sold_month'])]
    for level, keys in dimensions:
        for key, group in scored.groupby(keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            a = group.loc[group.candidate.eq(BASELINE)].set_index('ml_num').sort_index()
            b = group.loc[group.candidate.eq(CANDIDATE)].set_index('ml_num').sort_index()
            if not a.index.equals(b.index):
                raise ValueError('Paired cohort mismatch')
            left, right = metrics(a), metrics(b)
            row = dict(level=level, **dict(zip(keys, key)), n=len(a))
            for metric in ('mdape', 'mape', 'median_signed_error', 'over_30_pct'):
                row.update({f'current_best_{metric}': left[metric],
                            f'official_lightgbm_{metric}': right[metric],
                            f'delta_{metric}_pp': right[metric]-left[metric]})
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(out, comparison, checks):
    overall = comparison.loc[comparison.level.eq('overall') & comparison.protocol.eq('check')].iloc[0]
    lines = ['# LightGBM 官方接口接入：运行结果', '',
        '现有模型本来就使用官方 LightGBM 4.7.0。此次按官方 GitHub Python 示例，改用 Dataset → train → Booster；运行使用官方发行包，没有重新编译，也没有改成另一套算法。',
        '保留 all_three 数据处理、四个房型分支及其他既有路由，Log-L1、600轮、31叶、学习率0.05和所有等价参数保持一致。全部回测预测冻结后评分，无调参、早停或按分数选择房型。', '',
        '## 与当前最优完整方案比较', '',
        '同7773套、截至2026-01训练、模型和成交池冻结、预测2026-02至05。以下都是累计开发回测，隐藏测试成绩未知。', '',
        '| 方案 | MdAPE | MAPE | MdAPE差值/百分点 |', '|---|---:|---:|---:|',
        f'| 当前最优 all_three＋既有路由 | {overall.current_best_mdape:.6f}% | {overall.current_best_mape:.6f}% | 0 |',
        f'| 官方 LightGBM 原生接口＋相同架构 | {overall.official_lightgbm_mdape:.6f}% | {overall.official_lightgbm_mape:.6f}% | {overall.delta_mdape_pp:+.6f} |', '',
        '所有窗口及最终候选均与当前最优存档逐套预测完全相同。这证明本次接入复现了原模型，没有带来额外的算法或分数收益。', '',
        '## 同房型比较', '',
        '| 房型 | n | 当前最优MdAPE | 官方接口MdAPE | 差值/百分点 |', '|---|---:|---:|---:|---:|']
    for _, row in comparison.loc[comparison.level.eq('type') & comparison.protocol.eq('check')].iterrows():
        lines.append(f'| {row.house_type_name} | {int(row.n)} | {row.current_best_mdape:.3f}% | {row.official_lightgbm_mdape:.3f}% | {row.delta_mdape_pp:+.3f} |')
    for level, column, title, protocol in [('window', 'cutoff_exclusive', '2025年开发窗口', 'development'),
                                           ('month', 'sold_month', '2026年逐月', 'check')]:
        lines += ['', f'## {title}', '', '| 窗口/月 | n | 当前最优MdAPE | 官方接口MdAPE | 差值/百分点 |', '|---|---:|---:|---:|---:|']
        for _, row in comparison.loc[comparison.level.eq(level) & comparison.protocol.eq(protocol)].iterrows():
            lines.append(f'| {row[column]} | {int(row.n)} | {row.current_best_mdape:.3f}% | {row.official_lightgbm_mdape:.3f}% | {row.delta_mdape_pp:+.3f} |')
    lines += ['', '## 来源与核验', '',
        '- 官方仓库：https://github.com/lightgbm-org/LightGBM 。版本4.7.0，上游固定提交见upstream/provenance.json；示例、训练引擎、sklearn封装及许可证已存档。',
        '- 运行使用的Python模块及原生库哈希见protocol.json；原生源码参考不等同于从GitHub重新编译安装。',
        '- 官方示例说明接口；本实验继续使用房价任务的原参数，不复制示例数据、L2损失或早停设置。',
        '- 16个回测模型及4个最终模型均保存并加载重放，与当前最优逐套一致。',
        f'- 最终{checks["final_n"]}行隔离预测通过官方格式检查；从原始数据重建最终输入后重放一致。',
        '- 当前最优索引和正式提交保持不变；完整同ID配对结果见comparison_to_current_best.csv。']
    (out/'实验报告.md').write_text('\n'.join(lines)+'\n')


def capture_upstream(out):
    commit = '8f7036f03627054d5a54a6f965b13f4b9ff2cb63'
    paths = ['examples/python-guide/simple_example.py', 'python-package/lightgbm/engine.py',
             'python-package/lightgbm/sklearn.py', 'VERSION.txt', 'LICENSE']
    folder = out/'upstream'
    folder.mkdir()
    files = {}
    for name in paths:
        url = f'https://raw.githubusercontent.com/lightgbm-org/LightGBM/{commit}/{name}'
        request = urllib.request.Request(url, headers={'User-Agent': 'SSU-Data-Dash-reproducibility'})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        target = folder/name.replace('/', '__')
        target.write_bytes(content)
        files[name] = dict(url=url, local_file=str(target.relative_to(out)), sha256=sha(target))
    if (folder/'VERSION.txt').read_text().strip() != VERSION:
        raise ValueError('Upstream release version mismatch')
    metadata = dict(repository='https://github.com/lightgbm-org/LightGBM', tag='v'+VERSION,
                    commit=commit, retrieved_at=now(), files=files,
                    execution='Installed official release wheel; upstream files are reference sources, not a source build')
    write_json(folder/'provenance.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    validate_provenance()
    out = args.output.resolve()
    out.mkdir(parents=True)
    best = json.loads((REPO/'CURRENT_BEST_MODEL.json').read_text())
    write_json(out/'current_best_before.json', best)
    if best['off_market_fixed_four_months']['model_id'] != 'data_processing_round1/run_2026-10-19_v1/all_three':
        raise ValueError('Current best changed: rebase the experiment control first')
    sources = sorted(set([*dependency_sources(), *HERE.glob('*.py'), HERE/'requirements.txt',
                           HERE/'预先固定方案.md', HERE/'README.md', REPO/'tests/test_lightgbm_official.py']))
    upstream = capture_upstream(out)
    inputs = input_files(args.data_dir)
    protocol = dict(frozen_at=now(), windows=WINDOWS, final_cutoff=FINAL_CUTOFF,
        candidate=CANDIDATE, num_boost_round=ROUNDS, upstream=upstream, baseline_model=best['off_market_fixed_four_months'], params=PARAMS,
        target_types=list(TARGETS), parameter_search=False, early_stopping=False,
        score_based_selection=False, previously_examined_windows=True,
        scope='Official release native API integration; same algorithm, parameters and property routes',
        reference_pool_updates_within_window=False, feature_archive=str(PROCESSING), base_archive=str(RESIDUAL),
        source_sha256={str(p): sha(p) for p in sources}, input_sha256={str(p): sha(p) for p in inputs},
        versions={name: importlib.metadata.version(name) for name in
                  ('lightgbm', 'catboost', 'numpy', 'pandas', 'pyarrow', 'scipy')})
    library_files = [Path(lgb.__file__), *Path(lgb.__file__).parent.glob('lib/*')]
    protocol['runtime_sha256'] = {str(p): sha(p) for p in library_files if p.is_file()}
    write_json(out/'protocol.json', protocol)
    protocol_hash = sha(out/'protocol.json')
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot/str(path.relative_to(REPO)).replace('/', '__'))
    raw, test = load_data(args.data_dir)
    parts = [fit_window(raw, test, cutoff, end, out) for cutoff, end in WINDOWS]
    prediction = pd.concat(parts, ignore_index=True)
    prediction.to_csv(out/'all_predictions_unscored.csv', index=False)
    write_json(out/'prediction_freeze.json', dict(frozen_at=now(), protocol_sha256=protocol_hash,
        unscored_sha256=sha(out/'all_predictions_unscored.csv'), backtest_official_lightgbm_models=16,
        archived_baseline_exact=True, score_based_selection=False))
    log('全部16个官方LightGBM回测逐套复现当前最优，开始评分')
    scored, scores = score_predictions(prediction, raw)
    scored.to_csv(out/'all_predictions_scored.csv', index=False)
    scores.to_csv(out/'scores.csv', index=False)
    comparison = comparisons(scored)
    comparison.to_csv(out/'comparison_to_current_best.csv', index=False)
    final = fit_window(raw, test, FINAL_CUTOFF, None, out)
    prices = final.loc[final.candidate.eq(CANDIDATE), ['ml_num', 'predicted_price']].reset_index(drop=True)
    target = out/'final_predictions'
    target.mkdir()
    submission = target/'submission_off_market_lightgbm_official.csv'
    prices.to_csv(submission, index=False)
    errors = check_submission.check_file(submission, test.ml_num)
    if errors:
        raise ValueError(errors)
    log('从原始数据重建特征并重放最终官方 LightGBM')
    replay = fresh_final_predict(raw, test, out)
    pd.testing.assert_frame_equal(prices, replay)
    replay.to_csv(target/'replay.csv', index=False)
    for mapping in ('source_sha256', 'input_sha256', 'runtime_sha256'):
        if any(sha(p) != digest for p, digest in protocol[mapping].items()):
            raise ValueError(f'Protected {mapping} changed')
    if sha(out/'protocol.json') != protocol_hash:
        raise ValueError('Protocol changed')
    checks = dict(completed_at=now(), final_n=len(prices), official_lightgbm_models=20,
        exact_archived_branches=20, all_controls_exact=True, all_reload_exact=True,
        raw_final_feature_replay_exact=True, unchanged_other_branches=True,
        submission_errors=errors, protected_files_unchanged=True)
    write_json(out/'checks.json', checks)
    write_report(out, comparison, checks)
    log(f'完成 {out}')


if __name__ == '__main__':
    main()
