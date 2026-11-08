"""Run the preregistered XGBoost substitution without external-label selection."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.dart_dropout.models import DropoutModel
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
from .models import PARAMS, XGBoostPriceModel

HERE = Path(__file__).resolve().parent
CANDIDATE = 'xgboost_log_l1'
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
        # Ordinary LightGBM under the tested equivalent initial-constant
        # representation must reproduce the archived current-best branch.
        if end is not None:
            log(f'复现当前最优 {cutoff}/{kind}')
            baseline = DropoutModel().fit(xt, train.sold_price)
            baseline_prices = baseline.predict(xv)
            expected = control.set_index('ml_num').predicted_price.reindex(valid.ml_num).to_numpy()
            np.testing.assert_array_equal(baseline_prices, expected)
            baseline.save(folder/kind/'lightgbm_control', cutoff_exclusive=cutoff,
                          training_n=len(train), max_train_sale=str(train.date_sold.max()),
                          target_type=kind, training_types=sorted(train.house_type_name.unique()))
            np.testing.assert_array_equal(baseline_prices,
                DropoutModel.load(folder/kind/'lightgbm_control').predict(xv))
        started = time.monotonic()
        log(f'训练 XGBoost {cutoff}/{kind} n={len(train)} features={len(xt.columns)}')
        model = XGBoostPriceModel().fit(xt, train.sold_price)
        price = model.predict(xv)
        model_dir = folder/kind/CANDIDATE
        model.save(model_dir, cutoff_exclusive=cutoff, training_n=len(train),
                   max_train_sale=str(train.date_sold.max()), target_type=kind,
                   training_types=sorted(train.house_type_name.unique()))
        np.testing.assert_array_equal(price, XGBoostPriceModel.load(model_dir).predict(xv))
        if model.model.get_booster().num_boosted_rounds() != PARAMS['n_estimators']:
            raise ValueError('Unexpected boosting round count')
        pd.DataFrame({'ml_num': valid.ml_num, 'predicted_price': price}).to_csv(
            model_dir/'predictions_unscored.csv', index=False)
        gain = model.model.get_booster().get_score(importance_type='total_gain')
        pd.DataFrame({'feature': model.feature_names,
                      'total_gain': [gain.get(c, 0.) for c in model.feature_names]}).to_csv(
            model_dir/'feature_importance.csv', index=False)
        update = pd.Series(price, index=valid.ml_num)
        mask = prediction.house_type_name.eq(kind)
        prediction.loc[mask, 'predicted_price'] = update.reindex(prediction.loc[mask, 'ml_num']).to_numpy()
        audits.append(dict(target=kind, training_n=len(train), features=len(xt.columns),
                           reload_exact=True, baseline_exact=end is not None,
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
        model = XGBoostPriceModel.load(Path(run)/'windows'/FINAL_CUTOFF/kind/CANDIDATE)
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
                            f'xgboost_{metric}': right[metric],
                            f'delta_{metric}_pp': right[metric]-left[metric]})
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(out, comparison, checks):
    overall = comparison.loc[comparison.level.eq('overall') & comparison.protocol.eq('check')].iloc[0]
    lines = ['# XGBoost 第一版固定算法替换：实验结果', '',
        '主候选事前固定为 xgboost_log_l1；四个 LightGBM 分支一起替换，其他分支保留当前最优冻结预测。',
        '训练、类别词表及成交参考池均限于各截止日前；每个四个月窗口内不更新。全部回测预测冻结后统一评分，无调参、早停、融合或按分数拼接房型。', '',
        '## 当前最优完整方案对照', '',
        '2026-02至05，共7773套；截至2026-01训练，成交池冻结。当前最优是 all_three + 既有房型路由。', '',
        '| 方案 | MdAPE | MAPE | 相对当前最优MdAPE/百分点 |', '|---|---:|---:|---:|',
        f'| 当前最优 all_three | {overall.current_best_mdape:.6f}% | {overall.current_best_mape:.6f}% | 0 |',
        f'| XGBoost Log-L1 | {overall.xgboost_mdape:.6f}% | {overall.xgboost_mape:.6f}% | {overall.delta_mdape_pp:+.6f} |', '',
        '## 同房型对照', '', '| 房型 | n | 当前最优MdAPE | XGBoost MdAPE | 差值/百分点 |', '|---|---:|---:|---:|---:|']
    for _, row in comparison.loc[comparison.level.eq('type') & comparison.protocol.eq('check')].iterrows():
        lines.append(f'| {row.house_type_name} | {int(row.n)} | {row.current_best_mdape:.3f}% | {row.xgboost_mdape:.3f}% | {row.delta_mdape_pp:+.3f} |')
    for level, column, title, protocol in [('window', 'cutoff_exclusive', '2025年开发窗口', 'development'),
                                           ('month', 'sold_month', '2026年逐月', 'check')]:
        lines += ['', f'## {title}', '', '| 窗口/月 | n | 当前最优MdAPE | XGBoost MdAPE | 差值/百分点 |', '|---|---:|---:|---:|---:|']
        for _, row in comparison.loc[comparison.level.eq(level) & comparison.protocol.eq(protocol)].iterrows():
            lines.append(f'| {row[column]} | {int(row.n)} | {row.current_best_mdape:.3f}% | {row.xgboost_mdape:.3f}% | {row.delta_mdape_pp:+.3f} |')
    lines += ['', '## 参数、来源与验证', '',
        '- XGBoost 3.1.3，原生类别、CPU hist、depthwise、Log-L1、600轮、学习率0.05、max_depth=5、min_child_weight=20、L2=5；完整参数见protocol.json。',
        '- 保留训练数值缺失，以训练集对数价格中位数为base_score；预测指数还原后取整。Log-L1不等于MdAPE。',
        '- 论文：https://arxiv.org/abs/1603.02754 ，第2节正则化、2.3节shrinkage、3.4节缺失分支。现代hist及原生类别为工程适配，不是2016年逐项复现。',
        '- 全部16个受控LightGBM分支逐套复现当前最优对应存档；20个XGBoost模型保存后重放一致。',
        f'- 最终按截至2026-05历史重训；隔离文件{checks["final_n"]}行，原始数据重建特征后重放一致，官方格式检查通过。',
        '- 全部时间窗口已被项目反复查看，属于累计开发回测；隐藏测试分数未知。仅评价本次固定参数，不能据此推断所有XGBoost配置。',
        '- 当前最优索引仅在独立核验通过后视完整同口径成绩更新；原始正式提交不被覆盖。', '',
        '完整配对结果见comparison_to_current_best.csv；逐套预测见all_predictions_scored.csv；保存模型见windows/。',
        '本轮只执行方案中的E0与固定E1；Log-L2、小预算调参、融合、快照训练及官方逐月重训未执行。']
    (out/'实验报告.md').write_text('\n'.join(lines)+'\n')


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
                           HERE/'预先固定方案.md', HERE/'README.md', REPO/'tests/test_xgboost.py']))
    inputs = input_files(args.data_dir)
    protocol = dict(frozen_at=now(), windows=WINDOWS, final_cutoff=FINAL_CUTOFF,
        candidate=CANDIDATE, baseline_model=best['off_market_fixed_four_months'], params=PARAMS,
        target_types=list(TARGETS), parameter_search=False, early_stopping=False,
        score_based_selection=False, previously_examined_windows=True,
        scope='E0 current-best reproduction and E1 fixed XGBoost substitution',
        reference_pool_updates_within_window=False, feature_archive=str(PROCESSING), base_archive=str(RESIDUAL),
        source_sha256={str(p): sha(p) for p in sources}, input_sha256={str(p): sha(p) for p in inputs},
        versions={name: importlib.metadata.version(name) for name in
                  ('xgboost', 'lightgbm', 'catboost', 'numpy', 'pandas', 'pyarrow', 'scipy')})
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
        unscored_sha256=sha(out/'all_predictions_unscored.csv'), backtest_xgboost_models=16,
        reproduced_lightgbm_models=16, score_based_selection=False))
    log('全部16个XGBoost回测与16个受控复现完成，开始评分')
    scored, scores = score_predictions(prediction, raw)
    scored.to_csv(out/'all_predictions_scored.csv', index=False)
    scores.to_csv(out/'scores.csv', index=False)
    comparison = comparisons(scored)
    comparison.to_csv(out/'comparison_to_current_best.csv', index=False)
    final = fit_window(raw, test, FINAL_CUTOFF, None, out)
    prices = final.loc[final.candidate.eq(CANDIDATE), ['ml_num', 'predicted_price']].reset_index(drop=True)
    target = out/'final_predictions'
    target.mkdir()
    submission = target/'submission_off_market_xgboost.csv'
    prices.to_csv(submission, index=False)
    errors = check_submission.check_file(submission, test.ml_num)
    if errors:
        raise ValueError(errors)
    log('从原始数据重建特征并重放最终XGBoost')
    replay = fresh_final_predict(raw, test, out)
    pd.testing.assert_frame_equal(prices, replay)
    replay.to_csv(target/'replay.csv', index=False)
    for mapping in ('source_sha256', 'input_sha256'):
        if any(sha(p) != digest for p, digest in protocol[mapping].items()):
            raise ValueError(f'Protected {mapping} changed')
    if sha(out/'protocol.json') != protocol_hash:
        raise ValueError('Protocol changed')
    checks = dict(completed_at=now(), final_n=len(prices), xgboost_models=20,
        control_models=16, all_controls_exact=True, all_reload_exact=True,
        raw_final_feature_replay_exact=True, unchanged_other_branches=True,
        submission_errors=errors, protected_files_unchanged=True)
    write_json(out/'checks.json', checks)
    write_report(out, comparison, checks)
    log(f'完成 {out}')


if __name__ == '__main__':
    main()
