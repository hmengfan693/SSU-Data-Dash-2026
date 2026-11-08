"""Change only house-type representation and select on earlier windows."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.xgboost.models import PARAMS, XGBoostPriceModel
from experiments.xgboost.run_experiment import (
    REPO, WINDOWS, FINAL_CUTOFF, TARGETS, BASELINE, CANDIDATE as OLD_CANDIDATE,
    load_designs, archived_baseline, validate_provenance, load_data, sha, now,
    log, write_json, score_predictions, metrics, HistoryFeatures,
    ProcessingFeatures, assemble, text_features, LOT_COLUMNS)
from .models import OneHotPriceModel
from .selection import choose, NATIVE, ONEHOT

HERE = Path(__file__).resolve().parent
NATIVE_RUN = REPO/'experiments/xgboost/artifacts/run_2026-10-20_v1'


def validate_native_archive():
    validate_provenance()
    if not (NATIVE_RUN/'independent_verification.json').is_file():
        raise ValueError('Verified native experiment required')
    protocol = json.loads((NATIVE_RUN/'protocol.json').read_text())
    for mapping in ('source_sha256', 'input_sha256'):
        for path, digest in protocol[mapping].items():
            if sha(path) != digest:
                raise ValueError(f'Native experiment provenance changed: {path}')
    return protocol


def fit_window(raw, test, cutoff, end, out):
    homes, designs = load_designs(raw, test, cutoff, end)
    folder = out/'windows'/cutoff
    folder.mkdir(parents=True, exist_ok=False)
    control = archived_baseline(homes, cutoff, final=end is None)
    original = pd.read_csv(NATIVE_RUN/'windows'/cutoff/'predictions_unscored.csv')
    original = original.loc[original.candidate.eq(OLD_CANDIDATE)].set_index('ml_num')
    original_control = pd.read_csv(NATIVE_RUN/'windows'/cutoff/'baseline_predictions.csv').set_index('ml_num')
    np.testing.assert_array_equal(control.predicted_price,
        original_control.predicted_price.reindex(control.ml_num))
    native = control.copy()
    native['predicted_price'] = original.predicted_price.reindex(native.ml_num).to_numpy()
    changed = control.copy()
    audits = []
    for kind, (train, valid, xt, xv) in designs.items():
        if not (train.date_sold < pd.Timestamp(cutoff)).all() or set(train.ml_num) & set(valid.ml_num):
            raise ValueError('Invalid temporal split')
        old = XGBoostPriceModel.load(NATIVE_RUN/'windows'/cutoff/kind/OLD_CANDIDATE)
        np.testing.assert_array_equal(old.predict(xv), original.predicted_price.reindex(valid.ml_num))
        log(f'训练房型独热 {cutoff}/{kind} n={len(train)}')
        model = OneHotPriceModel().fit(xt, train.sold_price)
        price = model.predict(xv)
        path = folder/kind/ONEHOT
        model.save(path, cutoff_exclusive=cutoff, training_n=len(train),
            max_train_sale=str(train.date_sold.max()), target_type=kind,
            training_types=sorted(train.house_type_name.unique()))
        np.testing.assert_array_equal(price, OneHotPriceModel.load(path).predict(xv))
        pd.DataFrame(dict(ml_num=valid.ml_num, predicted_price=price)).to_csv(path/'predictions_unscored.csv', index=False)
        mask = changed.house_type_name.eq(kind)
        update = pd.Series(price, index=valid.ml_num)
        changed.loc[mask, 'predicted_price'] = update.reindex(changed.loc[mask, 'ml_num']).to_numpy()
        audits.append(dict(target=kind, training_n=len(train),
            raw_features=len(xt.columns), encoded_features=len(model.encode(xt.head()).columns),
            training_type_vocabulary=model.categories['house_type_name'],
            native_archive_replayed_exact=True, onehot_reload_exact=True))
        log(f'完成房型独热 {cutoff}/{kind}')
    unchanged = ~homes.house_type_name.isin(TARGETS).to_numpy()
    for rows in (native, changed):
        np.testing.assert_array_equal(rows.predicted_price.to_numpy()[unchanged],
                                      control.predicted_price.to_numpy()[unchanged])
    parts = []
    for name, rows in [(BASELINE, control), (NATIVE, native), (ONEHOT, changed)]:
        rows = rows.copy()
        rows['candidate'], rows['cutoff_exclusive'] = name, cutoff
        rows['protocol'] = 'final' if end is None else ('check' if cutoff == '2026-02-01' else 'development')
        parts.append(rows)
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(folder/'predictions_unscored.csv', index=False)
    write_json(folder/'audit.json', dict(models=audits, unchanged_other_branches=True))
    return combined


def fresh_final_predict(raw, test, run):
    reference = raw.sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    hidden = data.hide_answers(test, 'off_market').reset_index(drop=True)
    history = HistoryFeatures(reference, FINAL_CUTOFF)
    base = history.transform(hidden)
    target = hidden.loc[hidden.house_type_name.isin(TARGETS[:2])].reset_index(drop=True)
    provider = ProcessingFeatures(reference, FINAL_CUTOFF)
    processed = assemble(history.transform(target), provider.history(target),
        provider.land(target), text_features(target), 'all_three')
    result = archived_baseline(test, FINAL_CUTOFF, final=True).set_index('ml_num')
    for kind in TARGETS:
        homes, design = (target, processed) if kind in TARGETS[:2] else (hidden, base)
        mask = homes.house_type_name.eq(kind)
        x = design.loc[mask]
        if kind == 'Condo Townhouse':
            x = x.drop(columns=list(LOT_COLUMNS))
        model = OneHotPriceModel.load(Path(run)/'windows'/FINAL_CUTOFF/kind/ONEHOT)
        result.loc[homes.loc[mask, 'ml_num'], 'predicted_price'] = model.predict(x)
    return result.reindex(test.ml_num).reset_index()[['ml_num', 'predicted_price']]


def comparisons(scored):
    result = []
    dimensions = [('overall', ['protocol']), ('type', ['protocol', 'house_type_name']),
        ('window', ['protocol', 'cutoff_exclusive']), ('month', ['protocol', 'sold_month']),
        ('type_window', ['protocol', 'cutoff_exclusive', 'house_type_name'])]
    for level, keys in dimensions:
        for key, group in scored.groupby(keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            a = group.loc[group.candidate.eq(BASELINE)].set_index('ml_num').sort_index()
            baseline_metrics = metrics(a)
            for candidate in (NATIVE, ONEHOT):
                b = group.loc[group.candidate.eq(candidate)].set_index('ml_num').sort_index()
                if not a.index.equals(b.index):
                    raise ValueError('Paired comparison ID mismatch')
                values = metrics(b)
                row = dict(level=level, **dict(zip(keys, key)), candidate=candidate, n=len(a))
                for metric in ('mdape', 'mape', 'median_signed_error', 'over_30_pct'):
                    row.update({f'current_best_{metric}': baseline_metrics[metric],
                        f'candidate_{metric}': values[metric],
                        f'delta_{metric}_pp': values[metric]-baseline_metrics[metric]})
                result.append(row)
    return pd.DataFrame(result)


def write_report(out, comparison, selection):
    lines = ['# XGBoost 房型独热编码对照', '',
        '仅house_type_name改为显式全量0/1列，其余特征、四分支人群、600轮及Log-L1保持不变。',
        f'三个2025年开发窗口事先选择的XGBoost版本：`{selection["selected_candidate"]}`。按完整方案全局选择，没有按房型拼接。', '',
        '## 开发期版本选择（合并22529套）', '',
        '| 方案 | MdAPE | MAPE | 相对当前最优MdAPE/百分点 |', '|---|---:|---:|---:|']
    for protocol in ('development', 'check'):
        if protocol == 'check':
            lines += ['', '## 2026年2—5月（7773套）', '',
                '| 方案 | MdAPE | MAPE | 相对当前最优MdAPE/百分点 |', '|---|---:|---:|---:|']
        rows = comparison.loc[comparison.level.eq('overall') & comparison.protocol.eq(protocol)]
        first = rows.iloc[0]
        lines.append(f'| 当前最优all_three＋既有路由 | {first.current_best_mdape:.6f}% | {first.current_best_mape:.6f}% | 0 |')
        for _, row in rows.iterrows():
            lines.append(f'| {row.candidate} | {row.candidate_mdape:.6f}% | {row.candidate_mape:.6f}% | {row.delta_mdape_pp:+.6f} |')
    lines += ['', '## 2—5月房型对照（MdAPE）', '',
        '| 房型 | 当前最优 | 原生类别 | 独热 | 独热相对最优/百分点 |', '|---|---:|---:|---:|---:|']
    types = comparison.loc[comparison.level.eq('type') & comparison.protocol.eq('check')]
    for kind, group in types.groupby('house_type_name'):
        table = group.set_index('candidate')
        a, b = table.loc[NATIVE], table.loc[ONEHOT]
        lines.append(f'| {kind} | {b.current_best_mdape:.3f}% | {a.candidate_mdape:.3f}% | {b.candidate_mdape:.3f}% | {b.delta_mdape_pp:+.3f} |')
    check = comparison.loc[comparison.level.eq('overall') & comparison.protocol.eq('check')]
    selected = check.loc[check.candidate.eq(selection['selected_candidate'])].iloc[0]
    lines += ['', '## 结论与边界', '',
        f'冻结选择版本相对当前最优MdAPE变化为 {selected.delta_mdape_pp:+.6f} 个百分点。'+
        ('该版本没有超过当前最优，完整项目继续保留当前最优。' if selected.delta_mdape_pp >= 0 else '该版本在此累计开发回测超过旧最优，须经独立核验后记录。'),
        '- 开发期先选择、冻结selection.json后才运行本轮2—5月onehot评价；原生版该期成绩先前已知。全部窗口均为累计开发回测，隐藏测试成绩未知。',
        '- 仅house_type_name使用显式独热；house_style等仍为原生类别。全部类别词表只来自对应训练折，未知房型全零；密集数值0不是缺失。',
        '- 原生版各训练分支只有1—3类房型，而max_cat_to_onehot=4，原生库已可能采用单类别分裂；显式展开未必增加表达能力。实现依据：https://xgboost.readthedocs.io/en/release_3.1.0/tutorials/categorical.html 。',
        '- 每折训练截止日及成交池冻结条件相同；原生20模型复用且重放，新增20个onehot模型均保存重放。最终另从原始数据重建特征。',
        '- 两版3974行最终预测和按开发期选择的selected文件隔离保存；不覆盖正式提交。',
        '- 模型参数、编码映射、源码和输入哈希、预测冻结、全局选择及独立核验分别存档。', '',
        '逐月/逐窗口/偏差/MAPE见comparison_to_current_best.csv；逐套预测见all_predictions_scored.csv；独立验证见independent_verification.json。']
    (out/'实验报告.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old = validate_native_archive()
    out = args.output.resolve()
    out.mkdir(parents=True)
    best = json.loads((REPO/'CURRENT_BEST_MODEL.json').read_text())
    if best['off_market_fixed_four_months']['model_id'] != old['baseline_model']['model_id']:
        raise ValueError('Rebase the control to the updated current best')
    write_json(out/'current_best_before.json', best)
    sources = set(old['source_sha256'])
    sources.update(str(p) for p in [*HERE.glob('*.py'), HERE/'README.md', HERE/'预先固定方案.md', REPO/'tests/test_xgboost_onehot.py'])
    inputs = set(old['input_sha256'])
    inputs.update(str(p) for p in NATIVE_RUN.rglob('*') if p.is_file() and 'source_snapshot' not in p.parts)
    protocol = dict(frozen_at=now(), changed_column='house_type_name', encoding='dense_full_one_hot',
        params=PARAMS, windows=WINDOWS, final_cutoff=FINAL_CUTOFF, native_run=str(NATIVE_RUN),
        selection_rule='Pooled 2025 development MdAPE, then MAPE, then native',
        per_type_selection=False, reference_pool_updates_within_window=False,
        previously_examined_windows=True, baseline_model=best['off_market_fixed_four_months'],
        source_sha256={p: sha(p) for p in sorted(sources)}, input_sha256={p: sha(p) for p in sorted(inputs)},
        versions={name: importlib.metadata.version(name) for name in ('xgboost', 'numpy', 'pandas', 'lightgbm')})
    write_json(out/'protocol.json', protocol)
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for source in sources:
        path = Path(source)
        shutil.copy2(path, snapshot/str(path.relative_to(REPO)).replace('/', '__'))
    raw, test = load_data(args.data_dir)
    parts = [fit_window(raw, test, c, e, out) for c, e in WINDOWS[:-1]]
    development = pd.concat(parts, ignore_index=True)
    development.to_csv(out/'development_predictions_unscored.csv', index=False)
    dev_scored, _ = score_predictions(development, raw)
    selection = choose(dev_scored.loc[dev_scored.candidate.isin([NATIVE, ONEHOT])])
    selection.update(frozen_at=now(), development_predictions_sha256=sha(out/'development_predictions_unscored.csv'))
    write_json(out/'selection.json', selection)
    selection_hash = sha(out/'selection.json')
    log(f'开发期全局选择已冻结：{selection["selected_candidate"]}')
    write_json(out/'check_start.json', dict(started_at=now(), selection_sha256=selection_hash))
    parts.append(fit_window(raw, test, *WINDOWS[-1], out))
    backtest = pd.concat(parts, ignore_index=True)
    backtest.to_csv(out/'all_predictions_unscored.csv', index=False)
    write_json(out/'prediction_freeze.json', dict(frozen_at=now(), protocol_sha256=sha(out/'protocol.json'),
        selection_sha256=selection_hash, unscored_sha256=sha(out/'all_predictions_unscored.csv')))
    scored, scores = score_predictions(backtest, raw)
    scored.to_csv(out/'all_predictions_scored.csv', index=False)
    scores.to_csv(out/'scores.csv', index=False)
    comparison = comparisons(scored)
    comparison.to_csv(out/'comparison_to_current_best.csv', index=False)
    final = fit_window(raw, test, FINAL_CUTOFF, None, out)
    target = out/'final_predictions'
    target.mkdir()
    for name, filename in [(NATIVE, 'submission_off_market_native.csv'),
                            (ONEHOT, 'submission_off_market_onehot.csv'),
                            (selection['selected_candidate'], 'submission_off_market_selected.csv')]:
        rows = final.loc[final.candidate.eq(name), ['ml_num', 'predicted_price']].reset_index(drop=True)
        rows.to_csv(target/filename, index=False)
        errors = check_submission.check_file(target/filename, test.ml_num)
        if errors:
            raise ValueError(errors)
    pd.testing.assert_frame_equal(pd.read_csv(target/'submission_off_market_onehot.csv'), fresh_final_predict(raw, test, out))
    for mapping in ('source_sha256', 'input_sha256'):
        if any(sha(p) != digest for p, digest in protocol[mapping].items()):
            raise ValueError(f'Protected {mapping} changed')
    if sha(out/'selection.json') != selection_hash:
        raise ValueError('Selection changed after check started')
    write_json(out/'checks.json', dict(completed_at=now(), onehot_models=20, native_models_replayed=20,
        all_reload_exact=True, raw_final_feature_replay_exact=True, final_n=len(test), protected_hashes_unchanged=True))
    write_report(out, comparison, selection)
    log(f'完成 {out}')


if __name__ == '__main__':
    main()
