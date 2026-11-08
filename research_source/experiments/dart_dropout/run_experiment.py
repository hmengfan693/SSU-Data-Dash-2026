"""Paper-informed, fixed 1% DART on all current off-market LightGBM branches."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.data_processing_round1.features import ProcessingFeatures, assemble, text_features
from experiments.data_processing_round1.run_experiment import load_data, family_mask
from experiments.plan_a.features import HistoryFeatures, FORBIDDEN
from experiments.plan_a.run_experiment import DEV, metrics
from experiments.property_groups.models import LOT_COLUMNS
from experiments.residual_refinement.selection import route
from .models import DropoutModel, DROP_SEEDS, PRIMARY_SEED, parameters

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROCESSING = REPO/'experiments/data_processing_round1/artifacts/run_2026-10-19_v1'
RESIDUAL = REPO/'experiments/residual_refinement/artifacts/run_2026-10-18_v2'
TARGETS = ('Detached', 'Semi-Detached', 'Freehold Townhouse', 'Condo Townhouse')
LAND = TARGETS[:3]
WINDOWS = [*DEV, ('2026-02-01', '2026-06-01')]
FINAL_CUTOFF = '2026-06-01'


def now():
    return datetime.now(ZoneInfo('America/Toronto')).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def log(message):
    print(f'[{now()}] {message}', flush=True)


def source_files():
    paths = [*HERE.glob('*.py'), HERE/'预先固定方案.md', REPO/'tests/test_dart_dropout.py',
             REPO/'data.py', REPO/'check_submission.py']
    for package in ('plan_a', 'property_groups', 'data_processing_round1', 'residual_refinement'):
        paths.extend((REPO/'experiments'/package).glob('*.py'))
    return sorted(set(paths))


def input_files(data_dir):
    paths = [Path(data_dir)/'listings.parquet', Path(data_dir)/'test_off_market.parquet',
             PROCESSING/'protocol.json', PROCESSING/'all_predictions_scored.csv',
             PROCESSING/'final_predictions/submission_off_market.csv', RESIDUAL/'source_manifest.json',
             RESIDUAL/'selection.json', RESIDUAL/'development_candidates.csv',
             RESIDUAL/'check_selected_predictions.csv']
    for cutoff in [c for c, _ in WINDOWS]+[FINAL_CUTOFF]:
        paths.extend((PROCESSING/'windows'/cutoff/'features').glob('*'))
        cache = RESIDUAL/'bases'/cutoff
        paths.extend(cache/name for name in ('train_base.parquet', 'valid_base.parquet',
                                           'train_ids.csv', 'predict_ids.csv', 'feature_audit.json'))
    if (REPO/'submissions').exists():
        paths.extend((REPO/'submissions').glob('*.csv'))
    return sorted(set(p for p in paths if p.is_file()))


def validate_provenance():
    for path, key, relative in [(PROCESSING/'protocol.json', 'source_hashes', True),
                                 (RESIDUAL/'source_manifest.json', 'source_sha256', False)]:
        manifest = json.loads(path.read_text())
        for name, digest in manifest[key].items():
            file = REPO/name if relative else Path(name)
            if not file.is_file() or sha(file) != digest:
                raise ValueError(f'Archived feature/source provenance mismatch: {file}')
    # Confirm the current raw data still matches the cache-producing experiment.
    protected = json.loads((PROCESSING/'protocol.json').read_text())['protected_sha256']
    for name, digest in protected.items():
        path = Path(name)
        if path.suffix == '.parquet' and path.parent == REPO.parent/'data' and sha(path) != digest:
            raise ValueError(f'Raw input changed since feature generation: {path}')


def split(raw, test, cutoff, end):
    train = raw.loc[raw.date_sold < pd.Timestamp(cutoff)].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    homes = (test if end is None else raw.loc[(raw.date_sold >= pd.Timestamp(cutoff)) &
                                             (raw.date_sold < pd.Timestamp(end))]).reset_index(drop=True)
    if train.empty or homes.empty or set(train.ml_num) & set(homes.ml_num):
        raise ValueError('Invalid temporal split')
    return train, homes


def assert_ids(path, homes):
    ids = pd.read_csv(path)
    if not np.array_equal(ids.ml_num, homes.ml_num):
        raise ValueError(f'Cache ID order mismatch: {path}')
    if not np.array_equal(pd.to_datetime(ids.date_listed), pd.to_datetime(homes.date_listed)):
        raise ValueError(f'Cache date mismatch: {path}')


def load_designs(raw, test, cutoff, end):
    train, homes = split(raw, test, cutoff, end)
    base_dir = RESIDUAL/'bases'/cutoff
    assert_ids(base_dir/'train_ids.csv', train)
    assert_ids(base_dir/'predict_ids.csv', homes)
    base_train = pd.read_parquet(base_dir/'train_base.parquet')
    base_valid = pd.read_parquet(base_dir/'valid_base.parquet')
    land_train = train.loc[train.house_type_name.isin(LAND)].reset_index(drop=True)
    target_homes = homes.loc[homes.house_type_name.isin(TARGETS[:2])].reset_index(drop=True)
    proc_dir = PROCESSING/'windows'/cutoff/'features'
    assert_ids(proc_dir/'train_ids.csv', land_train)
    assert_ids(proc_dir/'predict_ids.csv', target_homes)
    processed = []
    for label in ('train', 'predict'):
        blocks = [pd.read_parquet(proc_dir/f'{label}_{name}.parquet') for name in ('base', 'history', 'land', 'text')]
        processed.append(assemble(*blocks, 'all_three'))
    if len(base_train) != len(train) or len(base_valid) != len(homes):
        raise ValueError('Base cache shape mismatch')
    # The two source caches must agree on their shared, unmodified baseline block.
    for label, expected in [('train', base_train.loc[train.house_type_name.isin(LAND)].reset_index(drop=True)),
                            ('predict', base_valid.loc[homes.house_type_name.isin(TARGETS[:2])].reset_index(drop=True))]:
        pd.testing.assert_frame_equal(pd.read_parquet(proc_dir/f'{label}_base.parquet'), expected)
    designs = {}
    for target in TARGETS:
        if target in TARGETS[:2]:
            tm = family_mask(land_train, target)
            vm = target_homes.house_type_name.eq(target).to_numpy()
            tr, va = land_train.loc[tm], target_homes.loc[vm]
            xt, xv = processed[0].loc[tm], processed[1].loc[vm]
        else:
            tm = (train.house_type_name.isin(LAND) if target == 'Freehold Townhouse' else train.house_type_name.eq(target)).to_numpy()
            vm = homes.house_type_name.eq(target).to_numpy()
            tr, va = train.loc[tm], homes.loc[vm]
            xt, xv = base_train.loc[tm], base_valid.loc[vm]
            if target == 'Condo Townhouse':
                xt, xv = xt.drop(columns=list(LOT_COLUMNS)), xv.drop(columns=list(LOT_COLUMNS))
        if FORBIDDEN.intersection(xt.columns) or list(xt) != list(xv):
            raise ValueError('Invalid design columns')
        designs[target] = (tr.reset_index(drop=True), va.reset_index(drop=True),
                           xt.reset_index(drop=True), xv.reset_index(drop=True))
    return homes, designs


def archived_baseline(homes, cutoff, final=False):
    if final:
        rows = pd.read_csv(PROCESSING/'final_predictions/submission_off_market.csv')
    else:
        if cutoff == '2026-02-01':
            rows = pd.read_csv(RESIDUAL/'check_selected_predictions.csv')
        else:
            candidates = pd.read_csv(RESIDUAL/'development_candidates.csv')
            choices = json.loads((RESIDUAL/'selection.json').read_text())['choices']
            rows = route(candidates.loc[candidates.cutoff_exclusive.eq(cutoff)], choices)
        updates = pd.read_csv(PROCESSING/'all_predictions_scored.csv')
        updates = updates.loc[updates.cutoff_exclusive.eq(cutoff) & updates.variant.eq('all_three')]
        replacement = updates.set_index('ml_num').predicted_price
        rows = rows.copy()
        mask = rows.house_type_name.isin(TARGETS[:2])
        rows.loc[mask, 'predicted_price'] = replacement.reindex(rows.loc[mask, 'ml_num']).to_numpy()
    aligned = homes[['ml_num', 'house_type_name', 'date_listed']].merge(rows[['ml_num', 'predicted_price']],
                  on='ml_num', how='left', validate='one_to_one')
    if len(aligned) != len(homes) or not np.isfinite(aligned.predicted_price).all():
        raise ValueError('Archived baseline coverage mismatch')
    return aligned


def fit_window(raw, test, cutoff, end, out):
    homes, designs = load_designs(raw, test, cutoff, end)
    folder = out/'windows'/cutoff
    folder.mkdir(parents=True, exist_ok=False)
    control = archived_baseline(homes, cutoff, final=end is None)
    control.to_csv(folder/'baseline_predictions.csv', index=False)
    results = {seed: control.copy() for seed in DROP_SEEDS}
    seeds = (PRIMARY_SEED,) if end is None else DROP_SEEDS
    model_audits = []
    for target, (tr, va, xt, xv) in designs.items():
        for seed in ((PRIMARY_SEED,) if end is None else (None, *seeds)):
            name = 'gbdt' if seed is None else f'dart_seed_{seed}'
            log(f'训练 {cutoff} {target}/{name} rows={len(tr)} features={len(xt.columns)}')
            started = time.monotonic()
            model = DropoutModel(seed).fit(xt, tr.sold_price)
            predicted = model.predict(xv)
            model_dir = folder/target/name
            model.save(model_dir, cutoff_exclusive=cutoff, training_n=len(tr),
                       max_train_sale=str(tr.date_sold.max()), target_type=target,
                       training_types=sorted(tr.house_type_name.unique().tolist()))
            if not np.array_equal(predicted, DropoutModel.load(model_dir).predict(xv)):
                raise ValueError('Model reload mismatch')
            expected = control.set_index('ml_num').predicted_price.reindex(va.ml_num).to_numpy()
            if seed is None:
                if not np.array_equal(expected, predicted):
                    raise ValueError(f'GBDT did not reproduce current control: {cutoff}/{target}')
            else:
                update = pd.Series(predicted, index=va.ml_num)
                mask = results[seed].house_type_name.eq(target)
                results[seed].loc[mask, 'predicted_price'] = update.reindex(results[seed].loc[mask, 'ml_num']).to_numpy()
            pd.DataFrame({'ml_num': va.ml_num, 'predicted_price': predicted}).to_csv(model_dir/'predictions_unscored.csv', index=False)
            tree_info = model.booster.dump_model()['tree_info']
            model_audits.append(dict(target=target, candidate=name, training_n=len(tr), features=len(xt.columns),
                iterations=model.booster.current_iteration(), reload_exact=True, baseline_exact=seed is None,
                seconds=time.monotonic()-started,
                trees_with_nonstandard_shrinkage=sum(not np.isclose(t['shrinkage'], model.params['learning_rate']) for t in tree_info[1:])))
            log(f'完成 {cutoff} {target}/{name}，未评分')
    parts = []
    for candidate, prediction in [('gbdt', control), *[(f'dart_seed_{seed}', results[seed]) for seed in seeds]]:
        unchanged = ~homes.house_type_name.isin(TARGETS).to_numpy()
        if not np.array_equal(prediction.predicted_price.to_numpy()[unchanged], control.predicted_price.to_numpy()[unchanged]):
            raise ValueError('CatBoost branch changed')
        prediction = prediction.copy()
        prediction['candidate'], prediction['cutoff_exclusive'] = candidate, cutoff
        prediction['protocol'] = 'final' if end is None else ('check' if cutoff == '2026-02-01' else 'development')
        parts.append(prediction)
    result = pd.concat(parts, ignore_index=True)
    result.to_csv(folder/'predictions_unscored.csv', index=False)
    write_json(folder/'audit.json', {'models': model_audits, 'catboost_branches_unchanged': True})
    return result


def score_predictions(predictions, raw):
    truth = raw[['ml_num', 'sold_price', 'date_sold']]
    scored = predictions.merge(truth, on='ml_num', validate='many_to_one')
    if len(scored) != len(predictions):
        raise ValueError('Missing truth')
    scored['sold_month'] = pd.to_datetime(scored.date_sold).dt.strftime('%Y-%m')
    rows = []
    dimensions = [('overall', ['protocol', 'candidate']),
                  ('type', ['protocol', 'candidate', 'house_type_name']),
                  ('window', ['protocol', 'candidate', 'cutoff_exclusive']),
                  ('type_window', ['protocol', 'candidate', 'cutoff_exclusive', 'house_type_name']),
                  ('month', ['protocol', 'candidate', 'sold_month'])]
    for level, keys in dimensions:
        for key, group in scored.groupby(keys, sort=False):
            rows.append(dict(level=level, **dict(zip(keys, key)), **metrics(group)))
    return scored, pd.DataFrame(rows)


def fresh_final_predict(raw, test, run):
    reference = raw.sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    hidden = data.hide_answers(test, 'off_market').reset_index(drop=True)
    base = HistoryFeatures(reference, FINAL_CUTOFF).transform(hidden)
    target = hidden.loc[hidden.house_type_name.isin(TARGETS[:2])].reset_index(drop=True)
    provider = ProcessingFeatures(reference, FINAL_CUTOFF)
    processed = assemble(HistoryFeatures(reference, FINAL_CUTOFF).transform(target),
                         provider.history(target), provider.land(target), text_features(target), 'all_three')
    result = archived_baseline(test, FINAL_CUTOFF, final=True).set_index('ml_num')
    for kind in TARGETS:
        homes = target if kind in TARGETS[:2] else hidden
        design = processed if kind in TARGETS[:2] else base
        mask = homes.house_type_name.eq(kind)
        x = design.loc[mask]
        if kind == 'Condo Townhouse':
            x = x.drop(columns=list(LOT_COLUMNS))
        model = DropoutModel.load(Path(run)/'windows'/FINAL_CUTOFF/kind/f'dart_seed_{PRIMARY_SEED}')
        result.loc[homes.loc[mask, 'ml_num'], 'predicted_price'] = model.predict(x)
    return result.reindex(test.ml_num).reset_index()[['ml_num', 'predicted_price']]


def write_report(out, scores, checks):
    lines = ['# 当前 LightGBM 加入 1% DART：固定方案结果', '',
        '四类 LightGBM 分支使用相同固定 dropout；两类公寓沿用冻结 CatBoost。特征、人群、Log-L1、学习率0.05和600轮不变。',
        'drop_rate=0.01，uniform_drop=True，skip_drop=0，max_drop=-1；drop_seed=42为事前指定主候选，43/44只作稳定性检查。', '',
        '采用论文树丢弃与归一化思想，1%来自论文表3的回归候选范围；保留本项目shrinkage和损失，LightGBM允许空丢弃集合，不是原文Binomial-plus-one的逐项复现。',
        '预检查发现首树携带初始价格基准时可能被丢弃而异常提前结束。初始训练集L1常数通过init_score保留在dropout之外，在推理时加回并随schema保存；普通GBDT对照也使用此处理并逐套复现原存档。此工程修正未依据新回测成绩。',
        '论文：https://proceedings.mlr.press/v38/korlakaivinayak15.pdf 。源码：https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/boosting/dart.hpp 。', '',
        '## 各种子整表结果', '', '| 口径 | 模型 | 套数 | MdAPE | MAPE | 中位偏差 |', '|---|---|---:|---:|---:|---:|']
    for _, row in scores.loc[scores.level.eq('overall')].iterrows():
        lines.append(f'| {row.protocol} | {row.candidate} | {int(row.n):,} | {row.mdape:.3f}% | {row.mape:.3f}% | {row.median_signed_error:+.3f}% |')
    lines += ['', '## 2026年2—5月：固定主候选42', '', '| 房型 | 套数 | GBDT MdAPE | DART MdAPE | 变化/百分点 |', '|---|---:|---:|---:|---:|']
    types = scores.loc[scores.level.eq('type') & scores.protocol.eq('check')]
    for kind in sorted(types.house_type_name.unique()):
        table = types.loc[types.house_type_name.eq(kind)].set_index('candidate')
        a, b = table.loc['gbdt'], table.loc['dart_seed_42']
        lines.append(f'| {kind} | {int(a.n):,} | {a.mdape:.3f}% | {b.mdape:.3f}% | {b.mdape-a.mdape:+.3f} |')
    summary = scores.loc[scores.level.eq('overall') & scores.protocol.eq('check')].set_index('candidate')
    delta = summary.loc['dart_seed_42', 'mdape']-summary.loc['gbdt', 'mdape']
    lines += ['', f'事前指定主候选总体MdAPE变化 {delta:+.3f} 个百分点。'+('本设置没有改善总体MdAPE。' if delta >= 0 else '本设置在该回测总体MdAPE较低，未来泛化尚未知。'),
        '没有根据结果改变dropout、选择种子或替换原提交；三个种子的平均分不是一个可部署模型的成绩。', '',
        '## 验证与边界', '',
        '- 三个2025年四个月窗口与2026年2—5月均已被项目查看，属于累计开发回测。模型和成交历史池四个月内均冻结。',
        '- 全部回测模型完成并冻结未评分预测后统一评分，无早停或外层标签选择。当前GBDT对照逐套复现存档。',
        '- 每个保存模型均重载核对，来源、缓存、源程序和原始数据哈希写入protocol.json；复用缓存的ID、顺序和训练截止日检查通过。',
        f'- 最终预定主候选已按截至5月标签重训，隔离整表{checks["final_n"]}行；原始数据重建特征后的预测重放一致，官方格式检查通过。',
        '- 600轮是与当前基线的固定预算对照，没有搜索DART适合的轮数；结果只适用于本设置。',
        '- 尚未声称树过度专门化是本项目已确认的误差原因，也没有把dropout视为市场趋势预测方法。', '',
        '逐月和逐窗口见scores.csv；逐套结果见all_predictions_scored.csv；参数、模型与审计见windows/。最终预测见final_predictions/submission_off_market_dart.csv。']
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
    sources = source_files()
    inputs = input_files(args.data_dir)
    protocol = dict(frozen_at=now(), windows=WINDOWS, final_cutoff=FINAL_CUTOFF,
        primary_drop_seed=PRIMARY_SEED, stability_drop_seeds=list(DROP_SEEDS), target_types=list(TARGETS),
        baseline_params=parameters(), dart_params={str(s): parameters(s) for s in DROP_SEEDS},
        parameter_search=False, score_based_selection=False, previously_examined_windows=True,
        feature_archive=str(PROCESSING), base_archive=str(RESIDUAL),
        source_sha256={str(p): sha(p) for p in sources}, input_sha256={str(p): sha(p) for p in inputs},
        versions={name: importlib.metadata.version(name) for name in ('lightgbm', 'numpy', 'pandas', 'pyarrow', 'scipy')})
    write_json(out/'protocol.json', protocol)
    protocol_hash = sha(out/'protocol.json')
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for p in sources:
        shutil.copy2(p, snapshot/str(p.relative_to(REPO)).replace('/', '__'))
    log(f'已冻结1% dropout，主种子42；protocol={protocol_hash}')
    raw, test = load_data(args.data_dir)
    parts = [fit_window(raw, test, cutoff, end, out) for cutoff, end in WINDOWS]
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out/'all_predictions_unscored.csv', index=False)
    write_json(out/'prediction_freeze.json', dict(frozen_at=now(), protocol_sha256=protocol_hash,
        unscored_sha256=sha(out/'all_predictions_unscored.csv'), all_64_backtest_models_completed=True))
    log('全部64个回测模型完成，开始统一评分')
    scored, scores = score_predictions(predictions, raw)
    scored.to_csv(out/'all_predictions_scored.csv', index=False)
    scores.to_csv(out/'scores.csv', index=False)
    final = fit_window(raw, test, FINAL_CUTOFF, None, out)
    primary = final.loc[final.candidate.eq(f'dart_seed_{PRIMARY_SEED}'), ['ml_num', 'predicted_price']].reset_index(drop=True)
    folder = out/'final_predictions'
    folder.mkdir()
    path = folder/'submission_off_market_dart.csv'
    primary.to_csv(path, index=False)
    problems = check_submission.check_file(path, test.ml_num)
    if problems:
        raise ValueError(problems)
    log('从原始数据重建最终特征并重放DART模型')
    replay = fresh_final_predict(raw, test, out)
    pd.testing.assert_frame_equal(primary, replay)
    replay.to_csv(folder/'replay.csv', index=False)
    for mapping in ('source_sha256', 'input_sha256'):
        if any(sha(p) != digest for p, digest in protocol[mapping].items()):
            raise ValueError(f'Protected {mapping} changed')
    if sha(out/'protocol.json') != protocol_hash:
        raise ValueError('Protocol changed')
    checks = dict(completed_at=now(), final_n=len(primary), model_count=68,
                  all_baselines_exact=True, all_reload_exact=True, fresh_final_replay_exact=True,
                  catboost_branches_unchanged=True, submission_errors=problems,
                  protected_files_unchanged=True, protocol_sha256=protocol_hash)
    write_json(out/'checks.json', checks)
    write_report(out, scores, checks)
    log(f'完成 {out}')


if __name__ == '__main__':
    main()
