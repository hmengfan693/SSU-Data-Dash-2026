"""Fit every predeclared variant before scoring; never select by scores."""
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
from experiments.plan_a.features import FORBIDDEN, HistoryFeatures
from experiments.plan_a.models import PARAMS
from experiments.plan_a.run_experiment import DEV, metrics
from experiments.property_groups.models import GroupModel
from .features import CONFIG, VARIANTS, ProcessingFeatures, assemble, text_features

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCHIVE = REPO/'experiments/residual_refinement/artifacts/run_2026-10-18_v2'
TARGETS = ('Detached', 'Semi-Detached')
LAND = ('Detached', 'Semi-Detached', 'Freehold Townhouse')
CHECK = ('2026-02-01', '2026-06-01')
FINAL_CUTOFF = '2026-06-01'


def now():
    return datetime.now(ZoneInfo('America/Toronto')).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def log(message):
    print(f'[{now()}] {message}', flush=True)


def load_data(folder):
    raw = pd.read_parquet(Path(folder)/'listings.parquet')
    test = pd.read_parquet(Path(folder)/'test_off_market.parquet')
    for column in ('date_listed', 'date_sold'):
        raw[column] = pd.to_datetime(raw[column])
    test['date_listed'] = pd.to_datetime(test.date_listed)
    if raw.ml_num.duplicated().any() or test.ml_num.duplicated().any() or set(raw.ml_num) & set(test.ml_num):
        raise ValueError('Input IDs must be unique and disjoint')
    if not (raw.date_sold < pd.Timestamp(FINAL_CUTOFF)).all():
        raise ValueError('Only labels through May 2026 allowed')
    return raw, test


def family_mask(homes, target):
    types = LAND if target == 'Detached' else LAND[1:]
    return homes.house_type_name.isin(types).to_numpy()


def source_files():
    return [*HERE.glob('*.py'), HERE/'预先固定方案.md', REPO/'tests/test_data_processing_round1.py',
            REPO/'experiments/plan_a/features.py', REPO/'experiments/plan_a/models.py',
            REPO/'experiments/plan_a/run_experiment.py',
            REPO/'experiments/property_groups/models.py', REPO/'data.py', REPO/'check_submission.py']


def prepare(raw, homes, cutoff, folder):
    reference = raw.loc[raw.date_sold < pd.Timestamp(cutoff)].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    train = reference.loc[reference.house_type_name.isin(LAND)].reset_index(drop=True)
    hidden = data.hide_answers(homes, 'off_market').reset_index(drop=True)
    folder.mkdir(parents=True, exist_ok=False)
    log(f'构造处理数据 cutoff={cutoff} reference={len(reference)} land_train={len(train)} predict={len(homes)}')
    base = HistoryFeatures(reference, cutoff)
    provider = ProcessingFeatures(reference, cutoff)
    blocks = {}
    coverage = []
    for label, h in [('train', data.hide_answers(train, 'off_market')), ('predict', hidden)]:
        arrays = [base.transform(h), provider.history(h), provider.land(h), text_features(h)]
        blocks[label] = arrays
        h[['ml_num', 'house_type_name', 'date_listed']].to_csv(folder/(label+'_ids.csv'), index=False)
        for name, x in zip(('base', 'history', 'land', 'text'), arrays):
            if len(x) != len(h) or FORBIDDEN.intersection(x.columns):
                raise ValueError('Invalid feature block')
            x.to_parquet(folder/f'{label}_{name}.parquet', index=False)
            for column in x:
                if name != 'base':
                    coverage.append({'dataset': label, 'block': name, 'feature': column,
                                     'n': len(x), 'missing_n': int(x[column].isna().sum())})
    pd.DataFrame(coverage).to_csv(folder/'feature_coverage.csv', index=False)
    write_json(folder/'audit.json', {'cutoff_exclusive': cutoff, 'max_reference_sale': str(reference.date_sold.max()),
        'reference_n': len(reference), 'land_train_n': len(train), 'predict_n': len(homes),
        'query_labels_hidden': True, 'query_date_sold_used': False,
        'history_query': '[min(listed, cutoff)-window, min(listed, cutoff))',
        'lot_query': CONFIG['lot_query'], 'feature_config': CONFIG})
    return train, blocks


def prediction_rows(homes, prices, cutoff, variant, protocol):
    p = homes[['ml_num', 'house_type_name', 'community', 'house_style']].reset_index(drop=True).copy()
    p['predicted_price'] = prices
    p['cutoff_exclusive'] = cutoff
    p['variant'] = variant
    p['protocol'] = protocol
    p['listed_month'] = pd.to_datetime(homes.date_listed).dt.strftime('%Y-%m').to_numpy()
    return p


def check_archived_control(predictions, cutoff, archive):
    source = pd.read_csv(archive/('check_candidates.csv' if cutoff == CHECK[0] else 'development_candidates.csv'))
    for target, candidate in [('Detached', 'shared_original'), ('Semi-Detached', 'previous')]:
        expected = source.loc[source.cutoff_exclusive.eq(cutoff) & source.house_type_name.eq(target) & source.candidate.eq(candidate)]
        control = predictions.loc[predictions.variant.eq('baseline') & predictions.house_type_name.eq(target)]
        aligned = control[['ml_num', 'predicted_price']].merge(expected[['ml_num', 'predicted_price']], on='ml_num',
            validate='one_to_one', suffixes=('_new', '_old'))
        if len(aligned) != len(control) or len(aligned) != len(expected) or not np.array_equal(aligned.predicted_price_new, aligned.predicted_price_old):
            raise ValueError(f'Baseline does not reproduce archived {target} at {cutoff}')


def fit_window(raw, homes, cutoff, protocol, out, archive, final=False):
    folder = out/'windows'/cutoff
    train, blocks = prepare(raw, homes, cutoff, folder/'features')
    predictions = []
    variants = ('all_three',) if final else VARIANTS
    train_design = {v: assemble(*blocks['train'], v) for v in variants}
    valid_design = {v: assemble(*blocks['predict'], v) for v in variants}
    for target in TARGETS:
        tm = family_mask(train, target)
        vm = homes.house_type_name.eq(target).to_numpy()
        scope = 'global' if target == 'Detached' else 'attached_freehold'
        for variant in variants:
            started = time.monotonic()
            log(f'训练 cutoff={cutoff} type={target} variant={variant} n={tm.sum()} features={train_design[variant].shape[1]}')
            model = GroupModel(scope, 'lightgbm').fit(train_design[variant].loc[tm], train.sold_price.loc[tm])
            price = model.predict(valid_design[variant].loc[vm])
            model_folder = folder/target/variant
            model.save_bundle(model_folder, cutoff, int(tm.sum()))
            loaded = GroupModel.load_bundle(model_folder)
            if not np.array_equal(price, loaded.predict(valid_design[variant].loc[vm])):
                raise ValueError('Model reload mismatch')
            p = prediction_rows(homes.loc[vm], price, cutoff, variant, protocol)
            p.to_csv(model_folder/'predictions_unscored.csv', index=False)
            importance = model.model.booster_.feature_importance(importance_type='gain')
            pd.DataFrame({'feature': model.feature_names, 'gain': importance}).to_csv(model_folder/'feature_importance.csv', index=False)
            write_json(model_folder/'audit.json', {'training_n': int(tm.sum()), 'prediction_n': int(vm.sum()),
                'target_type': target, 'training_types': list(LAND if target == 'Detached' else LAND[1:]),
                'variant': variant, 'model_params': PARAMS['lightgbm'], 'reload_exact': True,
                'max_train_sale': str(train.date_sold.loc[tm].max()), 'seconds': time.monotonic()-started})
            predictions.append(p)
            log(f'完成 {target}/{variant}，预测已保存，尚未评分')
    result = pd.concat(predictions, ignore_index=True)
    if not final:
        check_archived_control(result, cutoff, archive)
    result.to_csv(folder/'all_predictions_unscored.csv', index=False)
    return result


def score_table(predictions, raw):
    truth = raw[['ml_num', 'sold_price', 'date_sold']].copy()
    scored = predictions.merge(truth, on='ml_num', validate='many_to_one')
    if len(scored) != len(predictions):
        raise ValueError('Truth join dropped rows')
    scored['sold_month'] = pd.to_datetime(scored.date_sold).dt.strftime('%Y-%m')
    rows = []
    groupings = [('pooled', ['protocol', 'house_type_name', 'variant']),
                 ('window', ['protocol', 'cutoff_exclusive', 'house_type_name', 'variant']),
                 ('month', ['protocol', 'cutoff_exclusive', 'sold_month', 'house_type_name', 'variant'])]
    for level, keys in groupings:
        for key, g in scored.groupby(keys, sort=False):
            rows.append({'level': level, **dict(zip(keys, key)), **metrics(g)})
    return scored, pd.DataFrame(rows)


def slice_table(scored, out):
    rows = []
    for cutoff in scored.cutoff_exclusive.unique():
        folder = out/'windows'/cutoff/'features'
        ids = pd.read_csv(folder/'predict_ids.csv')
        land = pd.read_parquet(folder/'predict_land.parquet')
        text = pd.read_parquet(folder/'predict_text.parquet')
        history = pd.read_parquet(folder/'predict_history.parquet')
        slices = ids[['ml_num']].copy()
        slices['local_land_top5pct'] = np.where(land.land_context_available.eq(1),
            np.where(land.land_area_percentile.gt(.95), 'yes', 'no'), 'unavailable')
        slices['land_context_level'] = land.land_context_level.astype(int).astype(str)
        slices['mixed_original_updated'] = text.state_mixed_original_updated.astype(int).astype(str)
        slices['original_maintained'] = text.state_original_maintained.astype(int).astype(str)
        slices['reliable_90_level'] = history.info_reliable_90_level.astype(int).astype(str)
        g = scored.loc[scored.cutoff_exclusive.eq(cutoff)].merge(slices, on='ml_num', validate='many_to_one')
        for dim in slices.columns[1:]:
            for key, part in g.groupby(['protocol', 'house_type_name', 'variant', dim], sort=False):
                rows.append({'cutoff_exclusive': cutoff, 'protocol': key[0], 'house_type_name': key[1],
                    'variant': key[2], 'dimension': dim, 'value': key[3], **metrics(part)})
    return pd.DataFrame(rows)


def combine_final(target_predictions, test, archive):
    previous = pd.read_csv(archive/'final_predictions/submission_off_market.csv')
    submission = test[['ml_num']].merge(previous, on='ml_num', validate='one_to_one')
    if len(submission) != len(test):
        raise ValueError('Archived final coverage mismatch')
    target = test.house_type_name.isin(TARGETS).to_numpy()
    updates = target_predictions.set_index('ml_num').predicted_price
    if set(updates.index) != set(test.loc[target, 'ml_num']) or updates.index.duplicated().any():
        raise ValueError('Target coverage mismatch')
    submission.loc[target, 'predicted_price'] = updates.reindex(test.loc[target, 'ml_num']).to_numpy()
    if not np.array_equal(submission.loc[~target, 'predicted_price'], previous.set_index('ml_num').predicted_price.reindex(test.loc[~target, 'ml_num'])):
        raise ValueError('Non-target predictions changed')
    return submission


def infer_saved(raw, test, run, archive=ARCHIVE):
    reference = raw.loc[raw.date_sold < pd.Timestamp(FINAL_CUTOFF)].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    h = data.hide_answers(test.loc[test.house_type_name.isin(TARGETS)], 'off_market').reset_index(drop=True)
    provider = ProcessingFeatures(reference, FINAL_CUTOFF)
    x = assemble(HistoryFeatures(reference, FINAL_CUTOFF).transform(h), provider.history(h), provider.land(h), text_features(h), 'all_three')
    parts = []
    for target in TARGETS:
        mask = h.house_type_name.eq(target).to_numpy()
        model = GroupModel.load_bundle(Path(run)/'windows'/FINAL_CUTOFF/target/'all_three')
        parts.append(pd.DataFrame({'ml_num': h.loc[mask, 'ml_num'], 'predicted_price': model.predict(x.loc[mask])}))
    return combine_final(pd.concat(parts, ignore_index=True), test, archive)


def report(out, scores, checks):
    lines = ['# 第一轮数据处理实验：固定方案的真实结果', '',
        '本轮优先改变数据处理。LightGBM 参数、训练范围和验证窗口均不变。五个候选及 all_three 的候选身份在训练评分前已固定；没有根据分数调整特征、择优或回退。', '',
        '处理包括历史信息年龄/覆盖/可靠回退、过去属性参考中的土地与临街宽相对位置、分句与否定处理后的多维房况。具体规则见同目录预先固定方案.md、protocol.json 和 source_snapshot。', '',
        '独立屋使用三类有土地房型共享模型；半独立屋沿用半独立屋与自有产权联排合并训练。最终只更新独立屋和半独立屋，其他房型沿用旧冻结预测。', '',
        '## 全部固定候选', '', '| 口径 | 房型 | 处理版本 | 套数 | MdAPE | 中位偏差 |',
        '|---|---|---|---:|---:|---:|']
    pooled = scores.loc[scores.level.eq('pooled')]
    for _, s in pooled.iterrows():
        lines.append(f'| {s.protocol} | {s.house_type_name} | {s.variant} | {int(s.n):,} | {s.mdape:.2f}% | {s.median_signed_error:+.2f}% |')
    lines += ['', 'baseline=原处理；history_only=仅历史可靠性；land_only=仅土地相对表示；text_only=仅文本；all_three=三项合并。不同房型的基线训练人群不同，不能跨房型把分数差解释为处理收益。', '',
        '## 预先指定的三项合并版相对基线', '', '| 口径 | 房型 | 基线 MdAPE | 三项合并 MdAPE | 变化/百分点 |', '|---|---|---:|---:|---:|']
    for protocol in ('development', 'check'):
        for target in TARGETS:
            g = pooled.loc[pooled.protocol.eq(protocol) & pooled.house_type_name.eq(target)].set_index('variant')
            before, after = g.loc['baseline', 'mdape'], g.loc['all_three', 'mdape']
            lines.append(f'| {protocol} | {target} | {before:.2f}% | {after:.2f}% | {after-before:+.2f} |')
    lines += ['', '变化为三项合并减基线，负值为改善，正值为退步。单项消融说明处理与预测变化的关系，不能将三个差值相加，也不证明切片误差的因果来源。', '',
        '## 验证与交付', '',
        f'- 两种房型四个窗口的 baseline 均逐套复现存档；新模型保存重载预测完全一致。',
        f'- 已完成40个回测模型及2个截至5月的最终模型。最终隔离候选为 {checks["final_n"]:,} 套，格式校验通过。',
        '- 最终 all_three 候选是事前指定方案，不代表所有候选里分数最好。原始输入、旧模型、旧实验与旧提交哈希核对不变。',
        '- 新特征未输入待预测成交价、成交日、挂牌价或外部估价；未用真实未来成交日期作为特征。',
        '- 所有回测窗口都已被项目查看过，属于累计开发回测。6—9月没有真实成交价，不能报告隐藏测试误差。',
        '- 词句规则仍有语言覆盖限制；土地为宽×深代理；固定四个月预测中的历史信息持续陈旧，新增可靠性描述不等于知道未来市场趋势。', '',
        '逐月结果见 scores.csv；预先固定的切片见 error_slices.csv；逐套预测见 all_predictions_scored.csv；数据覆盖见各窗口 features/feature_coverage.csv；独立复核见 verification.json。', '']
    (out/'实验报告.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    args = parser.parse_args()
    out, archive = args.output.resolve(), args.archive.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = [args.data_dir/'listings.parquet', args.data_dir/'test_off_market.parquet']
    protected = [*inputs, *[p for p in archive.rglob('*') if p.is_file()],
                 *[p for p in (REPO/'submissions').rglob('*') if p.is_file()]]
    hashes = {str(p.resolve()): sha(p) for p in protected}
    sources = source_files()
    code_hashes = {str(p.relative_to(REPO)): sha(p) for p in sources}
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for p in sources:
        shutil.copy2(p, snapshot/str(p.relative_to(REPO)).replace('/', '__'))
    shutil.copy2(HERE/'预先固定方案.md', out/'预先固定方案.md')
    protocol = {'frozen_at_Toronto': now(), 'feature_config': CONFIG, 'variants': list(VARIANTS),
        'designated_candidate': 'all_three', 'selection_by_scores': False, 'parameter_tuning': False,
        'development_windows': DEV, 'check_window': CHECK, 'final_cutoff': FINAL_CUTOFF,
        'params': PARAMS['lightgbm'], 'source_hashes': code_hashes,
        'archive': str(archive), 'protected_sha256': hashes,
        'versions': {p: importlib.metadata.version(p) for p in ('lightgbm', 'numpy', 'pandas', 'scipy', 'pyarrow')},
        'all_windows_previously_examined': True}
    write_json(out/'protocol.json', protocol)
    frozen_hash = sha(out/'protocol.json')
    log(f'方案已冻结 sha256={frozen_hash}；本轮无择优或分数驱动修改')
    raw, test = load_data(args.data_dir)
    parts = []
    for protocol_name, windows in [('development', DEV), ('check', [CHECK])]:
        for cutoff, end in windows:
            homes = raw.loc[(raw.date_sold >= pd.Timestamp(cutoff)) & (raw.date_sold < pd.Timestamp(end)) & raw.house_type_name.isin(TARGETS)].reset_index(drop=True)
            parts.append(fit_window(raw, homes, cutoff, protocol_name, out, archive))
    # All trial predictions exist before any new metrics are calculated.
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out/'all_predictions_unscored.csv', index=False)
    write_json(out/'prediction_freeze.json', {'frozen_at_Toronto': now(), 'protocol_sha256': frozen_hash,
        'unscored_predictions_sha256': sha(out/'all_predictions_unscored.csv'), 'all_40_models_predicted_before_scoring': True})
    log('全部40个回测模型已完成预测，开始统一评分；处理规则及候选身份不改变')
    scored, scores = score_table(predictions, raw)
    scored.to_csv(out/'all_predictions_scored.csv', index=False)
    scores.to_csv(out/'scores.csv', index=False)
    slice_table(scored, out).to_csv(out/'error_slices.csv', index=False)
    target_test = test.loc[test.house_type_name.isin(TARGETS)].reset_index(drop=True)
    final = fit_window(raw, target_test, FINAL_CUTOFF, 'final', out, archive, final=True)
    submission = combine_final(final, test, archive)
    final_folder = out/'final_predictions'
    final_folder.mkdir()
    submission.to_csv(final_folder/'submission_off_market.csv', index=False)
    problems = check_submission.check_file(final_folder/'submission_off_market.csv', test.ml_num)
    if problems:
        raise ValueError(problems)
    log('重建最终特征并从保存模型独立重放')
    replay = infer_saved(raw, test, out, archive)
    if not submission.equals(replay):
        raise ValueError('Saved-model replay mismatch')
    if any(sha(p) != digest for p, digest in hashes.items()):
        raise ValueError('Protected input/archive/submission changed')
    if any(sha(REPO/p) != digest for p, digest in code_hashes.items()) or sha(out/'protocol.json') != frozen_hash:
        raise ValueError('Frozen source/protocol changed during run')
    checks = {'completed_at_Toronto': now(), 'protocol_sha256': frozen_hash, 'final_n': len(submission),
        'new_target_n': int(test.house_type_name.isin(TARGETS).sum()), 'non_target_unchanged': True,
        'all_baselines_reproduced': True, 'all_models_reload_exact': True, 'fresh_saved_replay_exact': True,
        'submission_errors': problems, 'protected_files_unchanged': len(hashes), 'sources_unchanged': True,
        'model_count': 42, 'prediction_count': len(predictions), 'score_row_count': len(scores),
        'designated_candidate': 'all_three', 'score_based_selection_performed': False}
    write_json(out/'checks.json', checks)
    report(out, scores, checks)
    log(f'完成。结果与处理方案：{out}')


if __name__ == '__main__':
    main()
