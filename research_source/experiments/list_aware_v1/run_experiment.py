"""Freeze a full candidate using 2025 only; check Feb-May; refit through May."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import check_submission
from .features import FeatureBuilder, RatioHistory, historical_prediction, list_prices
from .models import GROUPS, RatioModel, grouped_predict

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEV = [('2025-01-01', '2025-05-01'), ('2025-05-01', '2025-09-01'),
       ('2025-09-01', '2026-01-01')]
CHECK = ('2026-02-01', '2026-06-01')
FINAL = '2026-06-01'
CANDIDATES = ['list_price', 'historical_ratio', 'global_lgbm', 'grouped_lgbm']
NAMES = {'list_price': '挂牌价基线', 'historical_ratio': '历史成交挂牌比分层校正',
         'global_lgbm': '统一 LightGBM 成交挂牌比模型', 'grouped_lgbm': '固定四组 LightGBM 成交挂牌比模型'}


def now():
    return datetime.now(timezone.utc).isoformat()


def log(value):
    print(f'[{now()}] {value}', flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_data(folder):
    folder = Path(folder)
    raw = pd.read_parquet(folder/'listings.parquet')
    test = pd.read_parquet(folder/'test_list_aware.parquet')
    for df in (raw, test):
        df['date_listed'] = pd.to_datetime(df.date_listed)
        list_prices(df)
        if df.ml_num.duplicated().any():
            raise ValueError('Duplicate input IDs')
    raw['date_sold'] = pd.to_datetime(raw.date_sold)
    if (set(raw.ml_num) & set(test.ml_num) or not (raw.date_sold < pd.Timestamp(FINAL)).all()
            or not (raw.sold_price > 0).all()):
        raise ValueError('Invalid data IDs, labels or cutoff')
    return raw, test


def metric(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if (truth.ndim != 1 or truth.shape != prediction.shape or len(truth) == 0
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()
            or not (truth > 0).all() or not (prediction > 0).all()):
        raise ValueError('Invalid scoring inputs')
    ape = np.abs(prediction-truth)/truth*100
    return dict(n=len(ape), mdape=float(np.median(ape)), mape=float(np.mean(ape)))


def score_table(predictions, truth):
    """Strict one-to-one ID alignment, paired against the original incumbent."""
    rows = []
    truth = truth[['ml_num', 'sold_price', 'house_type_name', 'date_sold']].copy()
    truth['month'] = truth.date_sold.dt.strftime('%Y-%m')
    expected = set(truth.ml_num)
    if truth.ml_num.duplicated().any():
        raise ValueError('Duplicated truth IDs')
    paired = {}
    for candidate in CANDIDATES:
        pred = predictions.loc[predictions.candidate == candidate, ['ml_num', 'predicted_price']]
        if pred.ml_num.duplicated().any() or set(pred.ml_num) != expected:
            raise ValueError('Incomplete/duplicated candidate IDs')
        paired[candidate] = truth.merge(pred, on='ml_num', validate='one_to_one').sort_values('ml_num').reset_index(drop=True)
    baseline = paired['list_price']
    for candidate, p in paired.items():
        slices = [('overall', 'all', np.ones(len(p), bool))]
        for column in ('house_type_name', 'month'):
            slices.extend((column, str(value), p[column].eq(value).to_numpy()) for value in sorted(p[column].unique()))
        for level, group, mask in slices:
            current = metric(p.sold_price.to_numpy()[mask], p.predicted_price.to_numpy()[mask])
            prior = metric(baseline.sold_price.to_numpy()[mask], baseline.predicted_price.to_numpy()[mask])
            rows.append(dict(candidate=candidate, name=NAMES[candidate], level=level, group=group,
                             **current, incumbent_name=NAMES['list_price'], incumbent_mdape=prior['mdape'],
                             delta_mdape_pp=current['mdape']-prior['mdape']))
    return pd.DataFrame(rows)


def select_candidate(predictions, truth):
    if set(predictions.cutoff_exclusive) != {c for c, _ in DEV} or set(predictions.protocol) != {'development'}:
        raise ValueError('Selection accepts only the three 2025 development windows')
    expected = truth.loc[(truth.date_sold >= '2025-01-01') & (truth.date_sold < '2026-01-01')]
    table = score_table(predictions, expected)
    overall = table.loc[table.level == 'overall'].sort_values(['mdape', 'mape', 'candidate'])
    return overall.iloc[0].candidate, table


def fit_models(train, xt, homes, xv, folder, cutoff, threads, candidates):
    result = {}
    if 'global_lgbm' in candidates:
        log(f'{cutoff} 统一模型训练：{len(train)} 套')
        model = RatioModel(threads).fit(xt, train)
        target = folder/'global_lgbm'
        model.save(target, cutoff_exclusive=cutoff, training_n=len(train))
        result['global_lgbm'] = model.predict(xv, homes)
        np.testing.assert_array_equal(result['global_lgbm'], RatioModel.load(target).predict(xv, homes))
    if 'grouped_lgbm' in candidates:
        models = {}
        for group, types in GROUPS.items():
            ix = np.flatnonzero(train.house_type_name.isin(types).to_numpy())
            if not len(ix):
                raise ValueError(f'Empty training group {group}')
            log(f'{cutoff} 分组训练 {group}：{len(ix)} 套')
            model = RatioModel(threads).fit(xt.iloc[ix], train.iloc[ix])
            target = folder/'grouped_lgbm'/group
            model.save(target, cutoff_exclusive=cutoff, training_n=len(ix), training_types=types)
            models[group] = RatioModel.load(target)
            vi = np.flatnonzero(homes.house_type_name.isin(types).to_numpy())
            if len(vi):
                np.testing.assert_array_equal(model.predict(xv.iloc[vi], homes.iloc[vi]),
                                              models[group].predict(xv.iloc[vi], homes.iloc[vi]))
        result['grouped_lgbm'] = grouped_predict(models, xv, homes)
    return result


def predict_window(raw, homes, cutoff, folder, threads, candidates=CANDIDATES):
    import data
    folder.mkdir(parents=True, exist_ok=False)
    train = raw.loc[raw.date_sold < pd.Timestamp(cutoff)].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    homes = homes.reset_index(drop=True)
    hidden = data.hide_answers(homes, 'list_aware')
    train[['ml_num', 'date_sold']].to_csv(folder/'train_ids.csv', index=False)
    hidden[['ml_num', 'date_listed']].to_csv(folder/'predict_ids.csv', index=False)
    write_json(folder/'audit.json', dict(cutoff_exclusive=cutoff, training_n=len(train),
        max_reference_sale=str(train.date_sold.max()), prediction_n=len(homes),
        query='historical sale < min(date_listed, cutoff)', reference_pool_updates=False,
        query_answers_hidden=True))
    result = {}
    if 'list_price' in candidates:
        result['list_price'] = list_prices(hidden)
    if 'historical_ratio' in candidates:
        result['historical_ratio'] = historical_prediction(hidden, RatioHistory(train, cutoff).transform(hidden))
    if set(candidates) & {'global_lgbm', 'grouped_lgbm'}:
        log(f'{cutoff} 构造特征：训练 {len(train)} 套，预测 {len(homes)} 套')
        builder = FeatureBuilder(train, cutoff)
        xt = builder.transform(data.hide_answers(train, 'list_aware'))
        xv = builder.transform(hidden)
        xv.to_parquet(folder/'prediction_features.parquet', index=False)
        write_json(folder/'feature_schema.json', dict(features=list(xt), n_features=len(xt.columns)))
        log(f'{cutoff} 特征完成：{len(xt.columns)} 项')
        result.update(fit_models(train, xt, hidden, xv, folder, cutoff, threads, candidates))
    frames = [pd.DataFrame(dict(ml_num=homes.ml_num, candidate=c, predicted_price=result[c],
                               cutoff_exclusive=cutoff)) for c in candidates]
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(folder/'predictions_unscored.csv', index=False)
    return predictions


def report(output, selected, dev_scores, check_scores, verification):
    overall = check_scores.loc[check_scores.level == 'overall']
    winner = overall.loc[overall.candidate == selected].iloc[0]
    lines = ['# List-aware 第一版结果', '',
        '本轮复用 Off-market 的房屋属性、历史价格、历史可靠性、土地上下文及细化房况文字；新增挂牌价相对历史价格、挂牌单价和严格过去的成交挂牌比特征。', '',
        '目标：log(成交价/挂牌价)，预测后乘回挂牌价。固定 Log-L1、600 棵树、seed=42；不调参、不早停，不加入 Off-market 模型预测特征。四个完整候选先由 2025 年三个四个月窗口合并评分选定，再冻结评估 2026 年 2—5 月。', '',
        f'冻结入选：**{NAMES[selected]}**。选择依据只来自 2025 年；没有按 2026 年答案重新拼接房型。', '',
        '## 同口径完整方案对照', '',
        '原 List-aware 实现为挂牌价基线，本轮在同一批房屋上重算作为当前最优对照。Off-market 最优属于不同赛道，不能拿其 MdAPE 作为 List-aware 的改善幅度。', '',
        '| 完整方案 | 套数 | MdAPE | MAPE | 相对原挂牌价基线/百分点 |',
        '|---|---:|---:|---:|---:|']
    for r in overall.itertuples():
        lines.append(f'| {r.name}{"（冻结入选）" if r.candidate == selected else ""} | {r.n} | {r.mdape:.4f}% | {r.mape:.4f}% | {r.delta_mdape_pp:+.4f} |')
    lines += ['', f'冻结方案相对原当前最优：{winner.incumbent_mdape:.4f}% → {winner.mdape:.4f}%，变化 {winner.delta_mdape_pp:+.4f} 个百分点。', '',
        '## 入选方案按房型与月份对齐比较', '', '| 维度 | 分组 | 套数 | 原挂牌价基线 MdAPE | 入选 MdAPE | 差值/百分点 |',
        '|---|---|---:|---:|---:|---:|']
    for r in check_scores.loc[(check_scores.candidate == selected) & (check_scores.level != 'overall')].itertuples():
        lines.append(f'| {r.level} | {r.group} | {r.n} | {r.incumbent_mdape:.4f}% | {r.mdape:.4f}% | {r.delta_mdape_pp:+.4f} |')
    lines += ['', '## 验证边界与交付', '',
        '- 开发：截至 2024-12、2025-04、2025-08 的成交分别训练，固定预测其后四个月。按全年逐套误差合并选择完整候选。',
        '- 检查：截至 2026-01 训练，一次预测 2026-02 至 05；期间模型及历史成交池均不更新。',
        '- 历史统计均仅使用成交日严格早于 min(挂牌日, 截止日) 的记录；同日成交与样本自身排除。成交挂牌比使用对数均值分层平滑，指数还原为几何均值，不称为中位数。',
        '- 这些月份已参与整个项目的开发；所有分数均是累计开发回测，不是新的独立测试，也不是隐藏测试成绩。',
        '- 检查表列出全部事前候选；最终仅交付由较早窗口冻结的完整方案，即使另一个候选在检查窗口更低也不切换。',
        '- 最终使用截至 2026-05 的全部历史重训入选方案，生成 4,064 行预测；未覆盖原 submissions 文件。',
        f'- 独立核验：{json.dumps(verification, ensure_ascii=False)}。', '',
        '文件：development_comparison.csv、check_comparison.csv、selection.json、check/、final/、final_predictions/submission_list_aware.csv、verification.json。', '',
        '## 解释与限制', '',
        '- 挂牌价已提供很强的价格锚点，本轮检验的是其系统折扣与溢价能否预测。',
        '- 稀有房型统计向大组平滑；固定四组中的 Co-Op 与普通公寓共享训练。统一模型同时使用房型类别。',
        '- 历史成交价和挂牌价的关系可能变化；未来四个月无法获得新的成交答案，短期趋势特征会变旧。',
        '- 未验证置信区间与真实市场部署；坐标及社区为合成位置，不能连接真实地图。',
        '- 特征重要性仅表示模型分裂使用程度，不代表因果影响。']
    (output/'实验报告.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw, test = load_data(args.data_dir)
    best_path = REPO/'CURRENT_BEST_MODEL.json'
    best_before = json.loads(best_path.read_text())
    if 'list_aware_fixed_four_months' in best_before:
        raise ValueError('List-aware incumbent exists; adapt paired comparator before another run')
    shutil.copy2(best_path, out/'current_best_before.json')
    protected = {str(p): sha(p) for p in (REPO/'models.py', REPO/'features.py', *sorted((REPO/'submissions').glob('*.csv')))}
    sources = [*sorted(HERE.glob('*.py')), HERE/'README.md', HERE/'预先固定方案.md',
               REPO/'tests/test_list_aware_v1.py', REPO/'experiments/plan_a/features.py',
               REPO/'experiments/data_processing_round1/features.py',
               REPO/'experiments/lightgbm_official/models.py', REPO/'data.py', REPO/'check_submission.py']
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    source_hashes = {}
    for p in sources:
        source_hashes[str(p)] = sha(p)
        shutil.copy2(p, snapshot/str(p.relative_to(REPO)).replace('/', '__'))
    write_json(out/'protocol.json', dict(created_at=now(), track='list_aware', development_windows=DEV,
        check_window=CHECK, final_cutoff_exclusive=FINAL, candidates=CANDIDATES,
        groups=GROUPS, selection='lowest pooled 2025 MdAPE, MAPE breaks ties; entire candidate only',
        rounds=600, seed=42, threads=args.threads, early_stopping=False, tuning=False,
        reference_updates=False, target='log(sold_price/list_price)',
        ratio_windows=[90, 180, 365], ratio_smoothing={'type': 20, 'community_type': 30},
        historical_candidate='exp(community-type smoothed 180d mean log sold/list) times list',
        cumulative_development=True, data_dir=str(args.data_dir.resolve()),
        data_hashes={str(p.resolve()): sha(p) for p in [args.data_dir/'listings.parquet', args.data_dir/'test_list_aware.parquet']},
        source_hashes=source_hashes, protected_hashes=protected, best_index_hash_before=sha(best_path),
        versions={p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'lightgbm', 'pyarrow')}))
    development = []
    for cutoff, end in DEV:
        homes = raw.loc[(raw.date_sold >= cutoff) & (raw.date_sold < end)]
        predictions = predict_window(raw, homes, cutoff, out/'development'/cutoff, args.threads)
        development.append(predictions.assign(protocol='development'))
    development = pd.concat(development, ignore_index=True)
    development.to_csv(out/'development_predictions.csv', index=False)
    selected, dev_scores = select_candidate(development, raw)
    dev_scores.to_csv(out/'development_comparison.csv', index=False)
    write_json(out/'selection.json', dict(candidate=selected, name=NAMES[selected], frozen_at=now(),
        method='pooled 2025 MdAPE; MAPE breaks ties; select entire candidate',
        candidates=dev_scores.loc[dev_scores.level == 'overall'].to_dict('records')))
    log(f'2025 开发窗口已冻结完整方案：{NAMES[selected]}')
    homes = raw.loc[(raw.date_sold >= CHECK[0]) & (raw.date_sold < CHECK[1])]
    predictions = predict_window(raw, homes, CHECK[0], out/'check', args.threads)
    scores = score_table(predictions, homes)
    scores.to_csv(out/'check_comparison.csv', index=False)
    final = predict_window(raw, test, FINAL, out/'final', args.threads, [selected])
    final_dir = out/'final_predictions'
    final_dir.mkdir()
    submission = final[['ml_num', 'predicted_price']]
    submission.to_csv(final_dir/'submission_list_aware.csv', index=False)
    problems = check_submission.check_file(final_dir/'submission_list_aware.csv', test.ml_num)
    if problems:
        raise ValueError(problems)
    from .verify_results import verify
    checks = verify(out, args.data_dir)
    write_json(out/'verification.json', checks)
    report(out, selected, dev_scores, scores, checks)
    # Only independently verified, frozen, baseline-improving candidates enter
    # the registry. Preserve the off-market entry and original official files.
    winner = scores.loc[(scores.level == 'overall') & (scores.candidate == selected)].iloc[0]
    protocol = json.loads((out/'protocol.json').read_text())
    if sha(best_path) != protocol['best_index_hash_before']:
        raise ValueError('Best index changed concurrently')
    if winner.delta_mdape_pp < 0:
        best_before['recorded_date'] = '2026-11-04'
        best_before['list_aware_fixed_four_months'] = dict(
            model_id=str(out.relative_to(REPO)), display_name=NAMES[selected], candidate=selected,
            training_cutoff_exclusive=CHECK[0], evaluation_start_inclusive=CHECK[0],
            evaluation_end_exclusive=CHECK[1], reference_pool_updates_within_evaluation=False,
            selection_metric='MdAPE', metrics_percent=dict(n=int(winner.n), mdape=float(winner.mdape), mape=float(winner.mape)),
            prediction_evidence=str((out/'check/predictions_unscored.csv').relative_to(REPO)),
            prediction_candidate_filter=selected, prediction_evidence_sha256=sha(out/'check/predictions_unscored.csv'),
            final_predictions=str((final_dir/'submission_list_aware.csv').relative_to(REPO)),
            final_predictions_sha256=sha(final_dir/'submission_list_aware.csv'),
            comparison_note='2025 年三个固定四个月窗口选择完整候选，冻结后 2026-02 至 05 累计开发回测；非隐藏成绩。')
        write_json(best_path, best_before)
    write_json(out/'completion.json', dict(completed_at=now(), selected=selected,
        verification_passed=True, submission_rows=len(submission),
        best_index_updated=bool(winner.delta_mdape_pp < 0), best_index_sha256=sha(best_path)))
    log(scores.loc[scores.level == 'overall'].to_string(index=False))
    log(f'完成：{out}')


if __name__ == '__main__':
    main()
