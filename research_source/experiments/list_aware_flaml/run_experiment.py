"""Tune on Sep-Dec 2025, freeze, evaluate Feb-May 2026, refit through May."""
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
from experiments.flaml_tuning.models import validate_window
from experiments.list_aware_v1.features import FeatureBuilder
from experiments.list_aware_v1.models import RatioModel
from experiments.list_aware_v1.run_experiment import REPO, CHECK, FINAL, load_data, metric, now, log, sha, write_json
from .models import BASE_CONFIG, SEARCH_SPEC, search_space, validate_config, TunedRatioModel

HERE = Path(__file__).resolve().parent
ARCHIVE = REPO/'experiments/list_aware_v1/artifacts/run_2026-11-04_v1'
INNER = ('2025-09-01', '2026-01-01')
NAME = 'FLAML 自动调参统一 List-aware 成交挂牌比模型'


def prepare(raw, homes, cutoff, folder, archive_folder):
    folder.mkdir(parents=True, exist_ok=False)
    train = raw.loc[raw.date_sold < cutoff].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    hidden = data.hide_answers(homes, 'list_aware').reset_index(drop=True)
    np.testing.assert_array_equal(pd.read_csv(archive_folder/'train_ids.csv').ml_num, train.ml_num)
    np.testing.assert_array_equal(pd.read_csv(archive_folder/'predict_ids.csv').ml_num, hidden.ml_num)
    log(f'{cutoff} 重建训练特征：{len(train)} 套；预测 {len(hidden)} 套')
    builder = FeatureBuilder(train, cutoff)
    xt = builder.transform(data.hide_answers(train, 'list_aware'))
    xv = builder.transform(hidden)
    # Source archive is used as a control only after rebuilding all inputs.
    pd.testing.assert_frame_equal(xv, pd.read_parquet(archive_folder/'prediction_features.parquet'))
    old = pd.read_csv(archive_folder/'predictions_unscored.csv')
    old = old.loc[old.candidate == 'global_lgbm'].reset_index(drop=True)
    np.testing.assert_array_equal(old.ml_num, hidden.ml_num)
    np.testing.assert_array_equal(RatioModel.load(archive_folder/'global_lgbm').predict(xv, hidden), old.predicted_price)
    train[['ml_num', 'date_sold']].to_csv(folder/'train_ids.csv', index=False)
    hidden[['ml_num', 'date_listed']].to_csv(folder/'predict_ids.csv', index=False)
    xt.to_parquet(folder/'train_features.parquet', index=False)
    xv.to_parquet(folder/'prediction_features.parquet', index=False)
    old[['ml_num', 'predicted_price']].to_csv(folder/'baseline_predictions.csv', index=False)
    write_json(folder/'audit.json', dict(cutoff_exclusive=cutoff, training_n=len(train),
        max_train_sale=str(train.date_sold.max()), feature_count=len(xt.columns),
        train_features_sha256=sha(folder/'train_features.parquet'),
        predict_features_sha256=sha(folder/'prediction_features.parquet'),
        history_query='sale date < min(listing date, fixed cutoff)', pool_updates=False,
        raw_features_match_v1=True, v1_saved_model_replay_exact=True))
    return train, hidden, xt, xv, old


def select_trial(records):
    if not records or records[0]['config'] != BASE_CONFIG:
        raise ValueError('Trial zero must be the incumbent configuration')
    return min(records, key=lambda r: (r['mdape'], r['mape'], r['trial']))


def tune_inner(train, valid, xt, xv, baseline, folder, budget, trials, threads):
    from flaml import tune
    from flaml.tune.searcher import CFO
    validate_window(train, valid, *INNER)
    folder.mkdir()
    records = []
    def evaluate(config):
        config = validate_config(config)
        start = time.monotonic()
        model = TunedRatioModel(config, threads).fit(xt, train)
        prediction = model.predict(xv, valid)
        if not records:
            np.testing.assert_array_equal(prediction, baseline.predicted_price)
        score = metric(valid.sold_price, prediction)
        records.append(dict(trial=len(records), config=config, **score, seconds=time.monotonic()-start))
        write_json(folder/'trials.json', records)
        log(f'内层 trial {len(records)}：MdAPE={score["mdape"]:.4f}%')
        return {'mdape': score['mdape'], 'mape': score['mape']}
    space = search_space()
    search = CFO(space=space, metric='mdape', mode='min', seed=42,
                 points_to_evaluate=[dict(BASE_CONFIG)])
    tune.run(evaluate, config=space, search_alg=search, metric='mdape', mode='min',
             time_budget_s=budget, num_samples=trials, use_ray=False, verbose=0,
             local_dir=str(folder/'search_logs'))
    best = select_trial(records)
    write_json(folder/'selection.json', dict(config=best['config'], selected_trial=best['trial'],
        baseline_mdape=records[0]['mdape'], selected_mdape=best['mdape'],
        completed_trials=len(records), frozen_at=now(), selection_window=INNER))
    return best['config']


def paired_comparison(truth, baseline, prediction):
    if truth.ml_num.duplicated().any():
        raise ValueError('Duplicate truth IDs')
    pair = truth[['ml_num', 'sold_price', 'house_type_name', 'date_sold']].copy()
    for df, column in [(baseline, 'baseline_price'), (prediction, 'predicted_price')]:
        if df.ml_num.duplicated().any() or set(df.ml_num) != set(pair.ml_num):
            raise ValueError('Paired comparison requires identical unique IDs')
        pair = pair.merge(df[['ml_num', 'predicted_price']].rename(columns={'predicted_price': column}),
                          on='ml_num', validate='one_to_one')
    pair['month'] = pd.to_datetime(pair.date_sold).dt.strftime('%Y-%m')
    rows = []
    for level, groups in [('overall', [('all', pair)]), ('house_type_name', pair.groupby('house_type_name')),
                          ('month', pair.groupby('month'))]:
        for name, g in groups:
            before, after = metric(g.sold_price, g.baseline_price), metric(g.sold_price, g.predicted_price)
            rows.append(dict(level=level, group=name, n=len(g), baseline_mdape=before['mdape'],
                candidate_mdape=after['mdape'], delta_mdape_pp=after['mdape']-before['mdape'],
                baseline_mape=before['mape'], candidate_mape=after['mape']))
    return pair, pd.DataFrame(rows)


def fit_frozen(train, hidden, xt, xv, config, folder, cutoff, threads):
    model = TunedRatioModel(config, threads).fit(xt, train)
    model.save(folder/'model', cutoff_exclusive=cutoff, training_n=len(train), max_train_sale=str(train.date_sold.max()))
    prediction = model.predict(xv, hidden)
    np.testing.assert_array_equal(prediction, TunedRatioModel.load(folder/'model').predict(xv, hidden))
    result = pd.DataFrame(dict(ml_num=hidden.ml_num, predicted_price=prediction))
    result.to_csv(folder/'predictions_unscored.csv', index=False)
    return result


def report(run, table, inner):
    a = table.iloc[0]
    lines = ['# List-aware FLAML 自动调参', '',
        '在既有统一 LightGBM 成交挂牌比模型上接入与 Off-market 相同的 FLAML 2.7.0 CFO 搜索器。保留150项特征、统一训练人群、log(成交价/挂牌价)目标、Log-L1损失和seed=42，只搜索七个容量与正则参数。', '',
        '当前最优对照：统一 LightGBM 成交挂牌比模型（第一版）。同样的截至2026-01训练、固定模型和历史池预测2026-02至05，7773套房屋按ID对齐。', '',
        '| 当前最优 MdAPE | AutoML MdAPE | 变化/百分点 | 当前最优 MAPE | AutoML MAPE |',
        '|---:|---:|---:|---:|---:|',
        f'| {a.baseline_mdape:.6f}% | {a.candidate_mdape:.6f}% | {a.delta_mdape_pp:+.6f} | {a.baseline_mape:.6f}% | {a.candidate_mape:.6f}% |', '',
        f'内层完成 {inner["completed_trials"]} 次试验；2025-09至12的MdAPE由{inner["baseline_mdape"]:.6f}%降至{inner["selected_mdape"]:.6f}%。参数在检查窗口预测前冻结。', '',
        '## 相同样本分组比较', '', '| 维度 | 分组 | 套数 | 当前最优 MdAPE | AutoML MdAPE | 变化/百分点 |',
        '|---|---|---:|---:|---:|---:|']
    for r in table.iloc[1:].itertuples():
        lines.append(f'| {r.level} | {r.group} | {r.n} | {r.baseline_mdape:.4f}% | {r.candidate_mdape:.4f}% | {r.delta_mdape_pp:+.4f} |')
    lines += ['', '## 方法与限制', '',
        '- 搜索训练成交日 < 2025-09-01；内层验证成交日为2025-09至12。没有随机切分，没有早停，没有使用2026-02至05答案调参。',
        '- 搜索先完整训练第一版参数，要求逐套预测完全相同，然后探索新参数。按内层原价MdAPE选择，MAPE打破平局，不按检查答案拼接房型。',
        '- 七项参数和分布沿用Off-market；本轮默认单个统一模型搜索180秒，最多24次，已开始的训练会完成。全量训练，不抽样。',
        '- 时间预算影响试验数量；精确重放使用冻结参数和保存模型，不承诺重新搜索的路径一致。',
        '- 历史统计只使用成交日严格早于min(挂牌日,训练截止日)的记录；检查期间池不更新。训练/预测特征均从原始数据重建并与第一版预测特征一致。',
        '- 已使用过的时间窗口属于累计开发回测，不能声称新的独立测试或隐藏测试成绩。',
        '- AutoML在这里是既有LightGBM的参数搜索；没有加入算法比较、房型路由选择或新特征。',
        '- 截至2026-05的全量重训预测保存于final_predictions/submission_list_aware_flaml.csv（4064行）。无论是否改善均保留候选；仅核验通过且检查整体MdAPE改善时更新最优索引，原正式提交不覆盖。',
        '- 证据：protocol.json、tuning/trials.json、selected_config.json、comparison.csv、paired_predictions.csv、verification.json及各窗口保存模型。']
    (run/'实验报告.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--time-budget', type=float, default=180)
    parser.add_argument('--max-trials', type=int, default=24)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    if args.time_budget <= 0 or args.max_trials < 2 or args.threads < 1:
        raise ValueError('Positive budget/threads and at least two trials required')
    if importlib.metadata.version('flaml') != '2.7.0':
        raise ValueError('Requires flaml==2.7.0')
    best_path = REPO/'CURRENT_BEST_MODEL.json'
    best = json.loads(best_path.read_text())
    entry = best['list_aware_fixed_four_months']
    if entry['model_id'] != str(ARCHIVE.relative_to(REPO)) or entry['candidate'] != 'global_lgbm':
        raise ValueError('Incumbent has changed; adapt comparator to current best before tuning')
    if sha(REPO/entry['prediction_evidence']) != entry['prediction_evidence_sha256']:
        raise ValueError('Incumbent evidence hash mismatch')
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(best_path, run/'current_best_before.json')
    sources = [*sorted(HERE.glob('*.py')), HERE/'README.md', HERE/'预先固定方案.md',
        REPO/'tests/test_list_aware_flaml.py', *sorted((REPO/'experiments/list_aware_v1').glob('*.py')),
        REPO/'experiments/flaml_tuning/models.py', REPO/'experiments/plan_a/features.py',
        REPO/'experiments/data_processing_round1/features.py', REPO/'experiments/lightgbm_official/models.py',
        REPO/'data.py', REPO/'check_submission.py']
    snapshot = run/'source_snapshot'
    snapshot.mkdir()
    for p in sources:
        shutil.copy2(p, snapshot/str(p.relative_to(REPO)).replace('/', '__'))
    protected = [REPO/'models.py', REPO/'features.py', *sorted((REPO/'submissions').glob('*.csv')),
                 *sorted(p for p in ARCHIVE.rglob('*') if p.is_file())]
    write_json(run/'protocol.json', dict(created_at=now(), track='list_aware', archive=str(ARCHIVE),
        data_dir=str(args.data_dir.resolve()), inner_window=INNER, check_window=CHECK,
        final_cutoff_exclusive=FINAL, reference_updates=False, objective='log-ratio L1',
        target='log(sold_price/list_price)', search=SEARCH_SPEC, baseline_config=BASE_CONFIG,
        searcher='FLAML CFO', seed=42, threads=args.threads, time_budget_seconds=args.time_budget,
        max_trials=args.max_trials, time_budget_soft=True, early_stopping=False, cumulative_development=True,
        source_hashes={str(p): sha(p) for p in sources}, protected_hashes={str(p): sha(p) for p in protected},
        data_hashes={str(p.resolve()): sha(p) for p in [args.data_dir/'listings.parquet', args.data_dir/'test_list_aware.parquet']},
        best_index_hash_before=sha(best_path),
        versions={p: importlib.metadata.version(p) for p in ['flaml', 'lightgbm', 'numpy', 'pandas', 'pyarrow']}))
    raw, test = load_data(args.data_dir)
    valid = raw.loc[(raw.date_sold >= INNER[0]) & (raw.date_sold < INNER[1])].reset_index(drop=True)
    train, hidden, xt, xv, baseline = prepare(raw, valid, INNER[0], run/'inner', ARCHIVE/'development'/INNER[0])
    config = tune_inner(train, valid, xt, xv, baseline, run/'tuning', args.time_budget, args.max_trials, args.threads)
    write_json(run/'selected_config.json', dict(config=config, frozen_at=now(), selection_window=INNER))
    del train, hidden, xt, xv
    log('内层调参完成，参数已冻结；开始同7773套四个月回测')
    valid = raw.loc[(raw.date_sold >= CHECK[0]) & (raw.date_sold < CHECK[1])].reset_index(drop=True)
    train, hidden, xt, xv, baseline = prepare(raw, valid, CHECK[0], run/'check', ARCHIVE/'check')
    validate_window(train, valid, *CHECK)
    prediction = fit_frozen(train, hidden, xt, xv, config, run/'check', CHECK[0], args.threads)
    pair, table = paired_comparison(valid, baseline, prediction)
    np.testing.assert_allclose(table.iloc[0].baseline_mdape, entry['metrics_percent']['mdape'], rtol=0, atol=1e-12)
    pair.to_csv(run/'paired_predictions.csv', index=False)
    table.to_csv(run/'comparison.csv', index=False)
    del train, hidden, xt, xv
    log('开始截至5月全量重训与4064套最终预测')
    train, hidden, xt, xv, _ = prepare(raw, test, FINAL, run/'final', ARCHIVE/'final')
    final = fit_frozen(train, hidden, xt, xv, config, run/'final', FINAL, args.threads)
    final_dir = run/'final_predictions'
    final_dir.mkdir()
    final.to_csv(final_dir/'submission_list_aware_flaml.csv', index=False)
    del train, hidden, xt, xv
    from .verify_results import verify
    checks = verify(run, args.data_dir)
    write_json(run/'verification.json', checks)
    report(run, table, json.loads((run/'tuning/selection.json').read_text()))
    protocol = json.loads((run/'protocol.json').read_text())
    if sha(best_path) != protocol['best_index_hash_before']:
        raise ValueError('Current best index changed concurrently')
    a = table.iloc[0]
    if a.delta_mdape_pp < 0:
        best['recorded_date'] = '2026-11-04'
        best['list_aware_fixed_four_months'] = dict(model_id=str(run.relative_to(REPO)), display_name=NAME,
            candidate='global_ratio_flaml', training_cutoff_exclusive=CHECK[0],
            evaluation_start_inclusive=CHECK[0], evaluation_end_exclusive=CHECK[1],
            reference_pool_updates_within_evaluation=False, selection_metric='MdAPE',
            metrics_percent=dict(n=int(a.n), mdape=float(a.candidate_mdape), mape=float(a.candidate_mape)),
            prediction_evidence=str((run/'check/predictions_unscored.csv').relative_to(REPO)),
            prediction_evidence_sha256=sha(run/'check/predictions_unscored.csv'),
            final_predictions=str((final_dir/'submission_list_aware_flaml.csv').relative_to(REPO)),
            final_predictions_sha256=sha(final_dir/'submission_list_aware_flaml.csv'),
            lineage=[entry['model_id'], str(run.relative_to(REPO))],
            comparison_note='FLAML仅使用2025-09至12调参，冻结后2026-02至05同7773套累计开发回测；非隐藏测试成绩。')
        write_json(best_path, best)
    write_json(run/'completion.json', dict(completed_at=now(), verified=True,
        best_index_updated=bool(a.delta_mdape_pp < 0), baseline_mdape=float(a.baseline_mdape),
        candidate_mdape=float(a.candidate_mdape), delta_mdape_pp=float(a.delta_mdape_pp),
        final_rows=len(final)))
    log(table.iloc[[0]].to_string(index=False))
    log(f'完成：{run}')


if __name__ == '__main__':
    main()
