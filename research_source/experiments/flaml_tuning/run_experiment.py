"""Tune on Sep-Dec 2025; freeze; evaluate Feb-May 2026; refit for hidden homes."""
# 阅读顺序：main() 组织全流程 → tune_branch() 为各房型选参数 →
# fit_predict() 用冻结参数训练/预测 → comparison() 对齐评分 → 核验后决定是否更新最优索引。
# 本文件负责 Off-market（无挂牌价估值）；特征生成和公寓旧模型来自前序实验。
# 本次仅添加讲解注释，artifacts 中的历史源码快照与模型不作修改。

# 标准库：解析运行参数、记录依赖版本、读写结果、管理路径、保存快照和计时。
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

# 数值/表格工具：比较预测是否一致、按房屋 ID 合并数据、计算分组结果。
import numpy as np
import pandas as pd

# 项目依赖：复用已有的数据切分、特征缓存、房型路由、来源核对和日志函数。
# TARGETS 顺序是 Detached、Semi-Detached、Freehold Townhouse、Condo Townhouse；
# 其余两类公寓沿用存档预测，不在本轮重新调参。
# OfficialLightGBM 封装训练、预测及保存/加载；本目录 models.py 定义搜索参数与校验工具。
import check_submission
import data
from experiments.dart_dropout.run_experiment import (
    REPO, TARGETS, PROCESSING, RESIDUAL, archived_baseline, load_designs,
    validate_provenance, now, sha, write_json, log)
from experiments.data_processing_round1.run_experiment import load_data
from experiments.lightgbm_official.models import OfficialLightGBM
from .models import (BASE_CONFIG, SEARCH_SPEC, canonical_config, search_space, make_model, price_metrics,
                     validate_window, replace_branch)

# 三个时间节点都采用左闭右开约定：训练成交日 < cutoff，验证成交日在 [cutoff, end)。
# INNER：训练到 2025 年 8 月，验证 9—12 月，用于调参。
# OUTER：训练到 2026 年 1 月，预测 2—5 月，用于比较完整方案。
# FINAL：训练到 2026 年 5 月，预测隐藏测试房屋；BEST 保存当前最优索引。
HERE = Path(__file__).resolve().parent
INNER = ('2025-09-01', '2026-01-01')
OUTER = ('2026-02-01', '2026-06-01')
FINAL = '2026-06-01'
BEST = REPO/'CURRENT_BEST_MODEL.json'


def tune_branch(design, output, budget, trials, threads):
    """This function receives only the earlier window, never outer answers."""
    # design = (训练房屋表, 验证房屋表, 训练特征 xt, 验证特征 xv)。
    # 调参函数只接收较早窗口的数据；验证时间边界后保存 ID，方便追溯每次用了哪些房屋。
    from flaml import tune
    from flaml.tune.searcher import CFO
    train, valid, xt, xv = design
    validate_window(train, valid, *INNER)
    output.mkdir(parents=True)
    train[['ml_num', 'date_sold']].to_csv(output/'train_ids.csv', index=False)
    valid[['ml_num', 'date_sold']].to_csv(output/'valid_ids.csv', index=False)
    records = []

    # CFO 每提出一组参数，就调用一次 evaluate：完整训练该分支，再在内层窗口评分。
    # 不使用外层答案，也不使用早停；树数量本身就是搜索参数。
    def evaluate(config):
        # 统一参数类型 → 训练 Log-L1 模型 → 在原房价尺度预测并计算 MdAPE/MAPE。
        # 逐次写入 trials.json，保留参数、成绩和耗时；主目标是最小化 MdAPE。
        config = canonical_config(config)
        start = time.monotonic()
        model = make_model(config, threads).fit(xt, train.sold_price)
        result = price_metrics(valid.sold_price, model.predict(xv))
        records.append(dict(trial=len(records), config=dict(config), **result,
                            seconds=time.monotonic()-start))
        write_json(output/'trials.json', records)
        log(f'内层 {valid.house_type_name.iloc[0]} trial={len(records)} MdAPE={result["mdape"]:.4f}%')
        return {'mdape': result['mdape'], 'mape': result['mape']}

    # CFO 是 FLAML 的参数搜索器。固定 seed=42，并把旧参数作为第一个评估点，
    # 保证搜索至少从已有设置开始；时间预算按房型分配，在途试验可完成后再结束。
    # use_ray=False 表示不依赖 Ray；trials 限制试验次数。
    space = search_space()
    search = CFO(space=space, metric='mdape', mode='min', seed=42,
                 points_to_evaluate=[dict(BASE_CONFIG)])
    analysis = tune.run(evaluate, config=space, search_alg=search, metric='mdape', mode='min',
        time_budget_s=budget, num_samples=trials, use_ray=False, verbose=0,
        local_dir=str(output/'search_logs'))
    # 没有完成任何试验就停止；否则取内层 MdAPE 最低的参数，并核对首个试验确为旧参数。
    if not records or analysis.best_config is None:
        raise RuntimeError('No completed tuning trial')
    selected = canonical_config(analysis.best_config)
    # Search starts with the existing settings; all seven settings are persisted.
    if records[0]['config'] != BASE_CONFIG:
        raise ValueError('Search did not start from current baseline')
    # 保存该房型的获选参数、内层起点/最佳分数、完成次数和冻结时间，供后续审计。
    write_json(output/'selection.json', dict(config=selected,
        inner_baseline_mdape=records[0]['mdape'], inner_best_mdape=float(analysis.best_result['mdape']),
        completed_trials=len(records), frozen_at=now()))
    return selected


def fit_predict(raw, test, cutoff, end, configs, output, threads):
    # 按截止日读取已核对 ID 的特征缓存；end=None 表示最终隐藏预测，否则是历史回测。
    # 先复制旧完整方案，再逐个替换本轮训练的四个分支，保留另外两类公寓的原预测。
    homes, designs = load_designs(raw, test, cutoff, end)
    output.mkdir(parents=True)
    baseline = archived_baseline(homes, cutoff, final=end is None)
    result = baseline.copy()
    # 各房型使用各自既定训练人群和特征列；这里只改变模型参数，不重新选择房型路由。
    for target in TARGETS:
        train, valid, xt, xv = designs[target]
        if end is not None:
            validate_window(train, valid, cutoff, end)
        elif not (train.date_sold < pd.Timestamp(cutoff)).all():
            raise ValueError('Final training cutoff violated')
        # 用已冻结参数从头训练当前房型。外层训练时不使用验证答案调参或早停；
        # 最终训练时加入截至 5 月的历史成交，并保持同一组参数。
        log(f'冻结参数重训 {cutoff} {target}: {len(train)}套')
        model = make_model(configs[target], threads).fit(xt, train.sold_price)
        predicted = model.predict(xv)
        folder = output/target
        # 保存模型及训练截止日、训练条数、训练房型等信息，便于确认训练数据边界。
        model.save(folder, cutoff_exclusive=cutoff, training_n=len(train),
                   max_train_sale=str(train.date_sold.max()), target_type=target,
                   training_types=sorted(train.house_type_name.unique().tolist()))
        # 立即加载磁盘模型重放同一批特征，要求预测逐套完全一致，检查保存/加载过程。
        if not np.array_equal(predicted, OfficialLightGBM.load(folder).predict(xv)):
            raise ValueError('Saved model replay mismatch')
        # 每个分支保存预测与训练 ID，再按房屋 ID 写回完整预测表，保持全表顺序。
        valid[['ml_num', 'date_listed']].assign(predicted_price=predicted).to_csv(
            folder/'predictions.csv', index=False)
        train[['ml_num', 'date_sold']].to_csv(folder/'train_ids.csv', index=False)
        result = replace_branch(result, valid, predicted)
    # 同时保存旧完整预测与新完整预测；此处的候选表尚未附上真实成交价或评分。
    baseline.to_csv(output/'baseline_predictions.csv', index=False)
    result.to_csv(output/'predictions_unscored.csv', index=False)
    return homes, baseline, result


def comparison(homes, baseline, candidate):
    # 把答案、旧方案和候选方案按 ml_num 一对一合并，防止行顺序不同造成错误比较。
    # 确保覆盖全部验证房屋后，才允许评分。
    truth = homes[['ml_num', 'house_type_name', 'date_sold', 'sold_price']].copy()
    paired = truth.merge(baseline[['ml_num', 'predicted_price']].rename(
        columns={'predicted_price': 'baseline_price'}), on='ml_num', validate='one_to_one')
    paired = paired.merge(candidate[['ml_num', 'predicted_price']], on='ml_num', validate='one_to_one')
    if len(paired) != len(homes):
        raise ValueError('Scoring IDs differ')
    # 分别计算整体、房型和成交月份的 MdAPE/MAPE；每组都在同一批房屋上比较。
    # delta_mdape_pp = 候选 - 旧方案，单位为百分点，负值表示候选误差更小。
    # 总体指标直接由全体验证房屋计算，不取各房型 MdAPE 的平均值。
    paired['month'] = pd.to_datetime(paired.date_sold).dt.strftime('%Y-%m')
    rows = []
    for level, groups in [('overall', [('all', paired)]),
                          ('type', paired.groupby('house_type_name', sort=True)),
                          ('month', paired.groupby('month', sort=True))]:
        for name, group in groups:
            a, b = (price_metrics(group.sold_price, group[c])
                    for c in ('baseline_price', 'predicted_price'))
            rows.append(dict(level=level, group=name, n=len(group),
                baseline_mdape=a['mdape'], candidate_mdape=b['mdape'],
                delta_mdape_pp=b['mdape']-a['mdape'], baseline_mape=a['mape'], candidate_mape=b['mape']))
    return paired, pd.DataFrame(rows)


def report(output, table, baseline_name, selections):
    # 将总体成绩、分组比较、搜索次数、复现文件位置和研究局限写为中文报告。
    # 报告中的“当前最优对照”是本次运行开始时冻结的基线；隐藏测试没有答案，不能评分。
    overall = table.iloc[0]
    lines = ['# FLAML 自动调参实验', '',
        f'当前最优对照：{baseline_name}。保留数据处理、训练人群、房型路由和 Log-L1；FLAML CFO 搜索四个 LightGBM 分支的七个参数。',
        '搜索只使用截至2025年8月成交训练，固定预测2025年9—12月。各分支在该窗口按原价格尺度MdAPE选择参数；不使用2026年2—5月答案调参或早停。',
        '全部参数冻结后，截至2026年1月重训，固定预测2—5月，成交历史池不更新。该外层及内层窗口均已被项目反复查看，属于累计开发回测，不是全新独立测试。', '',
        '| 当前最优 MdAPE | 自动调参 MdAPE | 差值/百分点 |', '|---:|---:|---:|',
        f'| {overall.baseline_mdape:.6f}% | {overall.candidate_mdape:.6f}% | {overall.delta_mdape_pp:+.6f} |', '',
        '## 同样本房型与月份对比', '', '| 维度 | 分组 | 套数 | 当前最优MdAPE | 调参MdAPE | 差值/百分点 |',
        '|---|---|---:|---:|---:|---:|']
    for row in table.iloc[1:].itertuples():
        lines.append(f'| {row.level} | {row.group} | {row.n} | {row.baseline_mdape:.3f}% | {row.candidate_mdape:.3f}% | {row.delta_mdape_pp:+.3f} |')
    lines += ['', '## 搜索与复现', '',
        f'- 每分支完成试验数：{selections}。包含原始参数起点；实际时间受单次训练耗时影响，预算结束时让在途试验完成。',
        '- selected_configs.json 在外层预测前冻结；tuning/ 保存每次配置、内层分数与训练/验证ID。未按外层结果重新选择房型或参数。',
        '- models/ 保存外层模型，final_models/ 保存截至5月重训模型；均经过保存、加载、预测一致性核对。',
        '- 公寓两分支沿用原冻结模型的预测；因此重放仍依赖原始特征与模型存档，protocol.json记录依赖哈希，不能单独搬走本目录。',
        '- final_predictions/submission_off_market_flaml.csv 为3974套隔离候选，正式提交未覆盖。隐藏测试成绩未知。',
        '- 当前轮仅接入 Off-market 四个 LightGBM 分支；List-aware 和公寓 CatBoost 的参数搜索可后续扩展。',
        '- 核验见 verification.json；更优且全部核验通过时才更新最优索引，原索引保存在 current_best_before.json。',
        '- FLAML 官方调参文档：https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function/ 。', '']
    (output/'实验报告.md').write_text('\n'.join(lines))


def main():
    # 第 1 步：读取输出目录、数据目录、每房型时间预算、最多试验次数与线程数。
    # 输出目录必须是新的，避免覆盖旧实验；非法预算或线程数会直接报错。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--time-budget', type=float, default=90., help='Seconds per branch (in-flight trial may finish)')
    parser.add_argument('--max-trials', type=int, default=24)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    if args.time_budget <= 0 or args.max_trials < 2 or args.threads < 1:
        parser.error('Positive budget/threads and at least two trials required')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    # 第 2 步：核对前序特征源码/原始数据来源，再读取当前最优模型的索引与预测证据。
    # 这是为当时 all_three 基线写的实验适配器，当前索引已指向 FLAML 后，
    # 直接重跑此训练入口会被下面的检查阻止；复现现有最终预测应使用 predict_saved.py。
    validate_provenance()
    best = json.loads(BEST.read_text())
    baseline_info = best['off_market_fixed_four_months']
    # This adapter intentionally targets the archived all_three architecture.
    if baseline_info['model_id'] != 'data_processing_round1/run_2026-10-19_v1/all_three':
        raise ValueError('Current best changed: adapt baseline provenance before a new comparison')
    evidence = REPO/baseline_info['prediction_evidence']
    if sha(evidence) != baseline_info['prediction_evidence_sha256']:
        raise ValueError('Current best evidence hash mismatch')
    # 第 3 步：读取训练/隐藏测试数据，新建结果目录，保存运行前的最优索引。
    # 收集原始数据、旧实验预测及三个截止日的特征缓存/ID，记录它们的文件指纹。
    raw, test = load_data(args.data_dir)
    output.mkdir(parents=True)
    shutil.copy2(BEST, output/'current_best_before.json')
    inputs = [args.data_dir/'listings.parquet', args.data_dir/'test_off_market.parquet',
              PROCESSING/'protocol.json', RESIDUAL/'source_manifest.json', evidence,
              PROCESSING/'all_predictions_scored.csv', PROCESSING/'final_predictions/submission_off_market.csv',
              RESIDUAL/'development_candidates.csv', RESIDUAL/'selection.json']
    for cutoff in (INNER[0], OUTER[0], FINAL):
        inputs += list((RESIDUAL/'bases'/cutoff).glob('*.parquet'))
        inputs += list((RESIDUAL/'bases'/cutoff).glob('*ids.csv'))
        inputs += list((PROCESSING/'windows'/cutoff/'features').glob('*.parquet'))
        inputs += list((PROCESSING/'windows'/cutoff/'features').glob('*ids.csv'))
    # 收集本轮及前序依赖的源码、测试和依赖清单，记录完整代码来源，供审计而非重新训练。
    sources = sorted(set(list(HERE.glob('*.py')) + list(HERE.glob('*.txt')) +
        [REPO/'tests/test_flaml_tuning.py', REPO/'data.py', REPO/'check_submission.py'] +
        [p for package in ('plan_a', 'property_groups', 'data_processing_round1', 'residual_refinement',
                          'dart_dropout', 'lightgbm_official')
         for p in (REPO/'experiments'/package).glob('*.py')]))
    # 第 4 步：在调参前写入实验协议：时间窗口、搜索范围、种子、预算、版本与文件哈希。
    # 同时记录已有正式提交的指纹，以便事后确认实验没有改动正式提交。
    write_json(output/'protocol.json', dict(created_at=now(), track='off_market',
        inner_window=INNER, outer_window=OUTER, final_cutoff=FINAL, history_updates=False,
        outer_labels_used_for_tuning=False, previously_examined_windows=True,
        selection_metric='MdAPE on original prices', search=SEARCH_SPEC, baseline_config=BASE_CONFIG,
        branches=list(TARGETS), seed=42, threads=args.threads, time_budget_per_branch=args.time_budget,
        max_trials_per_branch=args.max_trials, budget_is_soft=True,
        baseline_model=baseline_info, baseline_index_sha256=sha(BEST),
        source_sha256={str(p): sha(p) for p in sources},
        input_sha256={str(p.resolve()): sha(p) for p in inputs},
        protected_submissions={str(p): sha(p) for p in (REPO/'submissions').glob('*.csv')},
        versions={p: importlib.metadata.version(p) for p in ('flaml', 'lightgbm', 'numpy', 'pandas', 'scikit-learn')}))
    # 复制当时实际使用的源码到独立快照；文件名用双下划线代替路径分隔符。
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot/str(path.relative_to(REPO)).replace('/', '__'))
    # 第 5 步：只载入内层窗口，逐房型搜索；每个分支按自己的内层成绩独立选参数。
    # 四组参数全部写入 selected_configs.json 后记录哈希，后续不得再改。
    _, designs = load_designs(raw, test, *INNER)
    configs = {}
    # 各房型使用各自既定训练人群和特征列；这里只改变模型参数，不重新选择房型路由。
    for target in TARGETS:
        log(f'开始自动调参 {target}，预算{args.time_budget:g}秒，仅使用2025年窗口')
        configs[target] = tune_branch(designs[target], output/'tuning'/target,
                                     args.time_budget, args.max_trials, args.threads)
    write_json(output/'selected_configs.json', dict(frozen_at=now(), configs=configs,
               tuning_window=INNER, no_outer_labels=True))
    frozen_hash = sha(output/'selected_configs.json')
    del designs
    # 第 6 步：释放内层设计表，用冻结参数重训外层模型，一次预测 2026 年 2—5 月。
    homes, baseline, candidate = fit_predict(raw, test, *OUTER, configs, output/'models', args.threads)
    # Verify the comparator is the indexed current best on exactly these homes.
    # 把基线按房屋 ID 重排，核对它与最优索引中保存的完整预测逐套一致。
    expected = pd.read_csv(evidence).set_index('ml_num').predicted_price.reindex(baseline.ml_num)
    if not np.array_equal(expected.to_numpy(), baseline.predicted_price.to_numpy()):
        raise ValueError('Comparator differs from indexed current best')
    # 基线核对通过后才计算整体/分组结果，并保存带答案的配对表和比较表。
    paired, table = comparison(homes, baseline, candidate)
    paired.to_csv(output/'paired_predictions.csv', index=False)
    table.to_csv(output/'comparison_to_current_best.csv', index=False)
    # 第 7 步：同样参数重训到 2026 年 5 月，预测隐藏测试房屋，写入独立候选文件。
    # 不根据外层比较结果再改变参数；输出只包含官方要求的房屋 ID 与预测价格两列。
    _, _, final = fit_predict(raw, test, FINAL, None, configs, output/'final_models', args.threads)
    dest = output/'final_predictions'
    dest.mkdir()
    final[['ml_num', 'predicted_price']].to_csv(dest/'submission_off_market_flaml.csv', index=False)
    # 确认评估期间没有改动冻结参数，随后写入完成记录并生成报告。
    if sha(output/'selected_configs.json') != frozen_hash:
        raise ValueError('Frozen parameters changed during evaluation')
    write_json(output/'completion.json', dict(completed_at=now(), selected_sha256=frozen_hash))
    counts = {t: json.loads((output/'tuning'/t/'selection.json').read_text())['completed_trials'] for t in TARGETS}
    report(output, table, baseline_info['display_name'], counts)
    # 第 8 步：核对输入/源码、时间边界、参数选择、模型重放、指标和隐藏预测格式。
    # 只有核验通过且同样本误差下降，才更新开发回测最优索引；这不等于正式提交。
    from .verify_results import verify_run, promote_if_better
    verified = verify_run(output, args.data_dir)
    promote_if_better(output, verified)
    log(f'完成：当前最优 {table.iloc[0].baseline_mdape:.6f}%，调参 {table.iloc[0].candidate_mdape:.6f}%，差值 {table.iloc[0].delta_mdape_pp:+.6f}个百分点')


# 通过 python -m experiments.flaml_tuning.run_experiment 运行时执行 main；被导入时不启动实验。
if __name__ == '__main__':
    main()
