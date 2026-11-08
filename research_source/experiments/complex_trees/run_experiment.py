"""Force a larger-capacity tree family, tune on earlier windows, then compare."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from experiments.dart_dropout.run_experiment import (
    REPO, TARGETS, archived_baseline, load_designs, validate_provenance,
    input_files, sha, write_json, now, log)
from experiments.data_processing_round1.run_experiment import load_data
from experiments.flaml_tuning.models import price_metrics, validate_window, replace_branch
from experiments.flaml_tuning.run_experiment import comparison
from experiments.lightgbm_official.models import OfficialLightGBM
from .models import SEARCH_SPEC, normalized, search_space, starting_points, make_model, complexity

HERE = Path(__file__).resolve().parent
INNER_WINDOWS = [('2025-05-01', '2025-09-01'), ('2025-09-01', '2026-01-01')]
OUTER = ('2026-02-01', '2026-06-01')
FINAL = '2026-06-01'
BEST = REPO/'CURRENT_BEST_MODEL.json'
BASE_RUN = REPO/'experiments/flaml_tuning/artifacts/run_2026-10-29_v1'


def fit_inner_config(config, folds, output, threads, early_stop):
    """Only receives the inner windows, with each historical pool already fixed."""
    all_truth, all_prediction = [], []
    details = []
    output.mkdir(parents=True)
    for cutoff, end, design in folds:
        train, valid, xt, xv = design
        validate_window(train, valid, cutoff, end)
        if pd.Timestamp(end) > pd.Timestamp(OUTER[0]):
            raise ValueError('Outer labels are forbidden in capacity tuning')
        model = make_model(config, threads).fit_inner(xt, train.sold_price, xv,
                                                     valid.sold_price, patience=early_stop)
        prediction = model.predict(xv)
        score = price_metrics(valid.sold_price, prediction)
        folder = output/cutoff
        folder.mkdir()
        valid[['ml_num', 'house_type_name', 'date_sold', 'sold_price']].assign(
            predicted_price=prediction).to_csv(folder/'predictions.csv', index=False)
        # Score training only for diagnosis; never use it to select a candidate.
        train_mask = train.house_type_name.eq(valid.house_type_name.iloc[0]).to_numpy()
        train_score = price_metrics(train.loc[train_mask, 'sold_price'], model.predict(xt.loc[train_mask]))
        details.append(dict(cutoff=cutoff, end=end, selected_rounds=model.selected_rounds,
                            train_target_mdape=train_score['mdape'], **score))
        all_truth.extend(valid.sold_price)
        all_prediction.extend(prediction)
        log(f'  {cutoff} 完成：最佳轮数 {model.selected_rounds}，内层MdAPE {score["mdape"]:.4f}%')
    pooled = price_metrics(all_truth, all_prediction)
    return dict(**pooled, folds=details)


def search_branch(target, folds, baseline_config, output, trials, threads, early_stop):
    from flaml import tune
    from flaml.tune.searcher import CFO
    output.mkdir(parents=True)
    for cutoff, end, (train, valid, _, _) in folds:
        validate_window(train, valid, cutoff, end)
        folder = output/'splits'/cutoff
        folder.mkdir(parents=True)
        train[['ml_num', 'date_sold']].to_csv(folder/'train_ids.csv', index=False)
        valid[['ml_num', 'date_sold']].to_csv(folder/'valid_ids.csv', index=False)
    records = []
    # Fixed current-best settings are diagnostic controls, not a search option:
    # every candidate selected below has at least 127 available leaves.
    controls = []
    for cutoff, end, (train, valid, xt, xv) in folds:
        log(f'{target} 当前最优参数对照 {cutoff}')
        model = make_model(baseline_config, threads).fit(xt, train.sold_price)
        price = model.predict(xv)
        train_mask = train.house_type_name.eq(target).to_numpy()
        item = dict(cutoff=cutoff, end=end, **price_metrics(valid.sold_price, price),
            train_target_mdape=price_metrics(train.loc[train_mask, 'sold_price'],
                                            model.predict(xt.loc[train_mask]))['mdape'])
        controls.append(item)
        folder = output/'control'/cutoff
        folder.mkdir(parents=True)
        valid[['ml_num', 'sold_price']].assign(predicted_price=price).to_csv(folder/'predictions.csv', index=False)
    write_json(output/'control.json', dict(config=baseline_config, folds=controls))

    def evaluate(config):
        config = normalized(config)
        number = len(records)
        log(f'搜索 {target} trial={number+1}/{trials} leaves={config["num_leaves"]} depth={config["max_depth"]}')
        started = time.monotonic()
        result = fit_inner_config(config, folds, output/f'trial_{number:03d}', threads, early_stop)
        record = dict(trial=number, config=config, seconds=time.monotonic()-started, **result)
        records.append(record)
        write_json(output/'trials.json', records)
        log(f'{target} trial={number+1} 两窗口合并MdAPE={record["mdape"]:.4f}%')
        return dict(mdape=result['mdape'], mape=result['mape'])

    space = search_space()
    search = CFO(space=space, metric='mdape', mode='min', seed=42,
                 points_to_evaluate=starting_points())
    tune.run(evaluate, config=space, search_alg=search, metric='mdape', mode='min',
             num_samples=trials, use_ray=False, verbose=0,
             local_dir=str(output/'search_logs'))
    if len(records) != trials:
        raise ValueError('Search stopped before requested number of completed trials')
    winner = min(records, key=lambda r: (r['mdape'], r['mape'], r['trial']))
    # Only earlier-window best rounds determine full-data training duration.
    rounds = int(np.floor(np.median([f['selected_rounds'] for f in winner['folds']])+.5))
    selected = dict(winner['config'], n_estimators=rounds)
    write_json(output/'selection.json', dict(config=selected, search_config=winner['config'],
        selected_trial=winner['trial'], inner_mdape=winner['mdape'], completed_trials=len(records),
        inner_best_rounds=[f['selected_rounds'] for f in winner['folds']],
        round_rule='nearest integer to median of inner best iterations', frozen_at=now()))
    return selected


def frozen_predict(raw, test, cutoff, end, configs, output, threads, baseline_file, baseline_models):
    homes, designs = load_designs(raw, test, cutoff, end)
    output.mkdir(parents=True)
    baseline = archived_baseline(homes, cutoff, final=end is None)
    stored = pd.read_csv(baseline_file)
    if stored.ml_num.duplicated().any() or set(stored.ml_num) != set(homes.ml_num):
        raise ValueError('Current-best prediction IDs differ')
    baseline['predicted_price'] = stored.set_index('ml_num').predicted_price.reindex(baseline.ml_num).to_numpy()
    result = baseline.copy()
    audits = []
    for target in TARGETS:
        train, valid, xt, xv = designs[target]
        if end is not None:
            validate_window(train, valid, cutoff, end)
        log(f'冻结大树重训 {cutoff}/{target}：{configs[target]["n_estimators"]}棵、{configs[target]["num_leaves"]}叶上限')
        model = make_model(configs[target], threads).fit(xt, train.sold_price)
        values = model.predict(xv)
        folder = output/target
        model.save(folder, cutoff_exclusive=cutoff, training_n=len(train), target_type=target,
                   max_train_sale=str(train.date_sold.max()),
                   parameter_selection='two earlier four-month windows only')
        loaded = OfficialLightGBM.load(folder)
        if not np.array_equal(loaded.predict(xv), values):
            raise ValueError('Large model replay mismatch')
        old = OfficialLightGBM.load(baseline_models/target)
        expected = baseline.set_index('ml_num').predicted_price.reindex(valid.ml_num).to_numpy()
        if not np.array_equal(old.predict(xv), expected):
            raise ValueError('Current-best model does not reproduce indexed predictions')
        mask = train.house_type_name.eq(target).to_numpy()
        for label, fitted in [('current_best', old), ('complex', model)]:
            audits.append(dict(target=target, candidate=label, **complexity(fitted),
                train_target_n=int(mask.sum()), train_target_mdape=price_metrics(
                    train.loc[mask, 'sold_price'], fitted.predict(xt.loc[mask]))['mdape']))
        train[['ml_num', 'date_sold']].to_csv(folder/'train_ids.csv', index=False)
        valid[['ml_num', 'date_listed']].assign(predicted_price=values).to_csv(folder/'predictions.csv', index=False)
        result = replace_branch(result, valid, values)
    baseline.to_csv(output/'baseline_predictions.csv', index=False)
    result.to_csv(output/'predictions_unscored.csv', index=False)
    pd.DataFrame(audits).to_csv(output/'capacity_audit.csv', index=False)
    return homes, baseline, result


def report(output, table, best_name):
    selected = json.loads((output/'selected_configs.json').read_text())
    lines = ['# 大容量 LightGBM 深度优化实验', '',
        f'当前最优完整对照：{best_name}。同Off-market、2026-02-01截止、冻结成交池、固定预测2026年2—5月7773套。',
        '保持特征、房型训练人群、Log-L1和两类公寓路由；四个LightGBM分支全部采用大容量候选，不按外层答案回退个别房型。',
        '内层为2025年5—8月、9—12月两个固定四个月窗口；每个窗口只用开始前的成交训练。FLAML以两个窗口逐套合并的目标房型MdAPE选参数。',
        '每个候选训练上限8000轮，仅内层原价MdAPE早停，patience=120。最终轮数取获选候选两个内层最佳轮数的中位数并四舍五入，外层和最终模型固定轮数重训，无外层早停。',
        '搜索强制127/255/511叶容量，深度上限12/16/24/不限制，联合学习率、每叶样本、L1/L2、行列采样及255/511分箱。当前最优仅作对照，不作为大树选择器的回退选项。',
        '全部窗口此前已被项目查看：属于累计开发回测，不是独立最终测试。搜索更广也会增加开发期过拟合风险，不能据此保证隐藏成绩。', '',
        '## 当前最优完整方案对比', '',
        '| 分组 | n | 当前最优MdAPE | 大树MdAPE | 差值/百分点 | 当前最优MAPE | 大树MAPE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in table.itertuples():
        lines.append(f'| {row.level}/{row.group} | {row.n} | {row.baseline_mdape:.4f}% | {row.candidate_mdape:.4f}% | {row.delta_mdape_pp:+.4f} | {row.baseline_mape:.4f}% | {row.candidate_mape:.4f}% |')
    lines += ['', '## 冻结的复杂模型参数', '', '| 房型 | 树数量 | 叶子上限 | 深度上限 | 学习率 | 最少叶样本 |',
              '|---|---:|---:|---:|---:|---:|']
    for kind, c in selected['configs'].items():
        lines.append(f'| {kind} | {c["n_estimators"]} | {c["num_leaves"]} | {c["max_depth"]} | {c["learning_rate"]:.5f} | {c["min_child_samples"]} |')
    lines += ['', '## 真实复杂度与训练拟合', '',
              '| 房型 | 方案 | 实际树数 | 实际最大深度 | 单树平均叶数 | 总叶数 | 同房型训练MdAPE |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in pd.read_csv(output/'models/capacity_audit.csv').itertuples():
        lines.append(f'| {row.target} | {row.candidate} | {row.trees} | {row.max_actual_depth} | {row.mean_leaves_per_tree:.1f} | {row.total_leaves} | {row.train_target_mdape:.4f}% |')
    lines += ['', '配置深度是上限；实际最大深度、平均叶数与总叶数来自保存模型结构，不能把叶子数直接称作深度。训练误差只用于诊断，不参与选择。', '',
        '## 文件与复现', '',
        '- protocol.json和源码快照冻结搜索范围、输入、源码、基线及正式提交哈希；tuning/保存每次配置、各窗口未选中候选预测和全部分数。',
        '- selected_configs.json在外层训练前冻结；models/与final_models/保存固定参数模型，完成保存/加载及原始最终特征重建重放。',
        '- final_predictions/submission_off_market_complex.csv包含3974套隔离候选；现有正式提交未覆盖，隐藏成绩未知。',
        '- 依赖原始数据、既有特征缓存与公寓冻结模型，不能只搬本目录复现；依赖路径和哈希见协议。',
        '- 核验全部通过且完整方案总体MdAPE更低时更新当前最优索引；旧索引另存current_best_before.json。',
        '- LightGBM参数依据：https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html 。', '']
    (output/'实验报告.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    if args.trials < 4 or args.threads < 1:
        parser.error('At least four trials and positive threads required')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    validate_provenance()
    best = json.loads(BEST.read_text())
    info = best['off_market_fixed_four_months']
    if info['model_id'] != str(BASE_RUN.relative_to(REPO)):
        raise ValueError('Adapter expects the FLAML all_three baseline; best index changed')
    baseline_evidence, baseline_final = REPO/info['prediction_evidence'], REPO/info['final_predictions']
    for path, key in [(baseline_evidence, 'prediction_evidence_sha256'), (baseline_final, 'final_predictions_sha256')]:
        if sha(path) != info[key]:
            raise ValueError('Baseline evidence hash mismatch')
    # Validate the feature/model-producing baseline before reusing any artifacts.
    original_protocol = json.loads((BASE_RUN/'protocol.json').read_text())
    for key in ('source_sha256', 'input_sha256'):
        for path, digest in original_protocol[key].items():
            if sha(path) != digest:
                raise ValueError(f'Prior run provenance mismatch: {path}')
    raw, test = load_data(args.data_dir)
    previous_configs = json.loads((BASE_RUN/'selected_configs.json').read_text())['configs']
    output.mkdir(parents=True)
    shutil.copy2(BEST, output/'current_best_before.json')
    inputs = set(input_files(args.data_dir))
    inputs.update(p for p in BASE_RUN.rglob('*') if p.is_file())
    sources = set(HERE.glob('*.py')) | set(HERE.glob('*.md')) | set(HERE.glob('*.txt'))
    for package in ('flaml_tuning', 'dart_dropout', 'lightgbm_official', 'plan_a',
                    'property_groups', 'data_processing_round1', 'residual_refinement'):
        sources.update((REPO/'experiments'/package).glob('*.py'))
    sources.update([REPO/'tests/test_complex_trees.py', REPO/'data.py', REPO/'check_submission.py'])
    protocol = dict(created_at=now(), track='off_market', inner_windows=INNER_WINDOWS,
        outer_window=OUTER, final_cutoff=FINAL, history_updates=False,
        outer_labels_used_for_selection=False, previously_examined_windows=True,
        branches=list(TARGETS), search_spec=SEARCH_SPEC, trials_per_branch=args.trials,
        no_search_time_limit=True, seed=42, threads=args.threads, early_stopping_rounds=120,
        selection='pooled inner target-type MdAPE; ties by MAPE then trial index',
        final_rounds='rounded median of selected inner best iterations',
        forced_large_capacity=True, current_best=info, baseline_run=str(BASE_RUN),
        source_sha256={str(p): sha(p) for p in sorted(sources)},
        input_sha256={str(p): sha(p) for p in sorted(inputs)},
        protected_submissions={str(p): sha(p) for p in (REPO/'submissions').glob('*.csv')},
        versions={p: importlib.metadata.version(p) for p in ('flaml', 'lightgbm', 'numpy', 'pandas')})
    write_json(output/'protocol.json', protocol)
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for path in sorted(sources):
        shutil.copy2(path, snapshot/str(path.relative_to(REPO)).replace('/', '__'))
    designs = [(cutoff, end, load_designs(raw, test, cutoff, end)[1]) for cutoff, end in INNER_WINDOWS]
    configs = {}
    for target in TARGETS:
        folds = [(cutoff, end, block[target]) for cutoff, end, block in designs]
        configs[target] = search_branch(target, folds, previous_configs[target],
            output/'tuning'/target, args.trials, args.threads, 120)
    write_json(output/'selected_configs.json', dict(frozen_at=now(), configs=configs,
        inner_windows=INNER_WINDOWS, outer_labels_used=False))
    frozen_hash = sha(output/'selected_configs.json')
    del designs, folds
    homes, baseline, candidate = frozen_predict(raw, test, *OUTER, configs,
        output/'models', args.threads, baseline_evidence, BASE_RUN/'models')
    _, _, final = frozen_predict(raw, test, FINAL, None, configs,
        output/'final_models', args.threads, baseline_final, BASE_RUN/'final_models')
    folder = output/'final_predictions'
    folder.mkdir()
    final[['ml_num', 'predicted_price']].to_csv(folder/'submission_off_market_complex.csv', index=False)
    # Outer labels are only scored once both evaluation and final models exist.
    paired, table = comparison(homes, baseline, candidate)
    paired.to_csv(output/'paired_predictions.csv', index=False)
    table.to_csv(output/'comparison_to_current_best.csv', index=False)
    if sha(output/'selected_configs.json') != frozen_hash:
        raise ValueError('Frozen selection changed')
    write_json(output/'completion.json', dict(completed_at=now(), selected_sha256=frozen_hash))
    report(output, table, info['display_name'])
    from .verify_results import verify_run, promote
    verified = verify_run(output, args.data_dir)
    promote(output, verified)
    row = table.iloc[0]
    log(f'完成：当前最优MdAPE={row.baseline_mdape:.6f}%；大树={row.candidate_mdape:.6f}%；变化={row.delta_mdape_pp:+.6f}个百分点')


if __name__ == '__main__':
    main()
