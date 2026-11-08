"""Recompute scores, check frozen inputs, and replay persisted models."""
# 本文件有三项职责：predict_saved 重建最终预测；verify_run 核验一次完整实验；
# promote_if_better 在核验且改善后更新最优索引。main 默认只核验，--promote 才申请更新。
# 历史 protocol.json 记录的是当时源码的逐字节哈希；添加注释也会改变当前源码指纹。
# 本次保留历史协议与 source_snapshot，不重写历史证据；旧实验的严格源码哈希核验
# 会识别这些注释改动。单纯加载模型预测的 predict_saved 不执行这项全量哈希核验。

# 标准库负责参数/路径/JSON；NumPy/Pandas 用于逐套重放、表格对齐和指标比较。
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 项目依赖：官方提交格式检查、旧房型预测、特征构造器、原生 LightGBM 模型封装。
# HistoryFeatures 生成基础历史特征，ProcessingFeatures/assemble 补充 all_three 特征；
# LOT_COLUMNS 是公寓联排分支需要移除的地块特征列。
import check_submission
import data
from experiments.dart_dropout.run_experiment import (
    REPO, TARGETS, LAND, archived_baseline, load_designs, sha, now, write_json)
from experiments.data_processing_round1.features import ProcessingFeatures, assemble, text_features
from experiments.data_processing_round1.run_experiment import load_data
from experiments.lightgbm_official.models import OfficialLightGBM
from experiments.plan_a.features import HistoryFeatures
from experiments.property_groups.models import LOT_COLUMNS
from .models import replace_branch, validate_window
from .run_experiment import INNER, OUTER, FINAL, BEST, comparison


def predict_saved(run, raw, test):
    """Rebuild final query features from raw data, without the feature caches."""
    # 第 1 步：按成交日期/ID 排好历史参考池；隐藏测试答案及 Off-market 禁用字段。
    # HistoryFeatures 使用 FINAL 截止日，基于合法历史成交为每套待估房屋生成基础特征。
    reference = raw.sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    hidden = data.hide_answers(test, 'off_market').reset_index(drop=True)
    provider = HistoryFeatures(reference, FINAL)
    base = provider.transform(hidden)
    # 第 2 步：仅为独立屋、半独立屋建立增强输入：基础、历史、地块、文本四块，
    # assemble(..., 'all_three') 按前序方案组合；这里从原始数据重算，不读取特征缓存。
    land_homes = hidden.loc[hidden.house_type_name.isin(TARGETS[:2])].reset_index(drop=True)
    processing = ProcessingFeatures(reference, FINAL)
    processed = assemble(provider.transform(land_homes), processing.history(land_homes),
                         processing.land(land_homes), text_features(land_homes), 'all_three')
    # 第 3 步：从旧完整预测起步，为四个 LightGBM 房型加载本次最终模型并替换。
    # 独立屋/半独立屋用增强特征，其余两分支用基础特征；另外两类公寓保留旧预测。
    result = archived_baseline(test, FINAL, final=True)
    for target in TARGETS:
        homes, features = (land_homes, processed) if target in TARGETS[:2] else (hidden, base)
        mask = homes.house_type_name.eq(target)
        x = features.loc[mask]
        # 公寓联排移除地块列，与训练时使用的输入保持一致；模型加载后还会按保存的列/类别编码。
        if target == 'Condo Townhouse':
            x = x.drop(columns=list(LOT_COLUMNS))
        model = OfficialLightGBM.load(run/'final_models'/target)
        result = replace_branch(result, homes.loc[mask], model.predict(x))
    # 返回官方提交需要的两列；房型替换按 ID 对齐，完整预测的行顺序保持不变。
    return result[['ml_num', 'predicted_price']]


def verify_run(run, data_dir):
    # 第 1 项核验：读取实验协议并重新计算源码、输入数据及正式提交文件的哈希。
    # 任意字节变化（包括注释）都算变化，停止核验，避免把后来修改误当成原实验版本。
    run = Path(run).resolve()
    protocol = json.loads((run/'protocol.json').read_text())
    for key in ('source_sha256', 'input_sha256', 'protected_submissions'):
        for name, digest in protocol[key].items():
            if sha(name) != digest:
                raise ValueError(f'Changed {key}: {name}')
    # 第 2 项核验：冻结参数文件必须与完成记录中的哈希一致；
    # 同时检查未用外层答案调参的声明，以及冻结时间早于运行完成时间。
    # 这些是协议/文件一致性检查，本身不能单独证明整个开发过程从未看过外层答案。
    completion = json.loads((run/'completion.json').read_text())
    if sha(run/'selected_configs.json') != completion['selected_sha256']:
        raise ValueError('Frozen configuration hash mismatch')
    selection = json.loads((run/'selected_configs.json').read_text())
    if not selection['no_outer_labels'] or selection['frozen_at'] >= completion['completed_at']:
        raise ValueError('Invalid parameter-freeze audit')
    # 第 3 项核验：对每个房型，从试验记录中找到获选配置，确认其内层 MdAPE 最小。
    # 再检查保存的训练/验证 ID 的时间切分，并与原始数据中的真实成交日期逐套对照。
    raw, test = load_data(data_dir)
    for target in TARGETS:
        folder = run/'tuning'/target
        records = json.loads((folder/'trials.json').read_text())
        config = selection['configs'][target]
        winners = [r for r in records if r['config'] == config]
        if not winners or winners[0]['mdape'] != min(r['mdape'] for r in records):
            raise ValueError('Selected parameters differ from inner optimum')
        train = pd.read_csv(folder/'train_ids.csv')
        valid = pd.read_csv(folder/'valid_ids.csv')
        validate_window(train, valid, *INNER)
        for frame in (train, valid):
            actual = raw.set_index('ml_num').date_sold.reindex(frame.ml_num)
            if not np.array_equal(actual.to_numpy(), pd.to_datetime(frame.date_sold).to_numpy()):
                raise ValueError('Tuning IDs do not match raw sale dates')
    # 第 4 项核验：重新读取外层设计表，确认训练 ID、预测 ID 顺序与存档相同。
    # 加载每个外层模型，对保存的特征重放，预测价格必须逐套完全一致。
    homes, designs = load_designs(raw, test, *OUTER)
    for target in TARGETS:
        train, valid, _, xv = designs[target]
        saved = pd.read_csv(run/'models'/target/'train_ids.csv')
        if not np.array_equal(saved.ml_num.to_numpy(), train.ml_num.to_numpy()):
            raise ValueError('Outer training IDs mismatch')
        model = OfficialLightGBM.load(run/'models'/target)
        expected = pd.read_csv(run/'models'/target/'predictions.csv')
        if not np.array_equal(expected.ml_num.to_numpy(), valid.ml_num.to_numpy()):
            raise ValueError('Outer prediction order mismatch')
        if not np.array_equal(model.predict(xv), expected.predicted_price.to_numpy()):
            raise ValueError('Outer model replay mismatch')
    # 第 5 项核验：旧完整预测必须与运行开始时最优索引指定的证据一致。
    # 再从真实答案和两份预测独立重算整体/房型/月度指标，与发布的比较表对照。
    baseline = pd.read_csv(run/'models/baseline_predictions.csv')
    evidence = pd.read_csv(REPO/protocol['baseline_model']['prediction_evidence'])
    expected = evidence.set_index('ml_num').predicted_price.reindex(baseline.ml_num)
    if not np.array_equal(expected.to_numpy(), baseline.predicted_price.to_numpy()):
        raise ValueError('Current best comparison mismatch')
    candidate = pd.read_csv(run/'models/predictions_unscored.csv')
    paired, scores = comparison(homes, baseline, candidate)
    pd.testing.assert_frame_equal(pd.read_csv(run/'comparison_to_current_best.csv'), scores,
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    # 第 6 项核验：把四个保存的分支预测重新组装，确认完整候选确实遵守既定房型路由。
    # Ensure the published full candidate is assembled from its saved branches.
    reconstructed = baseline.copy()
    for target in TARGETS:
        _, valid, _, _ = designs[target]
        prices = pd.read_csv(run/'models'/target/'predictions.csv').predicted_price.to_numpy()
        reconstructed = replace_branch(reconstructed, valid, prices)
    if not np.array_equal(reconstructed.predicted_price.to_numpy(), candidate.predicted_price.to_numpy()):
        raise ValueError('Complete-model routing mismatch')
    # 第 7 项核验：官方检查器检查隐藏预测的格式与 ID 覆盖；随后从原始数据重建特征，
    # 用最终保存模型生成整份预测，要求 ID 顺序和价格均与保存的候选文件完全相同。
    path = run/'final_predictions/submission_off_market_flaml.csv'
    problems = check_submission.check_file(path, test.ml_num)
    if problems:
        raise ValueError(problems)
    final = pd.read_csv(path)
    replay = predict_saved(run, raw, test)
    if not (np.array_equal(final.ml_num.to_numpy(), replay.ml_num.to_numpy())
            and np.array_equal(final.predicted_price.to_numpy(), replay.predicted_price.to_numpy())):
        raise ValueError('Raw-feature final replay mismatch')
    # 第 8 项核验：重算的基线总体 MdAPE 必须与当时的最优索引一致，再保存本轮核验摘要与哈希。
    row = scores.iloc[0]
    if not np.isclose(row.baseline_mdape, protocol['baseline_model']['metrics_percent']['mdape'], atol=1e-10, rtol=0):
        raise ValueError('Baseline MdAPE differs from indexed current best')
    result = dict(verified_at=now(), n=int(row.n), baseline_mdape=float(row.baseline_mdape),
        candidate_mdape=float(row.candidate_mdape), delta_mdape_pp=float(row.delta_mdape_pp),
        candidate_mape=float(row.candidate_mape), source_and_inputs_unchanged=True,
        outer_model_replay=True, raw_final_replay=True, final_n=len(final),
        selected_sha256=completion['selected_sha256'], final_sha256=sha(path),
        prediction_sha256=sha(run/'models/predictions_unscored.csv'),
        paired_sha256=sha(run/'paired_predictions.csv'))
    write_json(run/'verification.json', result)
    return result


def promote_if_better(run, verified):
    # 先把实验目录转为绝对路径，再依据已完成的核验结果判断是否更新索引。
    run = Path(run).resolve()
    # 只有候选总体 MdAPE 低于基线才继续；没有改善则只记录原因，不更新最优索引。
    # 调用方应传入 verify_run 成功返回的结果；此函数本身不会再执行一遍全量核验。
    if verified['delta_mdape_pp'] >= 0:
        write_json(run/'promotion.json', dict(promoted=False, reason='No improvement on matched development backtest'))
        return
    # 读取运行前索引与当前索引；若已由同一实验晋升，则允许重复核验而不重复更新。
    before = json.loads((run/'current_best_before.json').read_text())
    current = json.loads(BEST.read_text())
    # Allow rerunning verification after this same run has already been promoted.
    new_id = str(run.relative_to(REPO))
    if current['off_market_fixed_four_months']['model_id'] == new_id:
        return
    # 若运行期间其他实验更新了最优索引，就停止，避免覆盖更新后的记录。
    if current != before:
        raise ValueError('Best index changed during tuning; refusing to overwrite')
    # 以本次候选更新固定四个月赛道的模型名、时间口径、指标、预测证据和来源链。
    # 更新的是累计开发回测最优索引，不覆盖正式提交，也不声明隐藏测试成绩。
    updated = dict(before)
    updated['recorded_date'] = now()[:10]
    updated['off_market_fixed_four_months'] = dict(
        model_id=new_id, display_name='FLAML自动调参完整方案（all_three特征与既有房型路由）',
        training_cutoff_exclusive=OUTER[0], evaluation_start_inclusive=OUTER[0],
        evaluation_end_exclusive=OUTER[1], reference_pool_updates_within_evaluation=False,
        selection_metric='MdAPE', metrics_percent=dict(n=verified['n'],
            mdape=verified['candidate_mdape'], mape=verified['candidate_mape']),
        prediction_evidence=str((run/'models/predictions_unscored.csv').relative_to(REPO)),
        prediction_evidence_sha256=verified['prediction_sha256'],
        final_predictions=str((run/'final_predictions/submission_off_market_flaml.csv').relative_to(REPO)),
        final_predictions_sha256=verified['final_sha256'],
        lineage=[before['off_market_fixed_four_months']['model_id'], new_id],
        comparison_note='仅2025年9—12月窗口调参，冻结后同7773套2026年2—5月累计开发回测；非隐藏测试成绩，未覆盖正式提交。')
    # 写入新索引，并在实验目录记录晋升时间、旧索引位置及改善幅度。
    write_json(BEST, updated)
    write_json(run/'promotion.json', dict(promoted=True, promoted_at=now(),
        prior_index='current_best_before.json', delta_mdape_pp=verified['delta_mdape_pp']))


def main():
    # 独立核验入口：--run 指定实验目录；默认只核验并打印摘要。
    # 只有显式给出 --promote，才在核验成功后尝试更新最优索引。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--promote', action='store_true')
    args = parser.parse_args()
    verified = verify_run(args.run, args.data_dir)
    if args.promote:
        promote_if_better(args.run, verified)
    print(json.dumps(verified, ensure_ascii=False, indent=2))


# 直接作为模块运行时执行核验入口；被训练或预测脚本导入时只提供函数。
if __name__ == '__main__':
    main()
