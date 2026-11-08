"""Retrain the four routes specified in the conversation against one global baseline."""
import argparse
import importlib.metadata
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.plan_a.models import PARAMS
from experiments.plan_a.run_experiment import metrics
from experiments.property_groups.models import GROUPS, group_labels
from experiments.property_groups.run_experiment import (
    feature_pair, final_refit, fit_one, now, scores_bundle, sha, split_window, write_json,
)
from experiments.property_groups.selection import CHECK_WINDOW, DEV_WINDOWS, route_rows

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOG = logging.getLogger(__name__)
BASELINE = "global__lightgbm"
FIXED_ROUTES = {
    "apartments": "apartments__catboost",
    "condo_townhouse": "condo_townhouse__lightgbm",
    "detached": "detached__lightgbm",
    "attached_freehold": "attached_freehold__lightgbm",
    "__default__": BASELINE,
}


def paired_comparison(routed, baseline):
    """Align by ID, require identical labels, then compute genuinely paired scores."""
    left = routed.sort_values("ml_num").reset_index(drop=True)
    right = baseline.sort_values("ml_num").reset_index(drop=True)
    if left.ml_num.duplicated().any() or right.ml_num.duplicated().any():
        raise ValueError("Duplicate IDs in paired comparison")
    columns = ["ml_num", "sold_price", "house_type_name", "group", "month", "cutoff_exclusive"]
    if len(left) != len(right) or not left[columns].equals(right[columns]):
        raise ValueError("Paired comparison requires identical IDs, labels, and metadata")
    result = []
    for dimension in ("overall", "house_type_name", "group", "month", "cutoff_exclusive"):
        groups = [("all", left.index)] if dimension == "overall" else left.groupby(dimension).groups.items()
        for label, positions in groups:
            current, reference = metrics(left.loc[positions]), metrics(right.loc[positions])
            result.append({"dimension": dimension, "slice": label, **current,
                           "baseline_mdape": reference["mdape"], "baseline_mape": reference["mape"],
                           "baseline_median_signed_error": reference["median_signed_error"],
                           "delta_mdape_pp": current["mdape"] - reference["mdape"],
                           "delta_mape_pp": current["mape"] - reference["mape"]})
    return pd.DataFrame(result)


def evaluate_window(listings, out, cutoff, end, protocol):
    train, homes = split_window(listings, cutoff, end)
    folder = out / protocol / cutoff[:7]
    folder.mkdir(parents=True)
    LOG.info("[%s] %s train=%s evaluate=%s", protocol, cutoff, len(train), len(homes))
    history, x_train, x_valid = feature_pair(train, homes, cutoff)
    train_groups, valid_groups = group_labels(train), group_labels(homes)
    frames, audits = [], {}
    for identifier in sorted(set(FIXED_ROUTES.values())):
        scope, algorithm = identifier.split("__")
        train_pos = np.flatnonzero(np.ones(len(train), dtype=bool) if scope == "global" else train_groups.eq(scope))
        valid_pos = np.flatnonzero(np.ones(len(homes), dtype=bool) if scope == "global" else valid_groups.eq(scope))
        if len(train_pos) < 100 or not len(valid_pos):
            raise ValueError(f"Insufficient coverage: {identifier}")
        prediction, detail = fit_one(scope, algorithm, x_train.iloc[train_pos], train.sold_price.iloc[train_pos],
                                     x_valid.iloc[valid_pos], folder / identifier, cutoff)
        rows = homes.iloc[valid_pos][["ml_num", "sold_price", "house_type_name", "community"]].copy()
        rows["month"] = homes.date_sold.iloc[valid_pos].dt.strftime("%Y-%m")
        rows["group"] = valid_groups.iloc[valid_pos]
        rows["predicted_price"] = prediction
        rows["candidate"], rows["scope"], rows["algorithm"] = identifier, scope, algorithm
        rows["protocol"], rows["cutoff_exclusive"] = protocol, cutoff
        rows.to_csv(folder / identifier / "predictions.csv", index=False)
        detail["scores"] = metrics(rows)
        audits[identifier] = detail
        frames.append(rows)
        LOG.info("  %-34s MdAPE=%.3f%% (%0.1fs)", identifier, detail["scores"]["mdape"], detail["seconds"])
    write_json(folder / "audit.json", {
        "cutoff_exclusive": cutoff, "end_exclusive": end,
        "training_n": len(train), "evaluation_n": len(homes),
        "max_training_sale": str(train.date_sold.max()), "history_reference_n": history.reference_n,
        "history_query": "[min(date_listed, cutoff)-lookback, min(date_listed, cutoff))",
        "evaluation_labels_used_for_fit_or_early_stopping": False, "models": audits,
    })
    return pd.concat(frames, ignore_index=True)


def write_report(out, comparisons, scores, submission):
    check = comparisons.loc[comparisons.protocol.eq("check")]
    lines = ["# 四组固定模型：重新训练与验证结果", "", f"完成时间：{now()}。范围：Off-market。", "",
        "本次新增固定路由入口，复用现有 property_groups 的特征、训练、保存和推理实现，并重新训练。",
        "分组、算法与项目已有四组方案一致，因此这是可复现运行，不代表新增方法带来的改善。", "",
        "| 模型组 | 房型 | 固定模型 |", "|---|---|---|"]
    for group, spec in GROUPS.items():
        lines.append(f"| {spec['label']} | {', '.join(spec['types'])} | {FIXED_ROUTES[group]} |")
    lines += ["", "## 验证口径", "",
        "- 参数固定：600 轮，log-price MAE/L1，种子 42；不根据本次分数切换路由或调参。",
        "- 公寓两组使用 59 项输入，另两组及统一 LightGBM 使用 64 项输入；历史池和截止日一致。",
        "- 2025 年三个四个月窗口报告稳定性；截至 2026 年 1 月训练，固定预测 2—5 月作为后续回测。",
        "- 这四个窗口在项目中均已查看，属于累计开发，不能称为从未接触的独立测试。",
        "- 模型按成交截止日训练；不是逐套挂牌日重训仿真。未来四个月不补充窗口内成交答案。",
        "- 对照为统一 LightGBM；差值按相同房屋逐套对齐后计算，负数表示 MdAPE 改善。", "",
        "## 2026 年 2—5 月：六种房型", "",
        "| 房型 | 套数 | 四组 MdAPE | 统一 MdAPE | 变化（百分点） | 四组 MAPE | 四组中位偏差 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in check.loc[check.dimension.eq("house_type_name")].to_dict("records"):
        lines.append(f"| {row['slice']} | {row['n']:,} | {row['mdape']:.2f}% | {row['baseline_mdape']:.2f}% | "
                     f"{row['delta_mdape_pp']:+.3f} | {row['mape']:.2f}% | {row['median_signed_error']:+.2f}% |")
    overall = check.loc[check.dimension.eq("overall")].iloc[0]
    lines += ["", f"整体 {int(overall.n):,} 套：MdAPE {overall.baseline_mdape:.3f}% → {overall.mdape:.3f}%，"
              f"变化 {overall.delta_mdape_pp:+.3f} 个百分点；四组 MAPE {overall.mape:.3f}%。",
        "整体中位数由所有逐套误差合并计算，不是六种房型分数的平均。", "",
        "## 跨窗口结果", "", "| 截止日 | 套数 | 四组 MdAPE | 统一 MdAPE | 变化（百分点） |",
        "|---|---:|---:|---:|---:|"]
    for row in comparisons.loc[comparisons.dimension.eq("cutoff_exclusive")].to_dict("records"):
        lines.append(f"| {row['slice']} | {row['n']:,} | {row['mdape']:.3f}% | {row['baseline_mdape']:.3f}% | {row['delta_mdape_pp']:+.3f} |")
    lines += ["", "## 如何解读", "",
        "- 分组收益同时包含训练范围、算法和公寓土地列处理的变化，不能单独归因于其中一个因素。",
        "- 合作公寓单独样本较少，其成绩不能被普通公寓的合并成绩替代。",
        "- 半独立屋、自有产权联排要分别检查；不能因为共享组整体较好就宣称两类均改善。",
        "- 2026 年 6—9 月没有公开成交答案，不能报告隐藏测试 MdAPE。",
        "- 未按本次回测结果再校准价格或更改模型；统计显著性和未来收益尚未证实。", "",
        "## 产物与复现", "",
        f"重新训练 20 个回测模型和 5 个全量模型，均保存后重载核对预测。隐藏输入输出 {submission['n']:,} 行，官方格式检查通过。",
        "全量四组模型外另保存统一 LightGBM 作为未知房型的回退；当前六种房型均走固定专用路由。",
        "六房型与逐月、逐窗口对照见 comparison.csv；逐套预测见 check_routed_predictions.csv。",
        "verification.json 保存独立复算、边界检查和最终模型预测重放结果。", "",
        "```bash",
        ".venv/bin/python -u -m experiments.fixed_four_groups.run_experiment --output experiments/fixed_four_groups/artifacts/my_run",
        ".venv/bin/python -m experiments.fixed_four_groups.verify_results --run experiments/fixed_four_groups/artifacts/my_run",
        "```", ""]
    (out / "结果报告.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log")])
    sources = sorted({*HERE.glob("*.py"), *list((REPO / "experiments/property_groups").glob("*.py")),
                      REPO / "experiments/plan_a/features.py", REPO / "experiments/plan_a/models.py",
                      REPO / "experiments/plan_a/run_experiment.py", REPO / "data.py", REPO / "check_submission.py"})
    inputs = [args.data_dir / "listings.parquet", args.data_dir / "test_off_market.parquet"]
    protected = {str(p.resolve()): sha(p) for p in [*sources, *inputs]}
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot / str(path.relative_to(REPO)).replace("/", "__"))
    manifest = {"status": "running", "started_at_Toronto": now(), "python": sys.version,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "lightgbm", "catboost", "pyarrow")},
        "groups": GROUPS, "params": PARAMS, "choices": FIXED_ROUTES, "baseline": BASELINE,
        "development_windows": DEV_WINDOWS, "check_window": CHECK_WINDOW,
        "prior_project_use_of_all_evaluation_periods": True, "protected_sha256": protected}
    write_json(out / "manifest.json", manifest)
    write_json(out / "selection.json", {"choices": FIXED_ROUTES, "policy": "fixed in conversation before this run",
        "frozen_at_Toronto": now(), "this_run_labels_used_for_routing": False})
    selection_sha = sha(out / "selection.json")
    listings, test = (pd.read_parquet(path) for path in inputs)
    for column in ("date_listed", "date_sold"):
        listings[column] = pd.to_datetime(listings[column])
    test["date_listed"] = pd.to_datetime(test.date_listed)
    if listings.ml_num.duplicated().any() or test.ml_num.duplicated().any() or set(listings.ml_num).intersection(test.ml_num):
        raise ValueError("Duplicate or overlapping IDs")
    if not listings.date_sold.lt("2026-06-01").all() or not listings.sold_price.gt(0).all():
        raise ValueError("Invalid label availability or prices")
    group_labels(listings)
    group_labels(test)
    frames = [evaluate_window(listings, out, start, end, "development") for start, end in DEV_WINDOWS]
    frames.append(evaluate_window(listings, out, *CHECK_WINDOW, "check"))
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(out / "all_candidate_predictions.csv", index=False)
    scores, comparisons = {}, []
    for protocol in ("development", "check"):
        frame = rows.loc[rows.protocol.eq(protocol)]
        routed = route_rows(frame, FIXED_ROUTES)
        baseline = frame.loc[frame.candidate.eq(BASELINE)]
        routed.to_csv(out / f"{protocol}_routed_predictions.csv", index=False)
        scores[protocol] = {"routed": scores_bundle(routed), BASELINE: scores_bundle(baseline)}
        comparisons.append(paired_comparison(routed, baseline).assign(protocol=protocol))
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(out / "comparison.csv", index=False)
    write_json(out / "metrics.json", scores)
    submission = final_refit(listings, test, out, FIXED_ROUTES)
    if selection_sha != sha(out / "selection.json") or not all(sha(path) == digest for path, digest in protected.items()):
        raise ValueError("A frozen route, source, or input changed during this run")
    write_json(out / "checks.json", {"selection_sha256": selection_sha, "protected_inputs_unchanged": True,
                                    "evaluation_models_reloaded": 20, "final_models_reloaded": 5, "submission": submission})
    write_report(out, comparison, scores, submission)
    from .verify_results import verify
    LOG.info("VERIFY: %s", verify(out, args.data_dir))
    manifest.update(status="complete", finished_at_Toronto=now(), artifact_sha256={
        str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file() and path.name not in ("manifest.json", "run.log")})
    write_json(out / "manifest.json", manifest)
    LOG.info("CHECK SCORES: %s", scores["check"]["routed"]["overall"])
    LOG.info("DONE: %s", out / "结果报告.md")


if __name__ == "__main__":
    main()
