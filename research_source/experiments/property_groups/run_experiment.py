"""Train four property groups, freeze 2025 choices, check February-May, then refit."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.plan_a.features import FORBIDDEN, HistoryFeatures
from experiments.plan_a.models import PARAMS
from experiments.plan_a.run_experiment import metrics
from .models import ALGORITHMS, GROUPS, GroupModel, candidate_id, group_labels, routed_predict
from .selection import CHECK_WINDOW, DEV_WINDOWS, choose_routes, route_rows

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOG = logging.getLogger(__name__)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(ZoneInfo("America/Toronto")).isoformat()


def split_window(listings, cutoff, end):
    cutoff, end = pd.Timestamp(cutoff), pd.Timestamp(end)
    train = listings.loc[listings.date_sold < cutoff].sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    homes = listings.loc[(listings.date_sold >= cutoff) & (listings.date_sold < end)].reset_index(drop=True)
    if train.empty or homes.empty or set(train.ml_num).intersection(homes.ml_num):
        raise ValueError("Empty or overlapping train/evaluation sets")
    return train, homes


def feature_pair(train, homes, cutoff):
    history = HistoryFeatures(train, cutoff)
    hidden = data.hide_answers(homes, "off_market")
    x_train, x_homes = history.transform(train), history.transform(hidden)
    if list(x_train.columns) != list(x_homes.columns) or FORBIDDEN.intersection(x_train.columns):
        raise ValueError("Feature schema or inference contract violation")
    return history, x_train, x_homes


def fit_one(scope, name, x_train, prices, x_valid, folder, cutoff):
    started = time.monotonic()
    model = GroupModel(scope, name).fit(x_train, prices)
    prediction = model.predict(x_valid)
    model.save_bundle(folder, cutoff, len(x_train))
    replay = GroupModel.load_bundle(folder).predict(x_valid)
    if not np.array_equal(prediction, replay):
        raise ValueError(f"Saved model replay mismatch: {scope}/{name}")
    importance = (model.model.get_feature_importance() if name == "catboost"
                  else model.model.booster_.feature_importance(importance_type="gain"))
    pd.DataFrame({"feature": model.feature_names, "importance": importance}).sort_values(
        "importance", ascending=False).to_csv(folder / "feature_importance.csv", index=False)
    return prediction, {"training_n": len(x_train), "feature_count": len(model.feature_names),
                        "features": model.feature_names, "reload_exact": True,
                        "seconds": time.monotonic() - started}


def evaluate_window(listings, out, cutoff, end, protocol):
    train, homes = split_window(listings, cutoff, end)
    folder = out / protocol / cutoff[:7]
    folder.mkdir(parents=True)
    LOG.info("[%s] %s: train=%s, evaluation=%s", protocol, cutoff, f"{len(train):,}", f"{len(homes):,}")
    history, x_train, x_valid = feature_pair(train, homes, cutoff)
    train_groups, valid_groups = group_labels(train), group_labels(homes)
    frames, audits = [], {}
    for scope in ("global", *GROUPS):
        train_pos = np.flatnonzero(np.ones(len(train), dtype=bool) if scope == "global" else train_groups.eq(scope))
        valid_pos = np.flatnonzero(np.ones(len(homes), dtype=bool) if scope == "global" else valid_groups.eq(scope))
        if len(train_pos) < 100 or not len(valid_pos):
            raise ValueError(f"Insufficient group coverage: {scope}")
        for name in ALGORITHMS:
            identifier = candidate_id(scope, name)
            prediction, detail = fit_one(scope, name, x_train.iloc[train_pos], train.sold_price.iloc[train_pos],
                                          x_valid.iloc[valid_pos], folder / identifier, cutoff)
            rows = homes.iloc[valid_pos][["ml_num", "sold_price", "house_type_name", "community"]].copy()
            rows["month"] = homes.date_sold.iloc[valid_pos].dt.strftime("%Y-%m")
            rows["group"] = valid_groups.iloc[valid_pos]
            rows["predicted_price"] = prediction
            rows["candidate"] = identifier
            rows["algorithm"] = name
            rows["scope"] = scope
            rows["protocol"] = protocol
            rows["cutoff_exclusive"] = cutoff
            rows.to_csv(folder / identifier / "predictions.csv", index=False)
            detail["scores"] = metrics(rows)
            audits[identifier] = detail
            frames.append(rows)
            LOG.info("  %-34s MdAPE=%6.3f%% MAPE=%6.3f%% (%5.1fs)", identifier,
                     detail["scores"]["mdape"], detail["scores"]["mape"], detail["seconds"])
    write_json(folder / "audit.json", {"cutoff_exclusive": cutoff, "end_exclusive": end,
        "training_n": len(train), "evaluation_n": len(homes), "max_training_sale": str(train.date_sold.max()),
        "training_group_counts": {k: int(v) for k, v in train_groups.value_counts().items()},
        "evaluation_group_counts": {k: int(v) for k, v in valid_groups.value_counts().items()},
        "history_reference_n": history.reference_n,
        "history_query": "[min(date_listed, cutoff)-lookback, min(date_listed, cutoff))",
        "shared_history_definition_and_pool_for_all_candidates": True,
        "evaluation_labels_used_for_fit_or_early_stopping": False, "models": audits})
    return pd.concat(frames, ignore_index=True)


def breakdown(rows, keys):
    result = []
    for key, frame in rows.groupby(keys, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        result.append({**dict(zip(keys, key)), **metrics(frame)})
    return pd.DataFrame(result)


def scores_bundle(rows):
    return {"overall": metrics(rows), "by_group": breakdown(rows, ["group"]).to_dict("records"),
            "by_type": breakdown(rows, ["house_type_name"]).to_dict("records"),
            "by_month": breakdown(rows, ["month"]).to_dict("records"),
            "by_group_month": breakdown(rows, ["group", "month"]).to_dict("records"),
            "by_window": breakdown(rows, ["cutoff_exclusive"]).to_dict("records")}


def final_refit(listings, test, out, choices):
    train = listings.sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    cutoff = "2026-06-01"
    _, x_train, x_test = feature_pair(train, test, cutoff)
    groups = group_labels(train)
    identifiers = sorted(set(choices.values()))
    loaded, audit = {}, {}
    for identifier in identifiers:
        scope, name = identifier.split("__")
        pos = np.flatnonzero(np.ones(len(train), dtype=bool) if scope == "global" else groups.eq(scope))
        folder = out / "final_models" / identifier
        _, audit[identifier] = fit_one(scope, name, x_train.iloc[pos], train.sold_price.iloc[pos], x_test, folder, cutoff)
        loaded[identifier] = GroupModel.load_bundle(folder)
        LOG.info("[final] %-34s training_n=%s", identifier, f"{len(pos):,}")
    price, routes = routed_predict(x_test, test, choices, loaded)
    folder = out / "final_predictions"
    folder.mkdir()
    path = folder / "submission_off_market.csv"
    pd.DataFrame({"ml_num": test.ml_num, "predicted_price": price}).to_csv(path, index=False)
    errors = check_submission.check_file(path, test.ml_num.tolist())
    if errors:
        raise ValueError(errors)
    pd.DataFrame({"ml_num": test.ml_num, "house_type_name": test.house_type_name,
                  "group": group_labels(test), "selected_candidate": routes,
                  "predicted_price": price}).to_csv(folder / "prediction_routes.csv", index=False)
    write_json(out / "final_models" / "routes.json", {"choices": choices, "cutoff_exclusive": cutoff,
                                                       "models": audit})
    return {"n": len(test), "file": str(path.relative_to(out)), "errors": errors,
            "sha256": sha(path), "all_selected_models_reloaded_exactly": True}


def write_report(out, listings, choices, selection_scores, scores, comparison, checks):
    lines = ["# 四组房型建模：步骤与实测结果", "", "日期：2026-10-17（多伦多）。研究范围为 Off-market。", "",
        "## 1. 预先固定的分组", "", "| 组 | 房型 | 全量样本数 | 处理依据 |", "|---|---|---:|---|"]
    reasons = {"apartments": "两类 Apartment 共用；Co-Op 仅 729 条，不单独拟合。去掉土地相关五列。",
               "condo_townhouse": "与普通公寓的面积和样式不同；去掉土地相关五列。",
               "detached": "土地、室内面积、房况和停车共同建模，保留土地特征。",
               "attached_freehold": "两类有土地的相邻/联排住宅共享学习；保留具体房型标签和土地特征。"}
    labels = group_labels(listings)
    for group, spec in GROUPS.items():
        lines.append(f"| {spec['label']} | {', '.join(spec['types'])} | {int(labels.eq(group).sum()):,} | {reasons[group]} |")
    lines += ["", "这些是按可观测特征和样本规模预先划定的实验组，并非已经证明最优的分组。",
        "", "## 2. 训练、选择、回测与最终预测", "",
        "1. 读取原始 parquet；检查日期、正价格、房型和 ID，不改原始数据。",
        "2. 每个截止日仅用更早的成交，构建原方案相同的 90/180/365 天历史特征。所有候选使用同一个历史池，避免把统计口径变化混入分组效果。",
        "3. 每组分别训练 LightGBM 与 CatBoost，另训两个全房型统一模型。公寓两组删除五个土地常量字段；其余使用原 64 项特征。",
        "4. 算法参数沿用方案 A；目标为 log(sold_price) 的 L1/MAE，600 轮、种子 42、无验证早停。不同房型先不额外调参。",
        "5. 在 2025 年 1—4 月、5—8 月、9—12 月三个固定四个月窗口产生向前预测；每组比较两个专用模型和两个统一模型。按全年逐套合并 MdAPE 最低选择，MAPE 打破平局，再以候选名称确定完全相同的结果。",
        "6. 在看到 2026 年 2—5 月结果前写入 selection.json，冻结每组选择。训练截至 2026 年 1 月、固定预测 2—5 月，模型和历史池均不刷新，不依据该段结果换模型。",
        "7. 对整体、四组、六种房型和逐月误差分别评分；整体 MdAPE 从合并逐套预测计算，不平均分组 MdAPE。",
        "8. 冻结选择后，用截至 2026 年 5 月已知成交重训所需模型；保存、重载并核对预测，生成隔离的 Off-market CSV，不覆盖现有提交。",
        "", "## 3. 每组选用什么模型，以及误差多少", "",
        "下表选择来自较早的 2025 年开发窗口；误差来自随后固定的 2026 年 2—5 月回测。global 表示统一模型，其他 scope 表示专用模型。",
        "", "| 组 | 选用模型 | 验证套数 | MdAPE | MAPE | 中位有符号误差 | 同组统一基线 MdAPE |", "|---|---|---:|---:|---:|---:|---:|"]
    global_id = choices["__default__"]
    for group, spec in GROUPS.items():
        s = next(x for x in scores["check"]["routed"]["by_group"] if x["group"] == group)
        b = comparison.loc[(comparison.group == group) & (comparison.candidate == global_id)].iloc[0]
        lines.append(f"| {spec['label']} | {choices[group]} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% | {s['median_signed_error']:+.2f}% | {b.mdape:.2f}% |")
    lines += ["", f"统一基线算法也只按 2025 年开发成绩选择：{global_id}。每组更详细的四候选成绩见 group_comparison.csv。", "",
              "## 4. 合并后的整体误差", "", "| 方案 | 套数 | MdAPE | MAPE |", "|---|---:|---:|---:|"]
    for identifier, bundle in scores["check"].items():
        s = bundle["overall"]
        lines.append(f"| {identifier} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% |")
    delta = scores["check"]["routed"]["overall"]["mdape"] - scores["check"][global_id]["overall"]["mdape"]
    lines += ["", f"分组路由相对预先选定统一基线的 MdAPE 变化：{delta:+.3f} 个百分点，负数表示改善。是否改善据实际结果报告，不强制启用专用模型。",
        "", "## 5. 六种房型单独误差", "", "| 房型 | 套数 | MdAPE | MAPE |", "|---|---:|---:|---:|"]
    for s in scores["check"]["routed"]["by_type"]:
        lines.append(f"| {s['house_type_name']} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% |")
    lines += ["", "## 6. 解释与限制", "",
        "- 房型分组让每组有独立的训练选择；如果较早开发成绩更支持统一模型，该组明确回退，不把分组必然有效当作前提。",
        "- 删除公寓组全部缺失的土地列是输入清理，不能单独归因成准确率收益；本次实验同时包含分组训练和这项清理。",
        "- 统一树模型本来就包含房型特征与同房型历史统计，分组模型需要证明额外价值。模型内部特征重要性不代表因果解释，两个算法的重要性数值不可直接比较。",
        "- Co-Op 的验证样本很少，单独分数波动大；不要凭这一小组选复杂模型。",
        "- 本项目 2025 年窗口和 2026 年 2—5 月此前均参与过实验开发；本轮冻结选择避免了本轮后验挑选，但它们不是整个项目从未接触的独立测试。",
        "- 历史查询早于 min(挂牌日, 训练截止日)，模型训练样本按成交截止日选择；这是固定成交窗口回测，不是每个挂牌日实时重训的完整仿真。",
        "- 隐藏测试缺少实际成交日期；输入不使用该日期、实际成交等待天数、挂牌价或第三方估值。6—9 月没有公开标签，不能报告该段误差。",
        "", "## 7. 测试与复现", "",
        f"所有窗口的 40 个候选模型与最终所需模型都完成保存后重载，逐套预测完全一致。隐藏测试输出 {checks['n']:,} 行，官方格式检查通过。",
        "单元/集成测试和独立误差复算记录另见 checks.json、verification.json 与 pytest.log。",
        "", "在 SSU-Data-Dash 目录运行，输出路径必须不存在：", "", "```bash",
        ".venv/bin/python -u -m experiments.property_groups.run_experiment --output experiments/property_groups/artifacts/my_new_run",
        "```", "", "复用保存模型重新生成预测：", "", "```bash",
        ".venv/bin/python -m experiments.property_groups.predict_saved --run experiments/property_groups/artifacts/my_new_run --output /tmp/property_groups_replay.csv",
        "```", "", "README.md 解释每个文件与步骤；selection.json 保存选择规则、候选分数及冻结时间。", ""]
    (out / "训练步骤与结果报告.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "run_2026-10-17_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log")])
    sources = [*HERE.glob("*.py"), REPO / "data.py", REPO / "check_submission.py",
               REPO / "experiments/plan_a/features.py", REPO / "experiments/plan_a/models.py",
               REPO / "experiments/plan_a/run_experiment.py"]
    input_paths = [args.data_dir / "listings.parquet", args.data_dir / "test_off_market.parquet"]
    protected = {str(p.resolve()): sha(p) for p in [*sources, *input_paths]}
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for p in sources:
        shutil.copy2(p, snapshot / str(p.relative_to(REPO)).replace("/", "__"))
    manifest = {"started_at_Toronto": now(), "python": sys.version, "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in ("catboost", "lightgbm", "pandas", "numpy", "pyarrow")},
        "protected_sha256": protected, "groups": GROUPS, "params": PARAMS,
        "development_windows": DEV_WINDOWS, "check_window": CHECK_WINDOW,
        "selection_rule": "2025 pooled MdAPE, then MAPE, then candidate ID; four candidates per group",
        "prior_project_use_of_both_development_and_check_periods": True,
        "status": "running"}
    write_json(out / "manifest.json", manifest)
    listings, test = (pd.read_parquet(p) for p in input_paths)
    for c in ("date_listed", "date_sold"):
        listings[c] = pd.to_datetime(listings[c])
    test["date_listed"] = pd.to_datetime(test.date_listed)
    if listings.ml_num.duplicated().any() or test.ml_num.duplicated().any() or set(listings.ml_num).intersection(test.ml_num):
        raise ValueError("Duplicate or overlapping IDs")
    if not (listings.date_sold < "2026-06-01").all() or not (listings.sold_price > 0).all():
        raise ValueError("Invalid training label availability or price")
    group_labels(listings)
    group_labels(test)
    development = pd.concat([evaluate_window(listings, out, start, end, "development")
                             for start, end in DEV_WINDOWS], ignore_index=True)
    choices, candidate_scores = choose_routes(development)
    write_json(out / "selection.json", {"choices": choices, "scores": candidate_scores,
        "frozen_at_Toronto": now(), "selection_period": "2025-01 through 2025-12",
        "check_labels_used": False, "params": PARAMS})
    selection_sha = sha(out / "selection.json")
    LOG.info("FROZEN ROUTES: %s", choices)
    check = evaluate_window(listings, out, *CHECK_WINDOW, "check")
    routed_dev, routed_check = route_rows(development, choices), route_rows(check, choices)
    routed_dev.to_csv(out / "development_routed_predictions.csv", index=False)
    routed_check.to_csv(out / "check_routed_predictions.csv", index=False)
    all_predictions = pd.concat([development, check], ignore_index=True)
    all_predictions.to_csv(out / "all_candidate_predictions.csv", index=False)
    comparison = breakdown(all_predictions, ["protocol", "group", "candidate"])
    comparison["selected"] = comparison.apply(lambda row: choices[row.group] == row.candidate, axis=1)
    comparison.to_csv(out / "group_comparison.csv", index=False)
    scores = {}
    for protocol, frame, routed in (("development", development, routed_dev), ("check", check, routed_check)):
        scores[protocol] = {"routed": scores_bundle(routed)}
        for name in ALGORITHMS:
            identifier = candidate_id("global", name)
            scores[protocol][identifier] = scores_bundle(frame.loc[frame.candidate == identifier])
    write_json(out / "metrics.json", scores)
    breakdown(routed_check, ["group", "month"]).to_csv(out / "group_month_errors.csv", index=False)
    breakdown(routed_check, ["house_type_name"]).to_csv(out / "property_type_errors.csv", index=False)
    submission_checks = final_refit(listings, test, out, choices)
    write_json(out / "submission_checks.json", submission_checks)
    if sha(out / "selection.json") != selection_sha:
        raise ValueError("Frozen model choices changed during evaluation")
    unchanged = all(sha(path) == old for path, old in protected.items())
    if not unchanged:
        raise ValueError("An input or source changed during this experiment")
    write_json(out / "checks.json", {"protected_inputs_and_sources_unchanged": unchanged,
        "selection_frozen_before_check": True, "selection_sha256": selection_sha,
        "candidate_model_reloads_exact": 40, "submission": submission_checks})
    write_report(out, listings, choices, candidate_scores, scores,
                 comparison.loc[comparison.protocol == "check"], submission_checks)
    manifest.update(status="complete", finished_at_Toronto=now(),
                    artifact_sha256={str(p.relative_to(out)): sha(p) for p in out.rglob("*")
                                     if p.is_file() and p.name not in ("manifest.json", "run.log")})
    write_json(out / "manifest.json", manifest)
    LOG.info("FINAL CHECK: %s", scores["check"]["routed"]["overall"])
    LOG.info("DONE: %s", out / "训练步骤与结果报告.md")


if __name__ == "__main__":
    main()
