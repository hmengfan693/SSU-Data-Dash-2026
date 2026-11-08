"""Run from repository root: .venv/bin/python -m experiments.plan_a.run_experiment

[A1] Earlier four-month development windows.
[A4] Fixed 2026-02-01 cutoff, predict February-May without label updates.
[B]  Official monthly retraining, reported separately.
[A5] Freeze choice, refit through May, predict the 3,974 hidden-test homes.
"""
import argparse
import hashlib
import importlib.metadata
import json
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
from models import NearestSalesBaseline
from .features import HistoryFeatures
from .models import PARAMS, PriceModel

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_PLAN = REPO.parent / "Off-market训练与验证方案.md"
MODELS = ["catboost", "lightgbm"]
DEV = [("2025-01-01", "2025-05-01"), ("2025-05-01", "2025-09-01"), ("2025-09-01", "2026-01-01")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def metrics(rows):
    signed = 100 * (rows.predicted_price.to_numpy() / rows.sold_price.to_numpy() - 1)
    return {"n": len(rows), "mdape": float(np.median(np.abs(signed))),
            "mape": float(np.mean(np.abs(signed))), "median_signed_error": float(np.median(signed)),
            "over_30_pct": float(np.mean(np.abs(signed) > 30) * 100)}


def result_rows(homes, predicted, name, protocol, cutoff):
    rows = homes[["ml_num", "sold_price", "house_type_name", "community"]].copy()
    rows["month"] = homes.date_sold.dt.strftime("%Y-%m")
    rows["predicted_price"] = predicted
    rows["model"] = name
    rows["protocol"] = protocol
    rows["cutoff_exclusive"] = cutoff
    return rows


def evaluate_window(listings, out, cutoff, end, protocol, baseline=False):
    train = listings.loc[listings.date_sold < pd.Timestamp(cutoff)].sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    homes = listings.loc[(listings.date_sold >= pd.Timestamp(cutoff)) & (listings.date_sold < pd.Timestamp(end))].reset_index(drop=True)
    assert len(train) and len(homes) and not set(train.ml_num).intersection(homes.ml_num)
    folder = out / protocol / cutoff[:7]
    folder.mkdir(parents=True)
    hidden = data.hide_answers(homes, "off_market")
    print(f"[{protocol}] cutoff={cutoff}, train={len(train):,}, evaluate={len(homes):,}", flush=True)
    history = HistoryFeatures(train, cutoff)
    x_train, x_valid = history.transform(train), history.transform(hidden)
    assert list(x_train.columns) == list(x_valid.columns)
    all_rows = []
    details = {}
    for name in MODELS:
        start = time.monotonic()
        model = PriceModel(name).fit(x_train, train.sold_price)
        predicted = model.predict(x_valid)
        rows = result_rows(homes, predicted, name, protocol, cutoff)
        rows.to_csv(folder / f"{name}_predictions.csv", index=False)
        importance = (model.model.get_feature_importance() if name == "catboost"
                      else model.model.booster_.feature_importance(importance_type="gain"))
        pd.DataFrame({"feature": x_train.columns, "importance": importance}).sort_values("importance", ascending=False).to_csv(folder / f"{name}_feature_importance.csv", index=False)
        details[name] = {**metrics(rows), "fit_predict_seconds": time.monotonic() - start}
        print(f"  {name}: MdAPE={details[name]['mdape']:.3f}%, MAPE={details[name]['mape']:.3f}% ({details[name]['fit_predict_seconds']:.1f}s)", flush=True)
        all_rows.append(rows)
    if baseline:
        start = time.monotonic()
        print("  Computing official k=10 nearest-sales baseline at the SAME fixed cutoff...", flush=True)
        predicted = NearestSalesBaseline(k=10).fit(train).predict(hidden)
        rows = result_rows(homes, np.round(predicted), "nearest_sales", protocol, cutoff)
        rows.to_csv(folder / "nearest_sales_predictions.csv", index=False)
        details["nearest_sales"] = {**metrics(rows), "fit_predict_seconds": time.monotonic() - start}
        print(f"  nearest_sales: MdAPE={details['nearest_sales']['mdape']:.3f}%, MAPE={details['nearest_sales']['mape']:.3f}%", flush=True)
        all_rows.append(rows)
    write_json(folder / "audit.json", {
        "protocol": protocol, "cutoff_exclusive": cutoff, "evaluation_end_exclusive": end,
        "train_n": len(train), "validation_n": len(homes),
        "max_training_sale_date": str(train.date_sold.max()),
        "validation_labels_used_for_fit_or_early_stopping": False,
        "history_query": "[min(date_listed, cutoff)-lookback, min(date_listed, cutoff))",
        "history_reference_n": history.reference_n, "feature_count": len(x_train.columns),
        "features": list(x_train.columns), "scores": details,
    })
    return pd.concat(all_rows, ignore_index=True)


def grouped_scores(rows, column):
    return [{"group": str(key), **metrics(g)} for key, g in rows.groupby(column, observed=True)]


def summarize(rows):
    result = {}
    for (protocol, name), group in rows.groupby(["protocol", "model"]):
        result.setdefault(protocol, {})[name] = {"overall": metrics(group),
            "by_month": grouped_scores(group, "month"),
            "by_property_type": grouped_scores(group, "house_type_name"),
            "by_community": grouped_scores(group, "community")}
        priced = group.assign(price_band=pd.cut(group.sold_price, [0, 5e5, 1e6, 1.5e6, 2e6, np.inf],
                             labels=["<500k", "500k-1m", "1m-1.5m", "1.5m-2m", ">=2m"], right=False))
        result[protocol][name]["by_price_band"] = grouped_scores(priced, "price_band")
    return result


def write_report(out, scores, selected, submission_checks):
    lines = ["# 方案 A：CatBoost 与 LightGBM 实测结果", "",
             "本次仅研究 Off-market。两种模型采用相同特征与 Log-MAE，所有百分比指标越低越好。", "",
             "## ① 主结果：固定截至 1 月训练，一次预测 2026 年 2—5 月", "",
             "四个月内模型、历史成交池均不更新；2—5 月真实成交价只在预测完成后用于评分与模型选择。", "",
             "| 模型 | 房屋数 | MdAPE | MAPE | 中位有符号误差 | 误差超过 30% 的比例 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, item in scores["selection_A"].items():
        s = item["overall"]
        lines.append(f"| {name} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% | {s['median_signed_error']:+.2f}% | {s['over_30_pct']:.2f}% |")
    lines += ["", f"**按预定规则选定：{selected}。** 先比较四个月合并后的 MdAPE，完全相同时以 MAPE 打破平局。", "",
              "2—5 月属于模型选择期，并非独立测试成绩。只比较两套预先固定的设置，没有对这四个月早停或网格调参。", "",
              "## ② 主回测逐月表现", "", "| 月份 | 模型 | 房屋数 | MdAPE | MAPE |", "|---|---|---:|---:|---:|"]
    for name, item in scores["selection_A"].items():
        for s in item["by_month"]:
            lines.append(f"| {s['group']} | {name} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% |")
    lines += ["", "## ③ 较早历史开发窗口", "", "2025 年 1—4 月、5—8 月、9—12 月；每个窗口从其月初截止日固定训练。三窗合并统计，每个样本只计一次。", "",
              "| 模型 | 房屋数 | 合并 MdAPE | 合并 MAPE |", "|---|---:|---:|---:|"]
    for name, item in scores["development_A"].items():
        s = item["overall"]
        lines.append(f"| {name} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% |")
    lines += ["", "各窗口、各月、房型、社区、价位误差均保存在 metrics.json 和各窗口 audit.json。", "",
              "## ④ 官方口径对照 B：2025 年 12 月—2026 年 5 月逐月重训", "",
              "此处可以加入上月真实成交记录，因此与方案 A 分开比较。最近邻结果来自首次基线存档，已核验房屋 ID 和真实价格一致。", "",
              "| 模型 | 房屋数 | MdAPE | MAPE |", "|---|---:|---:|---:|"]
    for name, item in scores["official_B"].items():
        s = item["overall"]
        lines.append(f"| {name} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% |")
    lines += ["", "## ⑤ 房型误差：方案 A", "", "| 房型 | 模型 | 房屋数 | MdAPE | MAPE |", "|---|---|---:|---:|---:|"]
    for name in MODELS:
        for s in scores["selection_A"][name]["by_property_type"]:
            lines.append(f"| {s['group']} | {name} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% |")
    lines += ["", "## ⑥ 冻结方案后重训与预测", "",
              f"选择记录见 selection.json。两种模型均用截至 2026 年 5 月的 150,000 条记录重新拟合；推荐提交文件采用 {selected}。", "",
              "| 文件 | 行数 | 格式检查 |", "|---|---:|---|"]
    for name, item in submission_checks.items():
        lines.append(f"| {item['file']} | {item['n']:,} | {'通过' if not item['errors'] else item['errors']} |")
    lines += ["", "隐藏测试房屋没有公开真实成交价，因此不能报告 6—9 月误差。", "",
              "## ⑦ 解释边界", "",
              "- Log-MAE 是训练目标，不等于比赛的 MdAPE。",
              "- 历史价格特征使用平滑的对数均价、对数单价（取指数对应几何均值），并非历史中位数；90/180/365 天近似 3/6/12 个月。",
              "- 社区×房型与同坐标×房型统计回退到同房型；同房型回退到全市场。初始无历史的记录保留缺失，不用全样本目标补齐。",
              "- 文本只提取有限房况关键词并处理部分否定，未使用完整原文、价格数字或外部语言模型；规则仍可能误判。",
              "- 对于截止日前已挂牌的房屋，历史特征只查挂牌前成交，但整个模型仍遵循成交月份截止日训练；这不是严格挂牌当天模型回测。",
              "- 两个算法的树结构不同，本次固定参数只是首版比较，不能据此断言某个算法普遍更优。",
              "- 特征重要性是模型内部指标，两个库数值尺度不同，也不代表因果作用。",
              "- 同期房屋共享市场变化，未把逐套房 bootstrap 区间当作未来市场风险区间。", ""]
    (out / "结果报告.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "run_2026-10-16_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never silently overwrite results.
    listings_path = args.data_dir / "listings.parquet"
    test_path = args.data_dir / "test_off_market.parquet"
    listings, test = pd.read_parquet(listings_path), pd.read_parquet(test_path)
    for c in ["date_sold", "date_listed"]:
        listings[c] = pd.to_datetime(listings[c])
    test["date_listed"] = pd.to_datetime(test.date_listed)
    if listings.ml_num.duplicated().any() or test.ml_num.duplicated().any() or set(listings.ml_num).intersection(test.ml_num):
        raise ValueError("IDs must be unique and disjoint")
    if not (listings.date_sold < pd.Timestamp("2026-06-01")).all():
        raise ValueError("Training labels after May are forbidden")
    source_copy = HERE / "原始训练方案.md"
    if SOURCE_PLAN.exists() and not source_copy.exists():
        shutil.copy2(SOURCE_PLAN, source_copy)
    started = datetime.now(ZoneInfo("America/Toronto")).isoformat()
    manifest = {"started_at_Toronto": started, "python": sys.version, "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ["catboost", "lightgbm", "numpy", "pandas", "scikit-learn", "pyarrow"]},
                "inputs_sha256": {p.name: sha(p) for p in [listings_path, test_path, source_copy]},
                "code_sha256": {p.name: sha(p) for p in HERE.glob("*.py")},
                "params": PARAMS, "candidate_count": 2, "training_loss": "MAE(log(sold_price))",
                "development_windows": DEV, "selection_window": ["2026-02-01", "2026-06-01"],
                "early_stopping": False, "parameter_search": False}
    write_json(out / "manifest.json", manifest)
    results = []
    for cutoff, end in DEV:
        results.append(evaluate_window(listings, out, cutoff, end, "development_A"))
    selection = evaluate_window(listings, out, "2026-02-01", "2026-06-01", "selection_A", baseline=True)
    results.append(selection)
    candidate_scores = {name: metrics(selection.loc[selection.model == name]) for name in MODELS}
    selected = min(MODELS, key=lambda name: (candidate_scores[name]["mdape"], candidate_scores[name]["mape"]))
    write_json(out / "selection.json", {"selected_model": selected, "rule": "pooled February-May MdAPE; MAPE breaks exact ties",
        "selected_at_Toronto": datetime.now(ZoneInfo("America/Toronto")).isoformat(), "scores": candidate_scores,
        "params_frozen": PARAMS, "selection_period_is_not_independent_test": True})
    print(f"[A4] Frozen choice: {selected}", flush=True)
    for month in pd.period_range("2025-12", "2026-05", freq="M"):
        results.append(evaluate_window(listings, out, str(month.start_time.date()), str((month + 1).start_time.date()), "official_B"))
    # Reuse an already executed official baseline, never substitute it for A.
    baseline_path = REPO.parent / "资料说明" / "基线首次运行" / "outputs" / "validation_predictions.csv"
    baseline = pd.read_csv(baseline_path)
    baseline = baseline.loc[baseline.track == "off_market"].drop(columns="track")
    official = pd.concat(results).query("protocol == 'official_B' and model == 'catboost'")
    assert set(baseline.ml_num) == set(official.ml_num)
    joined = baseline.merge(official[["ml_num", "sold_price"]], on="ml_num", validate="one_to_one", suffixes=("", "_check"))
    assert (joined.sold_price == joined.sold_price_check).all()
    baseline = baseline.merge(listings[["ml_num", "community"]], on="ml_num", validate="one_to_one")
    baseline["model"] = "nearest_sales"
    baseline["protocol"] = "official_B"
    baseline["cutoff_exclusive"] = pd.to_datetime(baseline.month).dt.strftime("%Y-%m-01")
    baseline.to_csv(out / "official_B" / "archived_nearest_sales_predictions.csv", index=False)
    results.append(baseline)
    manifest["official_baseline_sha256"] = sha(baseline_path)
    combined = pd.concat(results, ignore_index=True)
    scores = summarize(combined)
    write_json(out / "metrics.json", scores)
    # Save group scores as a flat, spreadsheet-friendly CSV too.
    groups = []
    for protocol, models in scores.items():
        for name, entry in models.items():
            for dimension, values in entry.items():
                if dimension == "overall":
                    values = [{"group": "all", **values}]
                for value in values:
                    groups.append({"protocol": protocol, "model": name, "dimension": dimension, **value})
    pd.DataFrame(groups).to_csv(out / "分组误差.csv", index=False)
    print("[A5] Final refit on all known sales through May; hidden labels remain unavailable.", flush=True)
    train = listings.sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    history = HistoryFeatures(train, "2026-06-01")
    x_train, x_test = history.transform(train), history.transform(test)
    submissions = out / "final_predictions"
    submissions.mkdir()
    checks = {}
    model_hashes = {}
    for name in MODELS:
        model = PriceModel(name).fit(x_train, train.sold_price)
        model_path = model.save(out / "trained_models")
        model_hashes[model_path.name] = sha(model_path)
        path = submissions / f"submission_off_market_{name}.csv"
        pd.DataFrame({"ml_num": test.ml_num, "predicted_price": model.predict(x_test)}).to_csv(path, index=False)
        errors = check_submission.check_file(path, test.ml_num.tolist())
        if errors:
            raise ValueError(errors)
        checks[name] = {"file": str(path.relative_to(out)), "n": len(test), "errors": errors, "sha256": sha(path)}
        print(f"  {name}: {len(test):,} hidden-test predictions; submission check PASSED", flush=True)
    recommended = submissions / "submission_off_market.csv"
    shutil.copy2(submissions / f"submission_off_market_{selected}.csv", recommended)
    checks["recommended"] = {"file": str(recommended.relative_to(out)), "n": len(test), "errors": [], "sha256": sha(recommended)}
    write_json(out / "submission_checks.json", checks)
    write_report(out, scores, selected, checks)
    manifest["trained_models_sha256"] = model_hashes
    manifest["finished_at_Toronto"] = datetime.now(ZoneInfo("America/Toronto")).isoformat()
    manifest["status"] = "complete"
    write_json(out / "manifest.json", manifest)
    print(f"DONE: {out / '结果报告.md'}", flush=True)


if __name__ == "__main__":
    main()
