"""Compare half-life-12 training with archived, matched unweighted plan A.

Run from repository root with .venv/bin/python -u -m experiments.time_decay.run_experiment.
Only new output directories are allowed. No hidden-test predictions are changed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import Booster

import data
from experiments.plan_a.features import HistoryFeatures
from experiments.plan_a.models import PARAMS
from experiments.plan_a.run_experiment import DEV, metrics, result_rows, summarize
from .models import MONTH_DAYS, WeightedPriceModel, time_weights

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_NAMES = ["catboost", "lightgbm"]
HALF_LIFE = 12.0  # Fixed before scoring; no half-life search on February-May.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def protected_files(data_dir, previous):
    files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
    files += [REPO / n for n in ["models.py", "features.py", "run.py", "data.py"]]
    files += list((REPO / "experiments" / "plan_a").glob("*.py"))
    files += list(previous.rglob("*"))
    return {str(p): sha(p) for p in files if p.is_file()}


def checked_archive(path, homes, name, protocol, cutoff):
    rows = pd.read_csv(path)
    if rows.ml_num.duplicated().any() or set(rows.ml_num) != set(homes.ml_num):
        raise ValueError(f"Archived IDs do not match {path}")
    aligned = rows.set_index("ml_num").loc[homes.ml_num].reset_index()
    if not np.array_equal(aligned.sold_price.to_numpy(), homes.sold_price.to_numpy()):
        raise ValueError(f"Archived labels do not match {path}")
    expected_month = homes.date_sold.dt.strftime("%Y-%m").to_numpy()
    if not np.array_equal(aligned.month.to_numpy(), expected_month):
        raise ValueError(f"Archived dates do not match {path}")
    for column, value in [("model", name), ("protocol", protocol), ("cutoff_exclusive", cutoff)]:
        if not aligned[column].eq(value).all():
            raise ValueError(f"Archived {column} does not match {path}")
    aligned["scheme"] = "equal"
    return aligned


def weight_summary(train, cutoff, weights):
    ages = np.asarray((pd.Timestamp(cutoff) - pd.DatetimeIndex(train.date_sold)).total_seconds()) / (86400 * MONTH_DAYS)
    result = {"n": len(train), "mean": float(weights.mean()), "min": float(weights.min()),
              "max": float(weights.max()), "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
              "half_life_months": HALF_LIFE}
    for lo, hi, name in [(0, 6, "age_0_6_months"), (6, 12, "age_6_12_months"),
                         (12, 24, "age_12_24_months"), (24, np.inf, "age_over_24_months")]:
        selected = (ages >= lo) & (ages < hi)
        result[name] = {"rows": int(selected.sum()), "row_share_pct": float(100 * selected.mean()),
                        "weight_share_pct": float(100 * weights[selected].sum() / weights.sum())}
    return result


def save_and_verify(model, x, predicted, folder, cutoff):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (model.name + (".cbm" if model.name == "catboost" else ".txt"))
    if model.name == "catboost":
        model.model.save_model(str(path))
        loaded = CatBoostRegressor().load_model(str(path))
    else:
        model.model.booster_.save_model(str(path))
        loaded = Booster(model_file=str(path))
    replay = np.round(np.exp(loaded.predict(model.encode(x))))
    if not np.array_equal(predicted, replay):
        raise ValueError(f"Saved model replay differs: {model.name}")
    write_json(folder / f"{model.name}_schema.json", {
        "features": model.feature_names, "categories": model.categories,
        "params": PARAMS[model.name], "target": "log(sold_price)",
        "training_cutoff_exclusive": cutoff, "half_life_months": HALF_LIFE,
        "model_sha256": sha(path), "reload_exact_match": True,
    })


def evaluate(listings, previous, out, cutoff, end, protocol, checks):
    train = listings.loc[listings.date_sold < cutoff].sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    homes = listings.loc[(listings.date_sold >= cutoff) & (listings.date_sold < end)].reset_index(drop=True)
    assert len(train) and len(homes) and not set(train.ml_num).intersection(homes.ml_num)
    folder = out / protocol / cutoff[:7]
    folder.mkdir(parents=True)
    print(f"[{protocol}] train before {cutoff}: {len(train):,}; predict through {end}: {len(homes):,}", flush=True)
    history = HistoryFeatures(train, cutoff)
    x_train = history.transform(train)
    x_valid = history.transform(data.hide_answers(homes, "off_market"))
    audit = json.loads((previous / protocol / cutoff[:7] / "audit.json").read_text())
    if list(x_train.columns) != audit["features"] or len(train) != audit["train_n"]:
        raise ValueError("Features or training rows differ from the archived experiment")
    weights = time_weights(train.date_sold, cutoff, HALF_LIFE)
    summary = weight_summary(train, cutoff, weights)
    if protocol == "selection_A":
        pd.DataFrame({"ml_num": train.ml_num, "date_sold": train.date_sold, "sample_weight": weights}).to_csv(folder / "training_weights.csv", index=False)
    all_rows, details = [], {}
    for name in MODEL_NAMES:
        archived = previous / protocol / cutoff[:7] / f"{name}_predictions.csv"
        control_rows = checked_archive(archived, homes, name, protocol, cutoff)
        checks["archived_predictions_sha256"][str(archived)] = sha(archived)
        if protocol == "selection_A":
            # Refit both full-size controls to establish that the archived
            # predictions still reproduce under the current runtime.
            start = time.monotonic()
            control = WeightedPriceModel(name).fit(x_train, train.sold_price)
            replay = control.predict(x_valid)
            if not np.array_equal(replay, control_rows.predicted_price.to_numpy()):
                raise ValueError(f"Unweighted {name} does not reproduce the archive")
            checks["unweighted_main_reproduction"][name] = {
                "n": len(homes), "exact_match": True, "seconds": time.monotonic() - start}
            print(f"  {name} equal-weight control: exact archived reproduction", flush=True)
        control_rows.to_csv(folder / f"{name}_equal_predictions.csv", index=False)
        start = time.monotonic()
        weighted = WeightedPriceModel(name).fit(x_train, train.sold_price, weights)
        prediction = weighted.predict(x_valid)
        rows = result_rows(homes, prediction, name, protocol, cutoff)
        rows["scheme"] = "decay_12m"
        rows.to_csv(folder / f"{name}_decay_12m_predictions.csv", index=False)
        importance = (weighted.model.get_feature_importance() if name == "catboost"
                      else weighted.model.booster_.feature_importance(importance_type="gain"))
        pd.DataFrame({"feature": x_train.columns, "importance": importance}).sort_values("importance", ascending=False).to_csv(folder / f"{name}_decay_12m_feature_importance.csv", index=False)
        details[name] = {"equal": metrics(control_rows), "decay_12m": metrics(rows),
                         "weighted_fit_predict_seconds": time.monotonic() - start}
        if protocol == "selection_A":
            save_and_verify(weighted, x_valid, prediction, out / "trained_selection_models", cutoff)
        print(f"  {name}: MdAPE {details[name]['equal']['mdape']:.3f}% -> {details[name]['decay_12m']['mdape']:.3f}%; MAPE {details[name]['equal']['mape']:.3f}% -> {details[name]['decay_12m']['mape']:.3f}%", flush=True)
        all_rows.extend([control_rows, rows])
    write_json(folder / "audit.json", {
        "cutoff_exclusive": cutoff, "end_exclusive": end, "protocol": protocol,
        "training_n": len(train), "validation_n": len(homes),
        "max_training_date": str(train.date_sold.max()), "features": list(x_train.columns),
        "feature_count": len(x_train.columns), "weight_summary": summary, "scores": details,
        "validation_labels_used_for_training_or_early_stopping": False,
        "model_and_history_frozen_for_four_months": True,
        "history_aggregation_weights_changed": False,
    })
    return pd.concat(all_rows, ignore_index=True)


def score_table(rows, grouping):
    return [{**dict(zip(grouping, key if isinstance(key, tuple) else (key,))), **metrics(group)}
            for key, group in rows.groupby(grouping, observed=True)]


def write_report(out, rows, manifest, reference, summary):
    main = rows.query("protocol == 'selection_A'")
    def score(name, scheme, frame=main):
        return metrics(frame.loc[(frame.model == name) & (frame.scheme == scheme)])
    lines = ["# Off-market 时间衰减权重实验报告", "", "日期：2026 年 10 月 16 日（多伦多时间）。", "",
             "## 1. 本次实验回答什么", "",
             "与已有方案 A 比较：仅把训练样本的等权重改为半衰期 12 个月的时间衰减权重，近期成交是否能改善未来四个月的预测？",
             "沿用 CatBoost、LightGBM 的 64 项特征、Log-MAE、600 轮、随机种子 42 和其余固定参数。历史统计的计算方式保持原样；只改变训练样本权重。", "",
             "## 2. 主结果：截至 2026 年 1 月训练，固定预测 2—5 月", "",
             "训练 142,227 套，评价 7,773 套。四个月内模型和历史成交池均固定，验证标签不用于训练或早停。评分仍为所有验证房屋的等权 MdAPE／MAPE，不对评分加时间权重。", "",
             "| 模型 | 训练权重 | MdAPE ↓ | MAPE ↓ | 中位有符号误差 | 误差超过 30% 的比例 |",
             "|---|---|---:|---:|---:|---:|"]
    for name in MODEL_NAMES:
        for scheme, label in [("equal", "等权重（已有方案）"), ("decay_12m", "时间衰减，半衰期 12 月")]:
            s = score(name, scheme)
            lines.append(f"| {name} | {label} | {s['mdape']:.2f}% | {s['mape']:.2f}% | {s['median_signed_error']:+.2f}% | {s['over_30_pct']:.2f}% |")
    for name in MODEL_NAMES:
        a, b = score(name, "equal"), score(name, "decay_12m")
        delta = b['mdape'] - a['mdape']
        lines += ["", f"- {name}：MdAPE {'下降' if delta < 0 else '上升'} {abs(delta):.3f} 个百分点；MAPE 变化 {b['mape'] - a['mape']:+.3f} 个百分点。"]
    lines += ["", f"同截止日原始 k=10 最近邻基线：MdAPE {reference['mdape']:.2f}%，MAPE {reference['mape']:.2f}%。它只作方法参考，不用于单独归因时间权重的效果。", "",
              "## 3. 逐月表现（每个月都来自同一个固定模型）", "",
              "| 模型 | 月份 | 房屋数 | 等权 MdAPE | 加权 MdAPE | 变化／百分点 | 等权 MAPE | 加权 MAPE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in MODEL_NAMES:
        for month in sorted(main.month.unique()):
            frame = main.loc[main.month == month]
            a, b = score(name, "equal", frame), score(name, "decay_12m", frame)
            lines.append(f"| {name} | {month} | {a['n']:,} | {a['mdape']:.2f}% | {b['mdape']:.2f}% | {b['mdape']-a['mdape']:+.3f} | {a['mape']:.2f}% | {b['mape']:.2f}% |")
    lines += ["", "## 4. 较早四个月窗口的稳定性", "",
              "半衰期事先固定为 12 个月；没有依据结果继续搜索半衰期或改变其他参数。等权开发期预测复用上一轮存档，并核对数据、代码、参数、运行库版本、逐套 ID、成交价和月份。主选择期两个等权模型均已本次重新训练，逐套价格与存档完全一致。", "",
              "| 开发窗口 | 模型 | 等权 MdAPE | 加权 MdAPE | 变化／百分点 | 等权 MAPE | 加权 MAPE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    development = rows.query("protocol == 'development_A'")
    for cutoff in sorted(development.cutoff_exclusive.unique()):
        frame = development.loc[development.cutoff_exclusive == cutoff]
        period = f"{frame.month.min()}—{frame.month.max()}"
        for name in MODEL_NAMES:
            a, b = score(name, "equal", frame), score(name, "decay_12m", frame)
            lines.append(f"| {period} | {name} | {a['mdape']:.2f}% | {b['mdape']:.2f}% | {b['mdape']-a['mdape']:+.3f} | {a['mape']:.2f}% | {b['mape']:.2f}% |")
    lines += ["", "## 5. 房型误差：主选择期", "",
              "| 模型 | 房型 | 房屋数 | 等权 MdAPE | 加权 MdAPE | 变化／百分点 |",
              "|---|---|---:|---:|---:|---:|"]
    for name in MODEL_NAMES:
        for kind in sorted(main.house_type_name.unique()):
            frame = main.loc[main.house_type_name == kind]
            a, b = score(name, "equal", frame), score(name, "decay_12m", frame)
            lines.append(f"| {name} | {kind} | {a['n']:,} | {a['mdape']:.2f}% | {b['mdape']:.2f}% | {b['mdape']-a['mdape']:+.3f} |")
    audit = json.loads((out / "selection_A" / "2026-02" / "audit.json").read_text())
    ws = audit["weight_summary"]
    lines += ["", "## 6. 权重怎样进入训练", "", "$$", "w_i=2^{-\\Delta t_i/12},\\quad \\tilde w_i=w_i/\\overline w", "$$", "", "$$",
              "L=\\frac{\\sum_i\\tilde w_i|f(X_i)-\\log(P_i)|}{\\sum_i\\tilde w_i}", "$$", "",
              "年龄按成交日期到训练截止日的天数除以 365.25/12 计算。每个窗口使用自己的固定训练截止日；不读取之后四个月的答案。归一化后平均权重为 1，保持总体权重规模与等权对照一致，原正则化参数不变。",
              "原始衰减权重下，近期、一年前、两年前的相对权重约为 1、0.5、0.25。归一化只改变共同倍数，不改变相对比例。", "",
              "| 训练记录年龄 | 房屋数 | 原样本占比 | 加权后占比 |", "|---|---:|---:|---:|"]
    for key, label in [("age_0_6_months", "0—6 月"), ("age_6_12_months", "6—12 月"), ("age_12_24_months", "12—24 月"), ("age_over_24_months", "24 月及以上")]:
        item = ws[key]
        lines.append(f"| {label} | {item['rows']:,} | {item['row_share_pct']:.2f}% | {item['weight_share_pct']:.2f}% |")
    lines += ["", f"主期归一化权重范围 {ws['min']:.3f}—{ws['max']:.3f}，平均值 {ws['mean']:.3f}；权重有效样本量约 {ws['effective_sample_size']:,.0f}。该有效样本量只描述权重集中程度，不是独立市场样本量或置信区间。", "",
              "## 7. 结论与解释边界", ""]
    for name in MODEL_NAMES:
        main_delta = score(name, "decay_12m")['mdape'] - score(name, "equal")['mdape']
        wins = sum(score(name, "decay_12m", development.loc[development.cutoff_exclusive == c])['mdape'] < score(name, "equal", development.loc[development.cutoff_exclusive == c])['mdape'] for c in development.cutoff_exclusive.unique())
        lines.append(f"- {name}：主期 MdAPE {'改善' if main_delta < 0 else '恶化'}；三个较早窗口中 {wins}/3 个 MdAPE 改善。")
    candidates = [(score(name, scheme)['mdape'], score(name, scheme)['mape'], name, scheme) for name in MODEL_NAMES for scheme in ["equal", "decay_12m"]]
    best = min(candidates)
    lines += [f"- 主选择期四个候选中最低 MdAPE 为 {best[0]:.2f}%，来自 {best[2]}／{'等权重' if best[3]=='equal' else '半衰期12月'}。这是本轮条件下的排名，不代表已证明隐藏测试表现或全面最优。",
              "- 时间权重并非越偏近期越好；需要结合各月、房型和较早窗口判断，不能仅凭一个总分认定更好。",
              "- 2—5 月此前已用于算法选择，本次再次用于权重比较，属于累计开发过程，不能称为独立测试。",
              "- 本次四个月回测内模型与历史池均不更新，没有另外改变训练截止日与历史库截止日，也没有测试“冻结旧模型、提前刷新历史库”的另一方案。",
              "- 历史特征遵循上一轮规则：只查询 min(挂牌日, 训练截止日) 之前的成交；整个模型仍按成交日期截止训练，不是严格挂牌当天模型回测。",
              "- 本次没有新增官方 B 逐月加权回测，也没有生成或替换 6—9 月提交文件；隐藏测试答案仍不可见。",
              "- 原始数据、官方代码、上一轮实验代码及产物的文件哈希在运行前后保持一致。", "",
              "## 8. 文件与复现", "",
              "- `metrics.json`：整体、逐月、房型、社区和价格区间指标。",
              "- `窗口对照.csv`、`分组对照.csv`：可在表格软件中查看的数值。",
              "- `selection_A/2026-02/training_weights.csv`：主期逐套训练权重。",
              "- 各窗口 `*_predictions.csv`、`audit.json`：等权与加权预测、边界和权重摘要。",
              "- `trained_selection_models/`：截至 1 月训练的加权模型和 schema，重载后复现主期预测。",
              "- `manifest.json`、`checks.json`、`source_snapshot/`：配置、数据与代码哈希、重跑核验和运行源码。", "",
              "从 SSU-Data-Dash 仓库目录运行，输出目录必须是新的目录：", "", "```bash",
              ".venv/bin/python -u -m experiments.time_decay.run_experiment --output experiments/time_decay/artifacts/my_new_run", "```", "",
              "训练库官方接口：[CatBoost sample_weight](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit)、[LightGBM sample_weight](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)。", ""]
    (out / "实验报告.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--previous", type=Path, default=REPO / "experiments/plan_a/artifacts/run_2026-10-16_v1")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/run_2026-10-16_h12_v1")
    args = parser.parse_args()
    out, previous, data_dir = args.output.resolve(), args.previous.resolve(), args.data_dir.resolve()
    old = json.loads((previous / "manifest.json").read_text())
    if old["params"] != PARAMS or sha(data_dir / "listings.parquet") != old["inputs_sha256"]["listings.parquet"]:
        raise ValueError("Source dataset or parameters changed")
    for name in ["features.py", "models.py"]:
        if sha(REPO / "experiments/plan_a" / name) != old["code_sha256"][name]:
            raise ValueError("Plan-A features/models changed")
    for package in ["catboost", "lightgbm", "numpy", "pandas"]:
        if importlib.metadata.version(package) != old["packages"][package]:
            raise ValueError(f"Library version changed: {package}")
    out.mkdir(parents=True, exist_ok=False)
    before = protected_files(data_dir, previous)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in list(HERE.glob("*.py")) + [REPO / "experiments/plan_a" / n for n in ["features.py", "models.py", "run_experiment.py"]]:
        prefix = "plan_a_" if path.parent.name == "plan_a" else "time_decay_"
        (snapshot / (prefix + path.name)).write_bytes(path.read_bytes())
    manifest = {"status": "running", "started_at_Toronto": datetime.now(ZoneInfo("America/Toronto")).isoformat(),
                "python": platform.python_version(), "packages": old["packages"], "params": PARAMS,
                "half_life_months": HALF_LIFE, "weight_normalization": "divide by mean within training fold",
                "weight_age_unit_days": MONTH_DAYS, "training_loss": "weighted MAE(log(sold_price))",
                "evaluation_metrics_weighted": False, "early_stopping": False, "half_life_search": False,
                "source_dataset_sha256": sha(data_dir / "listings.parquet"), "previous_run": str(previous),
                "code_sha256": {p.name: sha(p) for p in snapshot.iterdir()}}
    write_json(out / "manifest.json", manifest)
    listings = pd.read_parquet(data_dir / "listings.parquet")
    for c in ["date_sold", "date_listed"]:
        listings[c] = pd.to_datetime(listings[c])
    if listings.ml_num.isna().any() or listings.ml_num.duplicated().any() or listings.date_sold.isna().any():
        raise ValueError("Invalid IDs or dates")
    if not (listings.sold_price > 0).all() or not (listings.date_sold < "2026-06-01").all():
        raise ValueError("Invalid historical labels")
    checks = {"unweighted_main_reproduction": {}, "archived_predictions_sha256": {}}
    all_rows = [evaluate(listings, previous, out, "2026-02-01", "2026-06-01", "selection_A", checks)]
    for cutoff, end in DEV:
        all_rows.append(evaluate(listings, previous, out, cutoff, end, "development_A", checks))
    combined = pd.concat(all_rows, ignore_index=True)
    if combined.duplicated(["protocol", "cutoff_exclusive", "model", "scheme", "ml_num"]).any():
        raise ValueError("Duplicate evaluation rows")
    combined.to_csv(out / "all_predictions.csv", index=False)
    scores = {}
    for scheme, frame in combined.groupby("scheme"):
        scores[scheme] = summarize(frame)
    write_json(out / "metrics.json", scores)
    window_scores = score_table(combined, ["protocol", "cutoff_exclusive", "model", "scheme"])
    pd.DataFrame(window_scores).to_csv(out / "窗口对照.csv", index=False)
    group_scores = []
    for scheme, protocols in scores.items():
        for protocol, models in protocols.items():
            for name, dimensions in models.items():
                for dimension, values in dimensions.items():
                    values = [{"group": "all", **values}] if dimension == "overall" else values
                    for entry in values:
                        group_scores.append({"scheme": scheme, "protocol": protocol, "model": name, "dimension": dimension, **entry})
    pd.DataFrame(group_scores).to_csv(out / "分组对照.csv", index=False)
    ref_homes = listings.loc[(listings.date_sold >= "2026-02-01") & (listings.date_sold < "2026-06-01")].reset_index(drop=True)
    ref_path = previous / "selection_A/2026-02/nearest_sales_predictions.csv"
    reference = metrics(checked_archive(ref_path, ref_homes, "nearest_sales", "selection_A", "2026-02-01"))
    write_report(out, combined, manifest, reference, scores)
    after = protected_files(data_dir, previous)
    if before != after:
        raise ValueError("Protected input or earlier experiment changed during the run")
    checks.update({"protected_files_unchanged": True, "protected_files_n": len(before),
                   "protected_files_sha256": before, "four_month_window_count": 4,
                   "weighted_full_fits": 8, "unweighted_reproduction_full_fits": 2,
                   "saved_weighted_models_replay_exact": True,
                   "selection_rows_per_model_and_scheme": len(ref_homes), "format_and_date_checks_passed": True})
    write_json(out / "checks.json", checks)
    manifest.update({"status": "complete", "finished_at_Toronto": datetime.now(ZoneInfo("America/Toronto")).isoformat()})
    write_json(out / "manifest.json", manifest)
    print(f"DONE: {out / '实验报告.md'}", flush=True)


if __name__ == "__main__":
    main()
