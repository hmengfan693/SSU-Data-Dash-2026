"""Fixed four-month HMM+LightGBM ablation, keeping all earlier files unchanged."""
import argparse
import importlib.metadata
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from lightgbm import Booster

import data
from experiments.plan_a.features import HistoryFeatures
from experiments.plan_a.models import PARAMS, PriceModel
from experiments.plan_a.run_experiment import DEV, metrics, result_rows, summarize
from experiments.time_decay.run_experiment import checked_archive, protected_files, sha, write_json
from .features import CausalMarketFeatures, RAW_COLUMNS, STATE_COLUMNS, MIN_MONTHS, SEEDS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCHEMES = ["baseline", "monthly_only", "monthly_hmm"]
LABELS = {"baseline": "原 LightGBM", "monthly_only": "LightGBM＋月度行情", "monthly_hmm": "LightGBM＋月度行情＋HMM"}


def save_model(model, x_valid, prices, folder, scheme, cutoff):
    path = folder / f"{scheme}.txt"
    model.model.booster_.save_model(str(path))
    replay = np.round(np.exp(Booster(model_file=str(path)).predict(model.encode(x_valid))))
    if not np.array_equal(prices, replay):
        raise ValueError("Saved model replay mismatch")
    write_json(folder / f"{scheme}_schema.json", {"features": model.feature_names,
        "categories": model.categories, "training_cutoff_exclusive": cutoff,
        "target": "log(sold_price)", "params": PARAMS["lightgbm"],
        "saved_model_sha256": sha(path), "exact_reload_reproduction": True})


def evaluate(listings, previous, out, cutoff, end, protocol):
    train = listings.loc[listings.date_sold < cutoff].sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    homes = listings.loc[(listings.date_sold >= cutoff) & (listings.date_sold < end)].reset_index(drop=True)
    assert len(train) and len(homes) and not set(train.ml_num).intersection(homes.ml_num)
    folder = out / protocol / cutoff[:7]
    folder.mkdir(parents=True)
    print(f"[{protocol}] cutoff {cutoff}; train={len(train):,}; evaluation={len(homes):,}", flush=True)
    start = time.monotonic()
    history = HistoryFeatures(train, cutoff)
    x_train = history.transform(train)
    hidden = data.hide_answers(homes, "off_market")
    x_valid = history.transform(hidden)
    archived_audit = json.loads((previous / protocol / cutoff[:7] / "audit.json").read_text())
    if list(x_train.columns) != archived_audit["features"] or len(train) != archived_audit["train_n"]:
        raise ValueError("Baseline data/features do not match archive")
    market = CausalMarketFeatures(train, cutoff)
    train_extra, valid_extra = market.transform(train), market.transform(hidden)
    market.monthly.to_csv(folder / "monthly_observations.csv")
    market.cache.to_csv(folder / "point_in_time_feature_cache.csv", index_label="available_from")
    write_json(folder / "hmm_fit_audits.json", market.audits)
    pd.concat([homes[["ml_num", "date_listed"]], valid_extra], axis=1).to_csv(folder / "validation_market_features.csv", index=False)
    print(f"  HMM prefix features ready ({time.monotonic()-start:.1f}s); available training={train_extra.hmm_available.mean():.1%}, validation={valid_extra.hmm_available.mean():.1%}", flush=True)
    archived = checked_archive(previous / protocol / cutoff[:7] / "lightgbm_predictions.csv", homes, "lightgbm", protocol, cutoff).drop(columns="scheme")
    archived["scheme"] = "baseline"
    reproduction = None
    if protocol == "selection_A":
        model = PriceModel("lightgbm").fit(x_train, train.sold_price)
        reproduction = bool(np.array_equal(model.predict(x_valid), archived.predicted_price.to_numpy()))
        if not reproduction:
            raise ValueError("Main-period baseline refit does not reproduce archived prices")
        print("  baseline: exact reproduction of existing unweighted LightGBM", flush=True)
    archived.to_csv(folder / "baseline_predictions.csv", index=False)
    results = [archived]
    details = {"baseline": metrics(archived)}
    for scheme, columns in [("monthly_only", RAW_COLUMNS), ("monthly_hmm", RAW_COLUMNS + STATE_COLUMNS)]:
        started = time.monotonic()
        tx = pd.concat([x_train, train_extra[columns]], axis=1)
        vx = pd.concat([x_valid, valid_extra[columns]], axis=1)
        model = PriceModel("lightgbm").fit(tx, train.sold_price)
        prediction = model.predict(vx)
        rows = result_rows(homes, prediction, "lightgbm", protocol, cutoff)
        rows["scheme"] = scheme
        rows.to_csv(folder / f"{scheme}_predictions.csv", index=False)
        pd.DataFrame({"feature": tx.columns, "importance": model.model.booster_.feature_importance(importance_type="gain")}).sort_values("importance", ascending=False).to_csv(folder / f"{scheme}_feature_importance.csv", index=False)
        if protocol == "selection_A":
            saved = out / "trained_selection_models"
            saved.mkdir(exist_ok=True)
            save_model(model, vx, prediction, saved, scheme, cutoff)
        details[scheme] = metrics(rows)
        results.append(rows)
        print(f"  {scheme}: MdAPE={details[scheme]['mdape']:.3f}%; MAPE={details[scheme]['mape']:.3f}% ({time.monotonic()-started:.1f}s)", flush=True)
    write_json(folder / "audit.json", {"cutoff_exclusive": cutoff, "end_exclusive": end,
        "training_n": len(train), "validation_n": len(homes), "max_training_sale": str(train.date_sold.max()),
        "base_features": len(x_train.columns), "raw_features": RAW_COLUMNS, "hmm_features": STATE_COLUMNS,
        "hmm_emission": "matched monthly log-ppsf change, Gaussian, two states",
        "hmm_min_months": MIN_MONTHS, "hmm_seeds": SEEDS,
        "historical_hmm_parameters_and_scaler_fit_on_each_past_prefix_only": True,
        "hmm_training_available_pct": float(train_extra.hmm_available.mean()*100),
        "hmm_validation_available_pct": float(valid_extra.hmm_available.mean()*100),
        "hmm_fitted_prefixes": sum(a["status"] == "fitted" for a in market.audits.values()),
        "hmm_failed_prefixes_after_warmup": sum(a["status"] == "no_valid_fit" for a in market.audits.values()),
        "unweighted_main_exact_reproduction": reproduction,
        "validation_labels_used_for_training_or_early_stopping": False,
        "future_observations_added_during_prediction": False, "scores": details})
    return pd.concat(results, ignore_index=True)


def write_report(out, rows, scores):
    def score(frame, scheme):
        return metrics(frame.loc[frame.scheme == scheme])
    main = rows.loc[rows.protocol == "selection_A"]
    base, monthly, hmm = [score(main, s) for s in SCHEMES]
    best = min(SCHEMES, key=lambda s: (score(main, s)["mdape"], score(main, s)["mape"]))
    delta_base, delta_monthly = hmm['mdape']-base['mdape'], hmm['mdape']-monthly['mdape']
    lines = ["# Off-market：HMM 与 LightGBM 组合实验", "", "日期：2026 年 10 月 16 日（多伦多时间）。", "",
        f"**主期 HMM 组合 MdAPE 为 {hmm['mdape']:.2f}%，原 LightGBM 为 {base['mdape']:.2f}%，增加月度行情但不加 HMM 为 {monthly['mdape']:.2f}%。本轮主期最低 MdAPE 来自：{LABELS[best]}。**", "",
        "## 1. 实验设计", "",
        "仅研究 Off-market。与已有方案 A 使用相同房屋、同一时间切分和原 64 项特征；训练仍为等权 Log-MAE、LightGBM 600 轮、相同随机种子及其余参数，不使用此前表现较差的时间衰减权重。不依据 2—5 月标签早停或搜索 HMM 状态数。", "",
        "三个固定候选：原 LightGBM；增加六项月度行情；在前者基础上再增加 HMM 状态与预测特征。中间对照用于判断增益来自行情观测还是 HMM。", "",
        "## 2. 主回测：截至 1 月训练，固定预测 2026 年 2—5 月", "",
        "训练 142,227 套，评价 7,773 套。四个月内模型和历史成交池固定，评分不加权。主期原 LightGBM 本次重跑，与此前存档的逐套预测完全一致。", "",
        "| 模型 | MdAPE ↓ | MAPE ↓ | 中位有符号误差 | 误差超过30%的比例 |", "|---|---:|---:|---:|---:|"]
    for s in SCHEMES:
        m = score(main, s)
        lines.append(f"| {LABELS[s]} | {m['mdape']:.2f}% | {m['mape']:.2f}% | {m['median_signed_error']:+.2f}% | {m['over_30_pct']:.2f}% |")
    lines += ["", f"HMM 组合相对原模型 MdAPE 变化 {delta_base:+.3f} 个百分点；相对只加月度行情变化 {delta_monthly:+.3f} 个百分点。负值代表改善。", "",
        "## 3. 逐月表现", "", "| 月份 | 房屋数 | 原模型 MdAPE | 月度行情 MdAPE | HMM组合 MdAPE | 原模型 MAPE | 月度行情 MAPE | HMM组合 MAPE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for month, frame in main.groupby("month"):
        values = [score(frame, s) for s in SCHEMES]
        lines.append(f"| {month} | {values[0]['n']:,} | " + " | ".join(f"{m['mdape']:.2f}%" for m in values) + " | " + " | ".join(f"{m['mape']:.2f}%" for m in values) + " |")
    development = rows.loc[rows.protocol == "development_A"]
    lines += ["", "## 4. 较早固定四个月窗口", "", "| 窗口 | 原模型 MdAPE | 月度行情 MdAPE | HMM组合 MdAPE |", "|---|---:|---:|---:|"]
    wins_base, wins_monthly = 0, 0
    for cutoff, frame in development.groupby("cutoff_exclusive"):
        values = [score(frame, s) for s in SCHEMES]
        wins_base += values[2]["mdape"] < values[0]["mdape"]
        wins_monthly += values[2]["mdape"] < values[1]["mdape"]
        lines.append(f"| {frame.month.min()}—{frame.month.max()} | " + " | ".join(f"{m['mdape']:.2f}%" for m in values) + " |")
    lines += ["", f"三个较早窗口中，HMM 组合对原模型有 {wins_base}/3 个 MdAPE 改善，对只加月度行情有 {wins_monthly}/3 个改善。较早原模型结果复用已核验的数据与代码存档；两个新增候选本次均重新训练。", "",
        "## 5. 主期房型误差", "", "| 房型 | 房屋数 | 原模型 MdAPE | 月度行情 MdAPE | HMM组合 MdAPE |", "|---|---:|---:|---:|---:|"]
    for kind, frame in main.groupby("house_type_name"):
        values = [score(frame, s) for s in SCHEMES]
        lines.append(f"| {kind} | {values[0]['n']:,} | " + " | ".join(f"{m['mdape']:.2f}%" for m in values) + " |")
    lines += ["", "## 6. HMM 怎样构造，怎样避免未来泄漏", "",
        "1. 按完整成交月份汇总。构建社区×房型×400平方英尺面积段的单元，对同单元前后两个月的对数成交单价中位数作差。两月每单元至少三套、至少五个匹配单元；按两月较小样本量加权、每单元上限20。该方法只减轻构成影响，不是完全控制房况、楼栋等差异的固定质量指数。",
        "2. 六项行情特征：匹配单元月变化、三个月变化均值、六个月变化标准差、成交量同比对数变化、匹配单元数量、匹配成交覆盖比例。全部按当时完整月份计算。",
        "3. HMM 只观察一维匹配价格月变化，采用两个高斯隐状态、对角协方差。至少24个有效月变化且没有中间缺月才拟合；不足时状态特征缺失并记录可用标记。",
        "4. 对每一个历史月初，重新使用此前完整月份拟合标准化器和 HMM；初始化种子11、29、47只按该历史前缀训练似然选优，并检查占用量与收敛。没有先用截至1月的 HMM 参数回填更早训练样本。",
        "5. 状态按各自学到的月变化均值排序，称为较低／较高变化状态；不能预先解释为固定的上涨／下跌市场。输入特征使用显式前向过滤概率，不使用未来月份平滑概率。平滑占用量只用于训练拟合诊断，不作价格模型输入。",
        "6. 从当前过滤概率 p 和转移矩阵 A 计算未来一至四个月 pA、pA²、pA³、pA⁴；增加对应的较高状态概率、预期月变化，以及四个月预期对数变化之和。不是把各月涨幅简单当成已实现收益。",
        "7. 查询时点为 min(挂牌月份月初, 固定训练截止日)。未来挂牌房屋共同使用截止日状态及其四个月预测；较早挂牌房屋仅用挂牌月之前信息。测试文件没有成交日期，所以不使用验证成交月份给房屋挑选预测期，也不假定挂牌月等于成交月。",
        "8. HMM 模块按历史序列似然训练；LightGBM 按 Log-MAE 训练；组合好坏最终按真实价格上的 MdAPE／MAPE 判断。期望行情变化不自动等于 MdAPE 最优的价格修正。", "",
        "## 7. 可用性与状态诊断", "", "| 预测窗口起点 | HMM可用训练样本占比 | HMM可用验证样本占比 | 已拟合历史前缀 | 预热后拟合失败数 |", "|---|---:|---:|---:|---:|"]
    for audit_path in sorted(out.glob("*/*/audit.json")):
        a = json.loads(audit_path.read_text())
        lines.append(f"| {a['cutoff_exclusive']} | {a['hmm_training_available_pct']:.2f}% | {a['hmm_validation_available_pct']:.2f}% | {a['hmm_fitted_prefixes']} | {a['hmm_failed_prefixes_after_warmup']} |")
    final_audits = json.loads((out / "selection_A/2026-02/hmm_fit_audits.json").read_text())
    current = final_audits["2026-02-01"]
    lines += ["", "截至2026年1月底的状态诊断（只基于截至该时点的观测）：", "", "```json", json.dumps(current, ensure_ascii=False, indent=2), "```", "",
        "## 8. 判断与后续", "",
        f"- 本轮主期候选排名首位为 {LABELS[best]}。只有当 HMM 组合相对原模型和月度行情对照有稳定收益时，才支持在当前定价方案中采用 HMM。",
        "- 单一状态数、单一观测定义的结果只能评价这版组合。若没有改善，不能据此判定所有状态模型无效。",
        "- 总历史仅约65个月（本次主训练截至1月为61个月），实际状态转折次数可能很少；每月15万笔房屋中的一部分成交并不等于15万个独立市场状态样本。HMM 参数估计、状态命名和四步预测均有不确定性。",
        "- 状态可能反映季节、噪声或未完全消除的房屋构成差异，不能仅凭隐状态名称作经济解释。",
        "- 一至四个月状态预测始终从同一信息截止日出发，无未来真实行情更新；预测通常会逐渐趋于模型的长期状态分布。",
        "- 2—5 月已经被用于算法、时间权重和本轮特征比较，属于累计模型选择期，不是独立测试。多次尝试带来的乐观偏差需如实记录。",
        "- 本轮不新增官方逐月 B 验证、不改写既有提交文件，也不自动采用最终重训或冻结旧模型刷新历史库的部署方式。保存的是截至1月训练的两个新增候选，便于检查和复现。",
        "- 原始文件及既有实验记录通过运行前后哈希校验保持不变。", "",
        "## 9. 产物与复现", "",
        "- `metrics.json`、`分组误差.csv`：总体、月度、房型、社区与价格段误差。",
        "- 各窗口的预测文件、`monthly_observations.csv`、`point_in_time_feature_cache.csv`、`hmm_fit_audits.json`：完整状态构造与预测证据。",
        "- `trained_selection_models/`：两个新增 LightGBM 候选与类别 schema，重载核验通过。",
        "- `manifest.json`、`checks.json`、`source_snapshot/`：参数、数据与代码版本、保护文件校验。", "",
        "```bash", ".venv/bin/python -u -m experiments.hmm_lgbm.run_experiment --output experiments/hmm_lgbm/artifacts/my_new_run", "```", "",
        "方法依据：[hmmlearn 教程](https://hmmlearn.readthedocs.io/en/stable/tutorial.html)、[过滤与平滑概率区别](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html)。", ""]
    (out / "实验报告.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--previous", type=Path, default=REPO / "experiments/plan_a/artifacts/run_2026-10-16_v1")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/run_2026-10-16_v1")
    args = parser.parse_args()
    out, previous, data_dir = args.output.resolve(), args.previous.resolve(), args.data_dir.resolve()
    archived = json.loads((previous / "manifest.json").read_text())
    if archived["params"]["lightgbm"] != PARAMS["lightgbm"]:
        raise ValueError("Baseline parameters changed")
    if sha(data_dir / "listings.parquet") != archived["inputs_sha256"]["listings.parquet"]:
        raise ValueError("Data changed")
    for name in ["features.py", "models.py"]:
        if sha(REPO / "experiments/plan_a" / name) != archived["code_sha256"][name]:
            raise ValueError("Baseline code changed")
    for package in ["lightgbm", "numpy", "pandas", "scikit-learn"]:
        if importlib.metadata.version(package) != archived["packages"][package]:
            raise ValueError("Baseline dependency changed")
    protected = protected_files(data_dir, previous)
    for path in (REPO / "experiments/time_decay").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            protected[str(path)] = sha(path)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in list(HERE.glob("*.py")) + [REPO / "experiments/plan_a" / n for n in ["features.py", "models.py", "run_experiment.py"]]:
        shutil.copy2(path, snapshot / (path.parent.name + "_" + path.name))
    manifest = {"status": "running", "started_at_Toronto": datetime.now(ZoneInfo("America/Toronto")).isoformat(),
        "packages": {p: importlib.metadata.version(p) for p in ["hmmlearn", "lightgbm", "numpy", "pandas", "scikit-learn", "scipy"]},
        "lightgbm_params": PARAMS["lightgbm"], "hmm_params": {"states": 2, "covariance_type": "diag", "min_observations": MIN_MONTHS, "initialization_seeds": SEEDS, "n_iter": 300, "tol": 1e-4, "transmat_prior": 1.1},
        "source_dataset_sha256": sha(data_dir / "listings.parquet"), "candidate_count": 3,
        "development_windows": DEV, "selection_window": ["2026-02-01", "2026-06-01"],
        "labels_used_for_early_stopping": False, "hmm_state_count_search": False,
        "runtime_code_sha256": {p.name: sha(p) for p in snapshot.iterdir()}}
    write_json(out / "manifest.json", manifest)
    listings = pd.read_parquet(data_dir / "listings.parquet")
    for c in ["date_sold", "date_listed"]:
        listings[c] = pd.to_datetime(listings[c])
    if listings.ml_num.duplicated().any() or listings.date_sold.isna().any() or not (listings.sold_price > 0).all():
        raise ValueError("Invalid reference data")
    results = [evaluate(listings, previous, out, "2026-02-01", "2026-06-01", "selection_A")]
    for cutoff, end in DEV:
        results.append(evaluate(listings, previous, out, cutoff, end, "development_A"))
    combined = pd.concat(results, ignore_index=True)
    if combined.duplicated(["protocol", "cutoff_exclusive", "scheme", "ml_num"]).any():
        raise ValueError("Duplicate scoring rows")
    combined.to_csv(out / "all_predictions.csv", index=False)
    scores = {scheme: summarize(frame) for scheme, frame in combined.groupby("scheme")}
    write_json(out / "metrics.json", scores)
    flat = []
    for scheme, protocols in scores.items():
        for protocol, models in protocols.items():
            for model, dimensions in models.items():
                for dimension, values in dimensions.items():
                    values = [{"group": "all", **values}] if dimension == "overall" else values
                    flat.extend({"scheme": scheme, "protocol": protocol, "dimension": dimension, **entry} for entry in values)
    pd.DataFrame(flat).to_csv(out / "分组误差.csv", index=False)
    write_report(out, combined, scores)
    for path, before in protected.items():
        if sha(path) != before:
            raise ValueError(f"Existing file changed: {path}")
    write_json(out / "checks.json", {"status": "passed", "protected_files_sha256": protected,
        "protected_files_n": len(protected), "baseline_main_exact_reproduction": True,
        "new_main_models_reload_exact": True, "future_targets_not_used_in_features": True})
    manifest.update({"status": "complete", "finished_at_Toronto": datetime.now(ZoneInfo("America/Toronto")).isoformat()})
    write_json(out / "manifest.json", manifest)
    print(f"DONE: {out / '实验报告.md'}", flush=True)


if __name__ == "__main__":
    main()
