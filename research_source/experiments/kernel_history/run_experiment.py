"""Compare frozen LightGBM with/without causal kernel historical features."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from lightgbm import Booster

import check_submission
import data
from experiments.plan_a.features import HistoryFeatures
from experiments.plan_a.models import PARAMS, PriceModel
from experiments.plan_a.run_experiment import DEV, metrics, result_rows, sha, summarize, write_json
from .features import CONFIG, KERNEL_COLUMNS, KernelHistoryFeatures

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def split(listings, cutoff, end=None):
    train = listings.loc[listings.date_sold < pd.Timestamp(cutoff)].sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    if train.empty:
        raise ValueError("Empty training period")
    if end is None:
        return train
    homes = listings.loc[(listings.date_sold >= pd.Timestamp(cutoff)) & (listings.date_sold < pd.Timestamp(end))].reset_index(drop=True)
    if homes.empty or set(train.ml_num).intersection(homes.ml_num):
        raise ValueError("Invalid validation period")
    return train, homes


def save_model(model, folder, cutoff, data_hashes):
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "lightgbm.txt"
    model.model.booster_.save_model(str(path))
    write_json(folder / "schema.json", dict(features=model.feature_names, categories=model.categories,
               params=PARAMS["lightgbm"], kernel_config=CONFIG, target="log(sold_price)",
               cutoff_exclusive=cutoff, weights_sha256=sha(path), inputs_sha256=data_hashes,
               source_sha256={str(p.relative_to(REPO)): sha(p) for p in
                              [HERE / "features.py", REPO / "experiments/plan_a/features.py", REPO / "experiments/plan_a/models.py"]}))


def load_model(folder):
    schema = json.loads((folder / "schema.json").read_text())
    if sha(folder / "lightgbm.txt") != schema["weights_sha256"]:
        raise ValueError("Saved model hash mismatch")
    model = PriceModel("lightgbm")
    model.feature_names, model.categories = schema["features"], schema["categories"]
    model.model = Booster(model_file=str(folder / "lightgbm.txt"))
    return model, schema


def archived_control(previous, homes, cutoff, protocol, train_n, original_columns):
    folder = previous / protocol / cutoff[:7]
    audit = json.loads((folder / "audit.json").read_text())
    if audit["train_n"] != train_n or audit["features"] != original_columns:
        raise ValueError("Archive training period or feature schema changed")
    rows = pd.read_csv(folder / "lightgbm_predictions.csv")
    if rows.ml_num.duplicated().any() or set(rows.ml_num) != set(homes.ml_num):
        raise ValueError("Archived validation IDs differ")
    rows = rows.set_index("ml_num").loc[homes.ml_num].reset_index()
    for column in ["sold_price", "house_type_name", "community"]:
        if not np.array_equal(rows[column], homes[column]):
            raise ValueError(f"Archived {column} differs")
    for column, value in [("model", "lightgbm"), ("protocol", protocol), ("cutoff_exclusive", cutoff)]:
        if not rows[column].eq(value).all():
            raise ValueError("Archived model or time boundary mismatch")
    if not np.array_equal(rows.month, homes.date_sold.dt.strftime("%Y-%m")):
        raise ValueError("Archived months mismatch")
    return rows


def evaluate(listings, previous, out, cutoff, end, protocol, data_hashes):
    start = time.monotonic()
    train, homes = split(listings, cutoff, end)
    folder = out / protocol / cutoff[:7]
    folder.mkdir(parents=True, exist_ok=False)
    print(f"[{protocol}] train < {cutoff}: {len(train):,}; evaluate {len(homes):,}", flush=True)
    history = KernelHistoryFeatures(train, cutoff)
    x_train = history.transform(train, logger=lambda value: print(value, flush=True))
    train_audit = history.last_audit.copy()
    x_valid = history.transform(data.hide_answers(homes, "off_market"))
    valid_audit = history.last_audit.copy()
    original_columns = [c for c in x_train if c not in KERNEL_COLUMNS]
    control = archived_control(previous, homes, cutoff, protocol, len(train), original_columns)
    control.to_csv(folder / "lightgbm_archived_predictions.csv", index=False)
    control_reproduced = None
    if protocol == "selection_A":
        print("  Refit original-feature control to verify exact archived reproduction", flush=True)
        baseline = PriceModel("lightgbm").fit(x_train[original_columns], train.sold_price)
        control_reproduced = bool(np.array_equal(baseline.predict(x_valid[original_columns]), control.predicted_price))
        if not control_reproduced:
            raise ValueError("Original-feature control failed exact reproduction")
    print(f"  Fitting kernel LightGBM with {len(x_train.columns)} features", flush=True)
    fit_start = time.monotonic()
    model = PriceModel("lightgbm").fit(x_train, train.sold_price)
    prediction = model.predict(x_valid)
    fit_seconds = time.monotonic() - fit_start
    rows = result_rows(homes, prediction, "lightgbm_kernel", protocol, cutoff)
    rows.to_csv(folder / "lightgbm_kernel_predictions.csv", index=False)
    save_model(model, folder / "model", cutoff, data_hashes)
    replay, _ = load_model(folder / "model")
    if not np.array_equal(prediction, replay.predict(x_valid)):
        raise ValueError("Kernel model reload does not reproduce predictions")
    gain = model.model.booster_.feature_importance("gain")
    pd.DataFrame(dict(feature=x_train.columns, gain=gain, gain_share_pct=100 * gain / gain.sum())).sort_values(
        "gain", ascending=False).to_csv(folder / "feature_importance.csv", index=False)
    for label, homes_frame, features in [("training", train, x_train), ("validation", homes, x_valid)]:
        features[KERNEL_COLUMNS].assign(ml_num=homes_frame.ml_num.to_numpy()).to_parquet(
            folder / f"{label}_kernel_features.parquet", index=False)
    scores = dict(lightgbm=metrics(control), lightgbm_kernel=metrics(rows))
    audit = dict(cutoff_exclusive=cutoff, end_exclusive=end, protocol=protocol,
                 training_n=len(train), validation_n=len(homes), max_training_sale=str(train.date_sold.max()),
                 original_feature_count=len(original_columns), feature_count=len(x_train.columns),
                 features=list(x_train.columns), kernel_config=CONFIG, params=PARAMS["lightgbm"],
                 training_kernel=train_audit, validation_kernel=valid_audit,
                 validation_labels_used_for_fit_or_early_stopping=False,
                 no_early_stopping=True, no_bandwidth_search=True,
                 model_and_reference_pool_frozen_for_four_months=True,
                 model_reload_exact=True, main_control_reproduced_exact=control_reproduced,
                 scores=scores, fit_predict_seconds=fit_seconds, total_seconds=time.monotonic() - start)
    write_json(folder / "audit.json", audit)
    print(f"  MdAPE {scores['lightgbm']['mdape']:.3f}% -> {scores['lightgbm_kernel']['mdape']:.3f}%; "
          f"MAPE {scores['lightgbm']['mape']:.3f}% -> {scores['lightgbm_kernel']['mape']:.3f}%; "
          f"validation fallback {valid_audit['fallback_pct']:.2f}%", flush=True)
    return pd.concat([control, rows], ignore_index=True)


def report(out, rows, scores, checks):
    main = scores["selection_A"]
    baseline, kernel = main["lightgbm"]["overall"], main["lightgbm_kernel"]["overall"]
    better = (kernel["mdape"], kernel["mape"]) < (baseline["mdape"], baseline["mape"])
    lines = ["# LightGBM＋核回归历史特征：首版实验", "",
             f"**主窗口 {'改善' if better else '未改善'}：MdAPE {baseline['mdape']:.2f}% → {kernel['mdape']:.2f}%，MAPE {baseline['mape']:.2f}% → {kernel['mape']:.2f}%。**", "",
             "本次仅研究 Off-market，在原 64 项输入上加入 9 项核特征，LightGBM 的 600 轮、Log-MAE、树参数与随机种子保持不变。只有一套预先固定的核设置，没有根据结果调带宽或筛选特征。", "",
             "## 各窗口", "", "| 训练截止日（不含） | 模型 | 房屋数 | MdAPE | MAPE |", "|---|---|---:|---:|---:|"]
    for (cutoff, name), group in rows.groupby(["cutoff_exclusive", "model"]):
        score = metrics(group)
        lines.append(f"| {cutoff} | {name} | {score['n']:,} | {score['mdape']:.2f}% | {score['mape']:.2f}% |")
    lines += ["", "## 主窗口逐月", "", "| 月份 | 模型 | MdAPE | MAPE | 中位有符号误差 |", "|---|---|---:|---:|---:|"]
    for (month, name), group in rows.query("protocol == 'selection_A'").groupby(["month", "model"]):
        score = metrics(group)
        lines.append(f"| {month} | {name} | {score['mdape']:.2f}% | {score['mape']:.2f}% | {score['median_signed_error']:+.2f}% |")
    lines += ["", "## 核特征的具体计算", "",
              "查询日期 q=min(挂牌时间, 窗口固定训练截止日)，只取 [q−365天,q) 的同房型成交；先过滤时间再检索邻居，同日成交、目标房屋自身及未来成交不能进入。",
              "面积、卫生间、经纬度完整的候选按距离平方 d²=(Δlog面积/0.25)²+(Δ卫生间/1)²+(Δ纬度/0.02)²+(Δ经度/0.03)² 检索最多 64 个最近邻，舍弃 d>3 的候选。",
              "权重 w=exp(−d²/2−成交距查询日期天数/180)。面积差 0.25 是自然对数单位；坐标带宽是合成坐标的度数，不解释为真实街区距离。时间常数 180 天是 e-fold，不是半衰期。",
              "对 log价格、log单价取归一化加权均值；计算 log价格加权总体标准差、有效邻居数 (Σw)²/Σw²、邻居数、平均成交年龄、最近距离。另加入 log单价均值+目标 log面积作为面积调整后的估价。",
              "有效邻居数<5 或目标几何输入缺失时，价格与单价回退到原同社区×房型的 365 天平滑统计；父级也无历史则保持缺失。保留回退标记、有效样本量和邻居数；回退时局部离散程度及平均年龄为缺失。",
              "没有拟合全量标准化或填补参数；距离尺度全部预先固定。训练早期缺少历史的样本同样按上述规则处理。", "",
              "## 每窗口核特征覆盖", "", "| 截止日 | 训练回退比例 | 验证回退比例 | 验证平均有效邻居数 |", "|---|---:|---:|---:|"]
    for path in sorted(out.glob("*_A/*/audit.json")):
        audit = json.loads(path.read_text())
        train, valid = audit["training_kernel"], audit["validation_kernel"]
        lines.append(f"| {audit['cutoff_exclusive']} | {train['fallback_pct']:.2f}% | {valid['fallback_pct']:.2f}% | {valid['mean_effective_n']:.1f} |")
    lines += ["", "## 主窗口核特征重要性", "", "以下是训练分裂收益，不能解释为因果贡献。", "",
              "| 核特征 | Gain占比 |", "|---|---:|"]
    importance = pd.read_csv(out / "selection_A/2026-02/feature_importance.csv")
    for row in importance.loc[importance.feature.isin(KERNEL_COLUMNS)].itertuples():
        lines.append(f"| {row.feature} | {row.gain_share_pct:.2f}% |")
    lines += ["", "## 验收和边界", "",
              "- 主窗口原 64 特征模型重新训练，与此前存档逐套完全一致。其他窗口对照复用方案 A，并核验数据、源码、参数、训练数、ID、标签、月份和截止日。",
              "- 四个核模型及最终核模型保存后重载，预测完全一致；各窗口完整核特征保存为 parquet 以便复算。",
              "- 每个四个月窗口内模型和参考成交池均冻结；已有 2—5 月属于累计开发评价，不是独立最终测试。没有新增官方 B 逐月重训。",
              "- 模型按成交日期截止训练，历史特征按 min(挂牌日期,截止日) 查询，不是严格挂牌当日的完整模型回测。",
              "- 同房型邻居并不强制同社区；社区仍是主模型输入。当前结果只适用于本轮带宽、邻居数和支持范围，不能证明其他核参数无效或全面优越。",
              f"- 最终候选用全部 {checks['final_training_n']:,} 条成交重训，另存 {checks['submission_n']:,} 套预测，格式检查通过；既有提交不替换，隐藏测试没有误差可报。",
              "- 原始数据、官方代码、方案 A 源码与存档哈希保持不变。", "",
              "参考：[Nadaraya–Watson 局部常数核回归](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html)、[SciPy 精确 KD-tree 查询](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)。", ""]
    (out / "实验报告.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--previous", type=Path, default=REPO / "experiments/plan_a/artifacts/run_2026-10-16_v1")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/run_2026-10-18_v1")
    args = parser.parse_args()
    out, previous, data_dir = args.output.resolve(), args.previous.resolve(), args.data_dir.resolve()
    old = json.loads((previous / "manifest.json").read_text())
    if sha(data_dir / "listings.parquet") != old["inputs_sha256"]["listings.parquet"] or PARAMS != old["params"]:
        raise ValueError("Original data or model configuration changed")
    for filename in ["features.py", "models.py"]:
        if sha(REPO / "experiments/plan_a" / filename) != old["code_sha256"][filename]:
            raise ValueError("Original features/models changed")
    source = [*HERE.glob("*.py"), *[REPO / p for p in ["experiments/plan_a/features.py", "experiments/plan_a/models.py",
                "experiments/plan_a/run_experiment.py", "data.py", "check_submission.py"]]]
    protected = [*data_dir.glob("*.parquet"), *data_dir.glob("*.csv"),
                 *[p for p in previous.rglob("*") if p.is_file()],
                 *[REPO / p for p in ["models.py", "features.py", "run.py", "data.py"]],
                 *[p for p in (REPO / "submissions").glob("*") if p.is_file()]]
    hashes = {str(p): sha(p) for p in protected}
    data_hashes = {name: sha(data_dir / name) for name in ["listings.parquet", "test_off_market.parquet"]}
    source_hashes = {str(p.relative_to(REPO)): sha(p) for p in source}
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_at_Toronto=datetime.now(ZoneInfo("America/Toronto")).isoformat(),
                    python=platform.python_version(), kernel_config=CONFIG, params=PARAMS["lightgbm"],
                    kernel_columns=KERNEL_COLUMNS, bandwidth_search=False, early_stopping=False,
                    packages={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "lightgbm", "pyarrow"]},
                    inputs_sha256=data_hashes, source_sha256=source_hashes, protected_sha256=hashes,
                    previous_manifest_sha256=sha(previous / "manifest.json"),
                    development_windows=DEV, selection_window=["2026-02-01", "2026-06-01"])
    write_json(out / "manifest.json", manifest)
    for path in source:
        target = out / "source_snapshot" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    listings = pd.read_parquet(data_dir / "listings.parquet")
    test = pd.read_parquet(data_dir / "test_off_market.parquet")
    for column in ["date_sold", "date_listed"]:
        listings[column] = pd.to_datetime(listings[column])
    test["date_listed"] = pd.to_datetime(test.date_listed)
    if (listings.ml_num.duplicated().any() or test.ml_num.duplicated().any()
            or set(listings.ml_num).intersection(test.ml_num)
            or not (listings.date_sold < pd.Timestamp("2026-06-01")).all()):
        raise ValueError("Invalid IDs or training time range")
    frames = [evaluate(listings, previous, out, cutoff, end, "development_A", data_hashes) for cutoff, end in DEV]
    frames.append(evaluate(listings, previous, out, "2026-02-01", "2026-06-01", "selection_A", data_hashes))
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(out / "all_predictions.csv", index=False)
    scores = summarize(rows)
    write_json(out / "metrics.json", scores)
    groups = []
    for protocol, models in scores.items():
        for name, dimensions in models.items():
            for dimension, values in dimensions.items():
                for value in ([dict(group="all", **values)] if dimension == "overall" else values):
                    groups.append(dict(protocol=protocol, model=name, dimension=dimension, **value))
    pd.DataFrame(groups).to_csv(out / "分组误差.csv", index=False)
    print("[final] Refit frozen kernel candidate on all sales through May", flush=True)
    train = split(listings, "2026-06-01")
    history = KernelHistoryFeatures(train, "2026-06-01")
    x_train = history.transform(train, logger=lambda value: print(value, flush=True))
    train_audit = history.last_audit.copy()
    x_test = history.transform(data.hide_answers(test, "off_market"))
    test_audit = history.last_audit.copy()
    model = PriceModel("lightgbm").fit(x_train, train.sold_price)
    prediction = model.predict(x_test)
    save_model(model, out / "trained_model", "2026-06-01", data_hashes)
    loaded, _ = load_model(out / "trained_model")
    if not np.array_equal(loaded.predict(x_test), prediction):
        raise ValueError("Final model replay differs")
    folder = out / "final_predictions"
    folder.mkdir()
    path = folder / "submission_off_market_kernel.csv"
    pd.DataFrame(dict(ml_num=test.ml_num, predicted_price=prediction)).to_csv(path, index=False)
    errors = check_submission.check_file(path, test.ml_num)
    if errors:
        raise ValueError(errors)
    x_test[KERNEL_COLUMNS].assign(ml_num=test.ml_num.to_numpy()).to_parquet(out / "final_test_kernel_features.parquet", index=False)
    if any(sha(p) != digest for p, digest in hashes.items()) or any(sha(REPO / p) != digest for p, digest in source_hashes.items()):
        raise ValueError("Protected files or running source changed")
    checks = dict(protected_files_unchanged=True, protected_file_count=len(hashes),
                  source_unchanged=True, all_five_model_reloads_exact=True,
                  main_baseline_refit_exact=True, submission_errors=errors,
                  final_training_n=len(train), submission_n=len(test), final_training_kernel=train_audit,
                  final_test_kernel=test_audit)
    write_json(out / "checks.json", checks)
    report(out, rows, scores, checks)
    manifest["completed_at_Toronto"] = datetime.now(ZoneInfo("America/Toronto")).isoformat()
    manifest["outputs_sha256"] = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    write_json(out / "manifest.json", manifest)
    print(f"Complete: {out / '实验报告.md'}", flush=True)


if __name__ == "__main__":
    main()
