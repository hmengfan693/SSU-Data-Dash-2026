"""Four fixed-horizon off-market backtests, nested early stopping and final refit.

Run from repository root: .venv/bin/python -u -m experiments.mlp.run_experiment
Only new output directories are accepted; existing submissions stay untouched.
"""
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

import check_submission
import data
from experiments.plan_a.features import HistoryFeatures
from experiments.plan_a.models import PARAMS
from experiments.plan_a.run_experiment import DEV, metrics, result_rows, summarize
from .models import DEFAULTS, MLPPriceModel, sha, write_json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def split_window(listings, cutoff, end=None):
    cutoff = pd.Timestamp(cutoff)
    train = listings.loc[listings.date_sold < cutoff].sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    if train.empty:
        raise ValueError("No training records before cutoff")
    if end is None:
        return train
    homes = listings.loc[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp(end))].reset_index(drop=True)
    if homes.empty or set(train.ml_num).intersection(homes.ml_num):
        raise ValueError("Empty or overlapping temporal validation")
    return train, homes


def select_epochs(train, cutoff, params, folder):
    """Rebuild *both* feature sets at an earlier, frozen inner cutoff."""
    cutoff = pd.Timestamp(cutoff)
    if not (train.date_sold < cutoff).all():
        raise ValueError("Outer labels must never enter early stopping")
    inner_cutoff = cutoff - pd.DateOffset(months=2)
    earlier, holdout = split_window(train, inner_cutoff, cutoff)
    history = HistoryFeatures(earlier, inner_cutoff)
    x_train = history.transform(earlier)
    x_valid = history.transform(data.hide_answers(holdout, "off_market"))
    print(f"  inner: train < {inner_cutoff.date()}, holdout {len(holdout):,} rows", flush=True)

    def log(record):
        if record["epoch"] == 1 or record["epoch"] % 5 == 0:
            print(f"    epoch {record['epoch']:3d}: log-MAE={record['train_log_mae']:.5f}, inner MdAPE={record['inner_mdape']:.3f}%", flush=True)

    model = MLPPriceModel(**params).fit(x_train, earlier.sold_price,
                                       validation=(x_valid, holdout.sold_price), logger=log)
    folder.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(model.training_history).to_csv(folder / "learning_curve.csv", index=False)
    pd.DataFrame({"ml_num": earlier.ml_num}).to_csv(folder / "train_ids.csv", index=False)
    pd.DataFrame({"ml_num": holdout.ml_num}).to_csv(folder / "holdout_ids.csv", index=False)
    write_json(folder / "audit.json", dict(
        inner_cutoff_exclusive=str(inner_cutoff.date()), outer_cutoff_exclusive=str(cutoff.date()),
        inner_train_n=len(earlier), inner_holdout_n=len(holdout),
        max_inner_training_sale=str(earlier.date_sold.max()),
        best_epoch=model.best_epoch, epochs_run=model.epochs_run,
        best_inner_mdape=model.training_history[model.best_epoch - 1]["inner_mdape"],
        history_reference_n=history.reference_n,
        feature_names=list(x_train.columns), preprocessing=model.preprocessor.state,
        target_mean=model.target_mean, target_scale=model.target_scale,
        preprocessing_and_history_fit_on_inner_training_only=True,
        outer_validation_labels_used=False,
        stopping_metric="unrounded price MdAPE; min_delta in percentage points"))
    print(f"  selected {model.best_epoch} epochs (ran {model.epochs_run})", flush=True)
    return model.best_epoch


def archived_rows(previous, homes, name, protocol, cutoff, feature_names, train_n):
    folder = previous / protocol / cutoff[:7]
    audit = json.loads((folder / "audit.json").read_text())
    if audit["features"] != feature_names or audit["train_n"] != train_n:
        raise ValueError("Archived features or training window differ")
    rows = pd.read_csv(folder / f"{name}_predictions.csv")
    if rows.ml_num.duplicated().any() or set(rows.ml_num) != set(homes.ml_num):
        raise ValueError("Archived IDs differ from this validation window")
    rows = rows.set_index("ml_num").loc[homes.ml_num].reset_index()
    for c in ["sold_price", "house_type_name", "community"]:
        if not np.array_equal(rows[c].to_numpy(), homes[c].to_numpy()):
            raise ValueError(f"Archived {c} mismatch")
    expected = {"model": name, "protocol": protocol, "cutoff_exclusive": cutoff}
    if any(not rows[c].eq(value).all() for c, value in expected.items()):
        raise ValueError("Archived protocol mismatch")
    if not np.array_equal(rows.month.to_numpy(), homes.date_sold.dt.strftime("%Y-%m").to_numpy()):
        raise ValueError("Archived months differ")
    price = rows.predicted_price.to_numpy()
    if not np.isfinite(price).all() or not (price > 0).all():
        raise ValueError("Invalid archived predictions")
    return rows


def evaluate(listings, previous, out, cutoff, end, protocol, params):
    train, homes = split_window(listings, cutoff, end)
    folder = out / protocol / cutoff[:7]
    folder.mkdir(parents=True, exist_ok=False)
    print(f"[{protocol}] cutoff={cutoff}: train={len(train):,}, evaluate={len(homes):,}", flush=True)
    started = time.monotonic()
    epochs = select_epochs(train, cutoff, params, folder / "inner_selection")
    history = HistoryFeatures(train, cutoff)
    x_train = history.transform(train)
    x_valid = history.transform(data.hide_answers(homes, "off_market"))
    model = MLPPriceModel(**params).fit(x_train, train.sold_price, epochs=epochs)
    prediction = model.predict(x_valid)
    rows = result_rows(homes, prediction, "mlp", protocol, cutoff)
    rows.to_csv(folder / "mlp_predictions.csv", index=False)
    model.save(folder / "model", dict(training_cutoff_exclusive=cutoff, protocol=protocol))
    replay = MLPPriceModel.load(folder / "model").predict(x_valid)
    if not np.array_equal(prediction, replay):
        raise ValueError("Saved model predictions differ")
    controls = []
    for name in ["lightgbm", "catboost"]:
        control = archived_rows(previous, homes, name, protocol, cutoff, list(x_train.columns), len(train))
        control.to_csv(folder / f"{name}_archived_predictions.csv", index=False)
        controls.append(control)
    scores = {"mlp": metrics(rows), **{r.model.iloc[0]: metrics(r) for r in controls}}
    write_json(folder / "audit.json", dict(
        protocol=protocol, cutoff_exclusive=cutoff, end_exclusive=end,
        train_n=len(train), validation_n=len(homes), features=list(x_train.columns),
        max_training_sale=str(train.date_sold.max()), selected_epochs=epochs,
        input_dimension=model.preprocessor.state["numeric_output_dim"] + sum(params["embedding_dims"]),
        parameter_count=sum(p.numel() for p in model.network.parameters()),
        history_reference_n=history.reference_n,
        history_query="[min(date_listed, cutoff)-lookback, min(date_listed, cutoff))",
        model_and_history_frozen_for_four_months=True,
        outer_labels_used_for_fit_or_early_stopping=False,
        reload_exact_match=True, scores=scores, seconds=time.monotonic() - started))
    print("  " + "; ".join(f"{name}: MdAPE={s['mdape']:.3f}%, MAPE={s['mape']:.3f}%" for name, s in scores.items()), flush=True)
    return pd.concat([rows, *controls], ignore_index=True)


def report(out, rows, scores, final_epochs):
    main = scores["selection_A"]
    lines = ["# Off-market MLP 实验报告", "", "类别 embedding + 128→64→32 MLP；相同 64 项原始特征，训练目标为标准化对数价格 MAE。", "",
             "## 主窗口：截至 2026 年 1 月训练，固定预测 2—5 月", "",
             "| 模型 | 房屋数 | MdAPE | MAPE | 中位有符号误差 | >30%误差比例 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ["mlp", "lightgbm", "catboost"]:
        s = main[name]["overall"]
        lines.append(f"| {name} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% | {s['median_signed_error']:+.2f}% | {s['over_30_pct']:.2f}% |")
    delta = main["mlp"]["overall"]["mdape"] - main["lightgbm"]["overall"]["mdape"]
    lines += ["", f"MLP 相比既有 LightGBM 的主窗口 MdAPE 变化 {delta:+.2f} 个百分点。", "",
              "## 各时间窗口", "", "| 截止日（不含） | 模型 | MdAPE | MAPE |", "|---|---|---:|---:|"]
    for (cutoff, name), group in rows.groupby(["cutoff_exclusive", "model"]):
        s = metrics(group)
        lines.append(f"| {cutoff} | {name} | {s['mdape']:.2f}% | {s['mape']:.2f}% |")
    lines += ["", "## 主窗口逐月", "", "| 月份 | 模型 | MdAPE | MAPE |", "|---|---|---:|---:|"]
    for (month, name), group in rows.query("protocol == 'selection_A'").groupby(["month", "model"]):
        s = metrics(group)
        lines.append(f"| {month} | {name} | {s['mdape']:.2f}% | {s['mape']:.2f}% |")
    lines += ["", "## 方法与边界", "",
              "- 每个窗口内部用最后两个月成交做早停验证；内层历史池、填补、标准化、词表、目标缩放全部重新拟合于更早成交。",
              "- 早停依据内层未取整预测的 MdAPE（耐心 10、最大 100 轮）；用选定轮数从头训练整个外层训练集。外层指标按取整到元的预测计算。",
              "- 参数在外层评分前固定，无参数搜索、模型融合或根据外层分数调整网络。仅运行种子 42，未评估种子间波动。",
              "- 类别维度 4/4/4/8/8；数值偏斜列做 log1p、中位数填补、连续列标准化，并添加训练时观察到的缺失指示；未知类别使用固定零向量。",
              "- 所有模型使用同样的原始特征定义；MLP 另做输入编码，因此不能将差异单独归因于网络结构。",
              "- 树模型预测复用已保存方案 A，并核验原始数据、特征源文件、参数、ID、标签、分组、月份和截止日。",
              "- 2—5 月已在此前实验用于模型选择，本轮仍是累计开发评价，不是独立最终测试。",
              "- 每个四个月外层窗口内模型与历史池均不更新；本次没有新增官方逐月重训 B 回测。",
              "- 模型按成交日期截止训练；历史查询按 min(挂牌日, 截止日)，不是严格挂牌当日的完整模型回测。",
              "- 隐藏测试没有标签，不能报告 6—9 月误差。未改动既有模型、官方提交文件或原始数据。",
              f"- 最终用截至 5 月的 150,000 条记录训练 MLP；4—5 月内部选出 {final_epochs} 轮，预测保存在 final_predictions/submission_off_market_mlp.csv，仅作为候选。",
              "- Checkpoint 包含权重、词表、填补值、缩放参数、目标缩放与哈希；各窗口和最终模型均重载核验。",
              "- CPU 固定种子、确定性运算；跨 PyTorch 版本或平台不保证逐比特一致。", "",
              "参考：[PyTorch Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)、[AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)、[复现限制](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。", ""]
    (out / "实验报告.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--previous", type=Path, default=REPO / "experiments/plan_a/artifacts/run_2026-10-16_v1")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/run_2026-10-17_v2")
    args = parser.parse_args()
    out, previous, data_dir = args.output.resolve(), args.previous.resolve(), args.data_dir.resolve()
    source_files = [*HERE.glob("*.py"), HERE / "requirements.txt",
                    REPO / "experiments/plan_a/features.py", REPO / "experiments/plan_a/models.py",
                    REPO / "experiments/plan_a/run_experiment.py", REPO / "data.py", REPO / "check_submission.py"]
    protected_paths = [*data_dir.glob("*.parquet"), *data_dir.glob("*.csv"),
                       *[p for p in previous.rglob("*") if p.is_file()],
                       *[REPO / name for name in ["models.py", "features.py", "run.py", "data.py"]],
                       *[p for p in (REPO / "submissions").glob("*") if p.is_file()]]
    protected = {str(p): sha(p) for p in protected_paths}
    source_hashes = {str(p.relative_to(REPO)): sha(p) for p in source_files}
    old = json.loads((previous / "manifest.json").read_text())
    if old["params"] != PARAMS or sha(data_dir / "listings.parquet") != old["inputs_sha256"]["listings.parquet"]:
        raise ValueError("Original data or archived model configuration changed")
    if sha(REPO / "experiments/plan_a/features.py") != old["code_sha256"]["features.py"]:
        raise ValueError("Original feature source changed")
    listings = pd.read_parquet(data_dir / "listings.parquet")
    test = pd.read_parquet(data_dir / "test_off_market.parquet")
    for c in ["date_listed", "date_sold"]:
        listings[c] = pd.to_datetime(listings[c])
    test["date_listed"] = pd.to_datetime(test.date_listed)
    if (listings.ml_num.duplicated().any() or test.ml_num.duplicated().any()
            or set(listings.ml_num).intersection(test.ml_num)
            or not (listings.date_sold < pd.Timestamp("2026-06-01")).all()):
        raise ValueError("Invalid IDs or training date boundary")
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_at_Toronto=datetime.now(ZoneInfo("America/Toronto")).isoformat(),
                    python=platform.python_version(), platform=platform.platform(),
                    packages={p: importlib.metadata.version(p) for p in
                              ["torch", "numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "catboost"]},
                    params=DEFAULTS, seed_count=1, parameter_search=False,
                    source_sha256=source_hashes,
                    input_sha256={p.name: sha(p) for p in [data_dir / "listings.parquet", data_dir / "test_off_market.parquet"]},
                    previous_manifest_sha256=sha(previous / "manifest.json"),
                    protected_sha256=protected,
                    development_windows=DEV, selection_window=["2026-02-01", "2026-06-01"],
                    official_B_run=False, outer_labels_used_for_early_stopping=False,
                    inner_holdout_months=2, device="cpu", training_loss="MAE(standardized log(sold_price))")
    write_json(out / "manifest.json", manifest)
    for path in source_files:
        target = out / "source_snapshot" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    frames = [evaluate(listings, previous, out, cutoff, end, "development_A", DEFAULTS)
              for cutoff, end in DEV]
    frames.append(evaluate(listings, previous, out, "2026-02-01", "2026-06-01", "selection_A", DEFAULTS))
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(out / "all_predictions.csv", index=False)
    scores = summarize(rows)
    write_json(out / "metrics.json", scores)
    group_rows = []
    for protocol, models in scores.items():
        for name, groups in models.items():
            for dimension, values in groups.items():
                for score in ([dict(group="all", **values)] if dimension == "overall" else values):
                    group_rows.append(dict(protocol=protocol, model=name, dimension=dimension, **score))
    pd.DataFrame(group_rows).to_csv(out / "分组误差.csv", index=False)
    print("[final] selecting epochs inside January 2021-May 2026, then refitting all training rows", flush=True)
    train = split_window(listings, "2026-06-01")
    epochs = select_epochs(train, "2026-06-01", DEFAULTS, out / "final_inner_selection")
    history = HistoryFeatures(train, "2026-06-01")
    x_train, x_test = history.transform(train), history.transform(data.hide_answers(test, "off_market"))
    model = MLPPriceModel(**DEFAULTS).fit(x_train, train.sold_price, epochs=epochs)
    prediction = model.predict(x_test)
    metadata = dict(training_cutoff_exclusive="2026-06-01", training_n=len(train),
                    input_sha256=manifest["input_sha256"],
                    feature_source_sha256=sha(REPO / "experiments/plan_a/features.py"))
    model.save(out / "trained_model", metadata)
    if not np.array_equal(MLPPriceModel.load(out / "trained_model").predict(x_test), prediction):
        raise ValueError("Final model replay differs")
    final_dir = out / "final_predictions"
    final_dir.mkdir()
    submission = final_dir / "submission_off_market_mlp.csv"
    pd.DataFrame(dict(ml_num=test.ml_num, predicted_price=prediction)).to_csv(submission, index=False)
    errors = check_submission.check_file(submission, test.ml_num)
    if errors:
        raise ValueError(errors)
    unchanged = all(sha(path) == digest for path, digest in protected.items())
    source_unchanged = all(sha(REPO / path) == digest for path, digest in source_hashes.items())
    if not unchanged or not source_unchanged:
        raise ValueError("Protected data/artifacts or running code changed")
    write_json(out / "checks.json", dict(protected_files_unchanged=unchanged,
               protected_file_count=len(protected), source_unchanged=source_unchanged,
               all_five_model_reloads_exact=True, submission_errors=errors,
               submission_n=len(test), final_training_n=len(train), final_epochs=epochs))
    report(out, rows, scores, epochs)
    manifest["completed_at_Toronto"] = datetime.now(ZoneInfo("America/Toronto")).isoformat()
    manifest["output_sha256"] = {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
                                if p.is_file() and p.name != "manifest.json"}
    write_json(out / "manifest.json", manifest)
    print(f"Completed. Report: {out / '实验报告.md'}", flush=True)


if __name__ == "__main__":
    main()
