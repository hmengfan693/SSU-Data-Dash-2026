"""Four nested temporal backtests and a final GPU ResNet candidate.

Run from the repository root with --device mps (Apple GPU) or --device cuda.
Existing output directories are never overwritten. No outer-score tuning.
"""
import argparse
import gc
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
import torch

import check_submission
import data
from experiments.mlp.run_experiment import archived_rows, split_window
from experiments.plan_a.features import HistoryFeatures
from experiments.plan_a.run_experiment import DEV, metrics, result_rows, summarize
from .models import DEFAULTS, ResNetPriceModel, resolve_device, sha, write_json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def now():
    return datetime.now(ZoneInfo("America/Toronto")).isoformat()


def select_epochs(train, cutoff, params, folder, device):
    cutoff = pd.Timestamp(cutoff)
    if not (train.date_sold < cutoff).all():
        raise ValueError("Outer labels must never enter early stopping")
    inner_cutoff = cutoff - pd.DateOffset(months=2)
    earlier, holdout = split_window(train, inner_cutoff, cutoff)
    history = HistoryFeatures(earlier, inner_cutoff)
    x_train = history.transform(earlier)
    x_valid = history.transform(data.hide_answers(holdout, "off_market"))
    print(f"  inner train={len(earlier):,}, holdout={len(holdout):,}, cutoff={inner_cutoff.date()}", flush=True)

    def log(record):
        if record["epoch"] == 1 or record["epoch"] % 5 == 0:
            print(f"    epoch {record['epoch']:3d}: log-MAE={record['train_log_mae']:.5f}, "
                  f"MdAPE={record['inner_mdape']:.3f}%, {record['seconds']:.2f}s", flush=True)

    model = ResNetPriceModel(device=device, **params).fit(
        x_train, earlier.sold_price, validation=(x_valid, holdout.sold_price), logger=log)
    folder.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(model.training_history).to_csv(folder / "learning_curve.csv", index=False)
    pd.DataFrame(dict(ml_num=earlier.ml_num)).to_csv(folder / "train_ids.csv", index=False)
    pd.DataFrame(dict(ml_num=holdout.ml_num)).to_csv(folder / "holdout_ids.csv", index=False)
    write_json(folder / "audit.json", dict(
        inner_cutoff_exclusive=str(inner_cutoff.date()), outer_cutoff_exclusive=str(cutoff.date()),
        inner_train_n=len(earlier), inner_holdout_n=len(holdout),
        max_inner_training_sale=str(earlier.date_sold.max()), best_epoch=model.best_epoch,
        epochs_run=model.epochs_run, best_inner_mdape=model.training_history[model.best_epoch - 1]["inner_mdape"],
        history_reference_n=history.reference_n, feature_names=list(x_train.columns),
        preprocessing=model.preprocessor.state, target_mean=model.target_mean, target_scale=model.target_scale,
        preprocessing_and_history_fit_on_inner_training_only=True, outer_validation_labels_used=False,
        execution=model.execution, stopping_metric="unrounded price MdAPE; min_delta in percentage points"))
    epochs = model.best_epoch
    print(f"  selected {epochs} epochs (ran {model.epochs_run})", flush=True)
    return epochs


def release_gpu(device):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def evaluate(listings, mlp_archive, tree_archive, out, cutoff, end, protocol, device):
    train, homes = split_window(listings, cutoff, end)
    folder = out / protocol / cutoff[:7]
    folder.mkdir(parents=True, exist_ok=False)
    print(f"[{protocol}] {cutoff} to {end}: train={len(train):,}, evaluate={len(homes):,}", flush=True)
    started = time.monotonic()
    epochs = select_epochs(train, cutoff, DEFAULTS, folder / "inner_selection", device)
    release_gpu(device)
    history = HistoryFeatures(train, cutoff)
    x_train = history.transform(train)
    x_valid = history.transform(data.hide_answers(homes, "off_market"))

    def log(record):
        if record["epoch"] == 1 or record["epoch"] % 5 == 0 or record["epoch"] == epochs:
            print(f"    refit {record['epoch']:3d}/{epochs}: log-MAE={record['train_log_mae']:.5f}, {record['seconds']:.2f}s", flush=True)

    model = ResNetPriceModel(device=device).fit(x_train, train.sold_price, epochs=epochs, logger=log)
    prediction = model.predict(x_valid)
    rows = result_rows(homes, prediction, "resnet", protocol, cutoff)
    rows.to_csv(folder / "resnet_predictions.csv", index=False)
    model.save(folder / "model", dict(training_cutoff_exclusive=cutoff, training_n=len(train), protocol=protocol))
    replay = ResNetPriceModel.load(folder / "model", device=device).predict(x_valid)
    if not np.array_equal(prediction, replay):
        raise ValueError("Same-device checkpoint replay differs")
    controls = []
    for name, previous in [("mlp", mlp_archive), ("lightgbm", tree_archive)]:
        control = archived_rows(previous, homes, name, protocol, cutoff, list(x_train.columns), len(train))
        control.to_csv(folder / f"{name}_archived_predictions.csv", index=False)
        controls.append(control)
    scores = {"resnet": metrics(rows), **{v.model.iloc[0]: metrics(v) for v in controls}}
    write_json(folder / "audit.json", dict(
        protocol=protocol, cutoff_exclusive=cutoff, end_exclusive=end, train_n=len(train),
        validation_n=len(homes), features=list(x_train.columns), max_training_sale=str(train.date_sold.max()),
        selected_epochs=epochs, input_dimension=model.preprocessor.state["numeric_output_dim"] + sum(DEFAULTS["embedding_dims"]),
        parameter_count=sum(p.numel() for p in model.network.parameters()), history_reference_n=history.reference_n,
        history_query="[min(date_listed, cutoff)-lookback, min(date_listed, cutoff))",
        model_and_history_frozen_for_four_months=True, outer_labels_used_for_fit_or_early_stopping=False,
        reload_exact_match=True, execution=model.execution, scores=scores, seconds=time.monotonic() - started))
    print("  " + "; ".join(f"{name}: MdAPE={s['mdape']:.3f}%, MAPE={s['mape']:.3f}%" for name, s in scores.items()), flush=True)
    return pd.concat([rows, *controls], ignore_index=True)


def report(out, rows, scores, final_schema):
    lines = ["# GPU Tabular ResNet 实验报告", "",
             f"实际训练设备：{final_schema['execution']['parameter_device']}；没有开启 MPS CPU fallback。",
             "沿用 MLP 的 64 项特征和预处理，114 维拼接输入；256 维主干，4 个残差块，每块内部 512 维。",
             "每块：LayerNorm → Linear(256,512) → ReLU → Dropout(0.1) → Linear(512,256) → 加回输入。",
             "输出头 LayerNorm → ReLU → Linear(256,1)。10 个 Linear 层；无卷积。",
             f"最终参数量 {final_schema['parameter_count']:,}；最终训练 150,000 条、{final_schema['epochs_run']} 轮。", "",
             "## 主窗口：截至 2026 年 1 月训练，固定预测 2—5 月", "",
             "| 模型 | 样本量 | MdAPE | MAPE | 中位有符号误差 | >30%误差比例 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ["resnet", "mlp", "lightgbm"]:
        s = scores["selection_A"][name]["overall"]
        lines.append(f"| {name} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% | {s['median_signed_error']:+.2f}% | {s['over_30_pct']:.2f}% |")
    lines += ["", "## 主窗口各房型", "", "| 房型 | 样本量 | ResNet MdAPE | MLP MdAPE | LightGBM MdAPE | ResNet MAPE | LightGBM MAPE |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    typed = {name: {s["group"]: s for s in scores["selection_A"][name]["by_property_type"]}
             for name in ["resnet", "mlp", "lightgbm"]}
    for typ, s in typed["resnet"].items():
        m, l = typed["mlp"][typ], typed["lightgbm"][typ]
        lines.append(f"| {typ} | {s['n']:,} | {s['mdape']:.2f}% | {m['mdape']:.2f}% | {l['mdape']:.2f}% | {s['mape']:.2f}% | {l['mape']:.2f}% |")
    lines += ["", "## 四个时间窗口", "", "| 预测起始日 | 模型 | MdAPE | MAPE |", "|---|---|---:|---:|"]
    for (cutoff, name), g in rows.groupby(["cutoff_exclusive", "model"]):
        s = metrics(g)
        lines.append(f"| {cutoff} | {name} | {s['mdape']:.2f}% | {s['mape']:.2f}% |")
    main = scores["selection_A"]
    delta = main["resnet"]["overall"]["mdape"] - main["lightgbm"]["overall"]["mdape"]
    lines += ["", "## 解释与边界", "",
              f"- 主窗口 ResNet 相对 LightGBM 的 MdAPE 差值：{delta:+.2f} 个百分点。",
              "- 只运行预先固定的一种更深更宽结构、种子 42；未根据外层分数再次调参。",
              "- AdamW，lr=0.001，weight_decay=0.0001，batch=512，标准化 log-price L1Loss；没有学习率调度或融合。",
              "- 各窗口用训练期最后两个月内部早停：最多 100 轮，patience=10，最小改善 0.0001 个百分点；选定轮数后从头重新训练完整训练集。",
              "- 内部历史池、词表、中位数填补、标准化、目标缩放均只拟合于更早训练记录。",
              "- 四个月内模型和历史池均冻结；标签按成交日期切分，历史查询按 min(挂牌日,截止日)，并非完整的严格挂牌时点模型回测。",
              "- MLP/LightGBM 复用经哈希、样本ID、标签、特征和时间窗口核对的已有预测。",
              "- 网络结构、归一化和运行后端同时不同，不能把全部差异只归因于残差连接。",
              "- 外层窗口已参与此前模型开发，属于累计开发评价；没有新的独立最终测试。",
              "- MPS 设置固定种子，但不保证逐次训练或跨设备逐比特一致；存档模型在相同后端重载复现。",
              "- 6—9 月没有标签，仅生成候选提交；不报告其误差。原始数据、官方代码、已有模型与正式提交均保留。", "",
              "参考：[Tabular ResNet / FT-Transformer](https://arxiv.org/abs/2106.11959)，",
              "[官方结构](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/resnet.py)。", ""]
    (out / "实验报告.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--mlp-archive", type=Path, default=REPO / "experiments/mlp/artifacts/run_2026-10-17_v2")
    parser.add_argument("--tree-archive", type=Path, default=REPO / "experiments/plan_a/artifacts/run_2026-10-16_v1")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/run_2026-10-19_v1")
    args = parser.parse_args()
    device = resolve_device(args.device).type
    torch.set_num_threads(DEFAULTS["threads"])
    out, data_dir = args.output.resolve(), args.data_dir.resolve()
    mlp_archive, tree_archive = args.mlp_archive.resolve(), args.tree_archive.resolve()
    archives = {name: json.loads((p / "manifest.json").read_text())
                for name, p in [("mlp", mlp_archive), ("lightgbm", tree_archive)]}
    inputs = {p.name: sha(p) for p in [data_dir / "listings.parquet", data_dir / "test_off_market.parquet"]}
    for name in inputs:
        if inputs[name] != archives["mlp"]["input_sha256"][name]:
            raise ValueError(f"MLP archive data differ: {name}")
    if inputs["listings.parquet"] != archives["lightgbm"]["inputs_sha256"]["listings.parquet"]:
        raise ValueError("LightGBM archive data differ")
    if sha(REPO / "experiments/plan_a/features.py") != archives["lightgbm"]["code_sha256"]["features.py"]:
        raise ValueError("Archived feature definition differs")
    for relative, digest in archives["mlp"]["source_sha256"].items():
        if sha(REPO / relative) != digest:
            raise ValueError(f"MLP source differs: {relative}")
    for relative, digest in archives["mlp"]["output_sha256"].items():
        if sha(mlp_archive / relative) != digest:
            raise ValueError(f"MLP archive corrupt: {relative}")
    protected_paths = [*data_dir.glob("*.parquet"), *data_dir.glob("*.csv"),
                       *[p for a in [mlp_archive, tree_archive] for p in a.rglob("*") if p.is_file()],
                       *[REPO / p for p in ["models.py", "features.py", "run.py", "data.py", "check_submission.py"]],
                       *[p for p in (REPO / "submissions").glob("*") if p.is_file()]]
    protected = {str(p): sha(p) for p in protected_paths}
    source_paths = [*HERE.glob("*.py"), HERE / "README.md", HERE / "requirements.txt",
                    REPO / "experiments/mlp/models.py", REPO / "experiments/mlp/run_experiment.py",
                    REPO / "experiments/mlp/verify_results.py",
                    *[REPO / "experiments/plan_a" / name for name in ["__init__.py", "features.py", "models.py", "run_experiment.py"]],
                    REPO / "data.py", REPO / "check_submission.py", REPO / "models.py",
                    REPO / "tests/test_resnet.py"]
    source_hashes = {str(p.relative_to(REPO)): sha(p) for p in source_paths}
    listings = pd.read_parquet(data_dir / "listings.parquet")
    test = pd.read_parquet(data_dir / "test_off_market.parquet")
    for c in ["date_listed", "date_sold"]:
        listings[c] = pd.to_datetime(listings[c])
    test["date_listed"] = pd.to_datetime(test.date_listed)
    if (listings.ml_num.duplicated().any() or test.ml_num.duplicated().any()
            or set(listings.ml_num).intersection(test.ml_num)
            or not (listings.date_sold < pd.Timestamp("2026-06-01")).all()):
        raise ValueError("Invalid training boundaries/IDs")
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_at_Toronto=now(), device=device, python=platform.python_version(), platform=platform.platform(),
                    packages={p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "pyarrow"]},
                    params=DEFAULTS, input_sha256=inputs, source_sha256=source_hashes,
                    protected_sha256=protected, seed_count=1, parameter_search=False,
                    development_windows=DEV, selection_window=["2026-02-01", "2026-06-01"],
                    official_B_run=False, outer_labels_used_for_early_stopping=False,
                    inner_holdout_months=2, training_loss="MAE(standardized log(sold_price))",
                    cpu_fallback_enabled=False, architecture_reference="https://arxiv.org/abs/2106.11959")
    write_json(out / "manifest.json", manifest)
    for path in source_paths:
        target = out / "source_snapshot" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(f"GPU device={device}; width=256, residual blocks=4, hidden=512", flush=True)
    frames = []
    for cutoff, end in DEV:
        frames.append(evaluate(listings, mlp_archive, tree_archive, out, cutoff, end, "development_A", device))
        release_gpu(device)
    frames.append(evaluate(listings, mlp_archive, tree_archive, out, "2026-02-01", "2026-06-01", "selection_A", device))
    release_gpu(device)
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(out / "all_predictions.csv", index=False)
    scores = summarize(rows)
    write_json(out / "metrics.json", scores)
    groups = []
    for protocol, models in scores.items():
        for name, dims in models.items():
            for dimension, values in dims.items():
                for s in ([dict(group="all", **values)] if dimension == "overall" else values):
                    groups.append(dict(protocol=protocol, model=name, dimension=dimension, **s))
    pd.DataFrame(groups).to_csv(out / "分组误差.csv", index=False)
    print("[final] internal selection then full GPU refit through May 2026", flush=True)
    train = split_window(listings, "2026-06-01")
    epochs = select_epochs(train, "2026-06-01", DEFAULTS, out / "final_inner_selection", device)
    release_gpu(device)
    history = HistoryFeatures(train, "2026-06-01")
    x_train, x_test = history.transform(train), history.transform(data.hide_answers(test, "off_market"))
    model = ResNetPriceModel(device=device).fit(x_train, train.sold_price, epochs=epochs,
        logger=lambda r: print(f"    final {r['epoch']:3d}/{epochs}: log-MAE={r['train_log_mae']:.5f}, {r['seconds']:.2f}s", flush=True)
        if r["epoch"] == 1 or r["epoch"] % 5 == 0 or r["epoch"] == epochs else None)
    prediction = model.predict(x_test)
    model.save(out / "trained_model", dict(training_cutoff_exclusive="2026-06-01", training_n=len(train),
               input_sha256=inputs, feature_source_sha256=sha(REPO / "experiments/plan_a/features.py")))
    if not np.array_equal(prediction, ResNetPriceModel.load(out / "trained_model", device=device).predict(x_test)):
        raise ValueError("Final same-device replay differs")
    final_dir = out / "final_predictions"
    final_dir.mkdir()
    submission = final_dir / "submission_off_market_resnet.csv"
    pd.DataFrame(dict(ml_num=test.ml_num, predicted_price=prediction)).to_csv(submission, index=False)
    errors = check_submission.check_file(submission, test.ml_num)
    if errors:
        raise ValueError(errors)
    unchanged = all(sha(p) == h for p, h in protected.items())
    source_unchanged = all(sha(REPO / p) == h for p, h in source_hashes.items())
    if not unchanged or not source_unchanged:
        raise ValueError("Protected data/artifacts or running source changed")
    write_json(out / "checks.json", dict(protected_files_unchanged=unchanged, protected_file_count=len(protected),
               source_unchanged=source_unchanged, all_five_model_reloads_exact=True,
               submission_errors=errors, submission_n=len(test), final_training_n=len(train), final_epochs=epochs))
    final_schema = json.loads((out / "trained_model/schema.json").read_text())
    report(out, rows, scores, final_schema)
    manifest["completed_at_Toronto"] = now()
    manifest["output_sha256"] = {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
                                if p.is_file() and p.name != "manifest.json"}
    write_json(out / "manifest.json", manifest)
    print(f"Completed: {out / '实验报告.md'}", flush=True)


if __name__ == "__main__":
    main()
