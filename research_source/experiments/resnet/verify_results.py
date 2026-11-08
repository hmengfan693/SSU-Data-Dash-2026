"""Independent metrics, training-state/temporal checks and GPU checkpoint replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.mlp.models import Preprocessor
from experiments.mlp.run_experiment import split_window
from experiments.mlp.verify_results import independent_metrics, same_metrics, verify_inner
from experiments.plan_a.features import HistoryFeatures
from .models import ResNetPriceModel, sha, write_json
from .run_experiment import REPO


def verify_gpu(execution):
    backend = execution["device"]
    if backend not in ["mps", "cuda"] or execution["cpu_batches"] or not execution["gpu_batches"]:
        raise ValueError("Training did not exclusively execute GPU batches")
    for key in ["parameter_device", "numeric_device", "categorical_device", "target_device"]:
        if not execution[key].startswith(backend):
            raise ValueError(f"Incorrect training device for {key}")
    if execution["cpu_fallback_enabled"]:
        raise ValueError("CPU fallback was enabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--device", choices=["auto", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    out = args.artifacts.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    checks = dict(verification_source_sha256=sha(Path(__file__)), artifact_hashes_checked=0,
                  protected_files_checked=0, source_files_checked=0, windows={})
    for relative, digest in manifest["output_sha256"].items():
        if sha(out / relative) != digest:
            raise ValueError(f"Artifact hash mismatch: {relative}")
        checks["artifact_hashes_checked"] += 1
    for path, digest in manifest["protected_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"Protected file changed: {path}")
        checks["protected_files_checked"] += 1
    for relative, digest in manifest["source_sha256"].items():
        if sha(out / "source_snapshot" / relative) != digest or sha(REPO / relative) != digest:
            raise ValueError(f"Source hash mismatch: {relative}")
        checks["source_files_checked"] += 1
    for relative, digest in manifest["input_sha256"].items():
        if sha(args.data_dir / relative) != digest:
            raise ValueError(f"Input hash mismatch: {relative}")
    listings = pd.read_parquet(args.data_dir / "listings.parquet")
    for c in ["date_sold", "date_listed"]:
        listings[c] = pd.to_datetime(listings[c])
    all_rows = pd.read_csv(out / "all_predictions.csv")
    scores = json.loads((out / "metrics.json").read_text())
    groups_checked = 0
    for protocol, models in scores.items():
        for name, entry in models.items():
            rows = all_rows.loc[(all_rows.protocol == protocol) & (all_rows.model == name)]
            if rows.ml_num.duplicated().any():
                raise ValueError("Duplicate pooled prediction IDs")
            same_metrics(independent_metrics(rows), entry["overall"])
            groups_checked += 1
            for dimension, column in [("by_month", "month"), ("by_property_type", "house_type_name"), ("by_community", "community")]:
                for expected in entry[dimension]:
                    same_metrics(independent_metrics(rows.loc[rows[column].astype(str) == expected["group"]]), expected)
                    groups_checked += 1
            bands = pd.cut(rows.sold_price, [0, 5e5, 1e6, 1.5e6, 2e6, np.inf],
                           labels=["<500k", "500k-1m", "1m-1.5m", "1.5m-2m", ">=2m"], right=False)
            for expected in entry["by_price_band"]:
                same_metrics(independent_metrics(rows.loc[bands == expected["group"]]), expected)
                groups_checked += 1
    checks["metric_groups_checked"] = groups_checked
    checks["prediction_rows_checked"] = len(all_rows)
    for audit_path in sorted(out.glob("*_A/*/audit.json")):
        folder = audit_path.parent
        audit = json.loads(audit_path.read_text())
        cutoff, end = audit["cutoff_exclusive"], audit["end_exclusive"]
        train, homes = split_window(listings, cutoff, end)
        inner = verify_inner(train, cutoff, folder / "inner_selection", manifest["params"])
        verify_gpu(json.loads((folder / "inner_selection/audit.json").read_text())["execution"])
        if len(train) != audit["train_n"] or len(homes) != audit["validation_n"]:
            raise ValueError("Window counts differ")
        history = HistoryFeatures(train, cutoff)
        x_train = history.transform(train)
        x_valid = history.transform(data.hide_answers(homes, "off_market"))
        model = ResNetPriceModel.load(folder / "model", device=args.device)
        verify_gpu(model.execution)
        if model.execution != audit["execution"]:
            raise ValueError("GPU execution evidence differs")
        if Preprocessor().fit(x_train).state != model.preprocessor.state:
            raise ValueError("Outer preprocessing mismatch")
        if model.epochs_run != inner["selected_epochs"] or audit["selected_epochs"] != model.epochs_run:
            raise ValueError("Refit epochs differ from inner selection")
        log_prices = np.log(train.sold_price.to_numpy())
        if model.target_mean != float(log_prices.mean()) or model.target_scale != float(log_prices.std()):
            raise ValueError("Outer target scaling mismatch")
        rows = pd.read_csv(folder / "resnet_predictions.csv")
        if not np.array_equal(rows.ml_num, homes.ml_num) or not np.array_equal(rows.sold_price, homes.sold_price):
            raise ValueError("Outer labels/IDs differ")
        if not np.array_equal(rows.predicted_price, model.predict(x_valid)):
            raise ValueError("Saved model replay differs")
        for name in ["resnet", "mlp", "lightgbm"]:
            filename = "resnet_predictions.csv" if name == "resnet" else f"{name}_archived_predictions.csv"
            frame = pd.read_csv(folder / filename)
            if not np.array_equal(frame.ml_num, homes.ml_num) or not np.array_equal(frame.sold_price, homes.sold_price):
                raise ValueError("Control labels/IDs differ")
            same_metrics(independent_metrics(frame), audit["scores"][name])
            pooled = all_rows.loc[(all_rows.protocol == audit["protocol"]) & (all_rows.cutoff_exclusive == cutoff) & (all_rows.model == name)]
            if not np.array_equal(frame.ml_num, pooled.ml_num) or not np.array_equal(frame.predicted_price, pooled.predicted_price):
                raise ValueError("Pooled predictions differ")
        key = f"{audit['protocol']}/{cutoff}"
        checks["windows"][key] = dict(**inner, evaluation_n=len(homes), reload_exact_match=True, training_device=model.execution["device"])
        print(f"Verified {key}: {len(homes):,} exact predictions on {model.device}", flush=True)
    final_train = split_window(listings, "2026-06-01")
    inner = verify_inner(final_train, "2026-06-01", out / "final_inner_selection", manifest["params"])
    verify_gpu(json.loads((out / "final_inner_selection/audit.json").read_text())["execution"])
    test = pd.read_parquet(args.data_dir / "test_off_market.parquet")
    test["date_listed"] = pd.to_datetime(test.date_listed)
    history = HistoryFeatures(final_train, "2026-06-01")
    model = ResNetPriceModel.load(out / "trained_model", device=args.device)
    verify_gpu(model.execution)
    if Preprocessor().fit(history.transform(final_train)).state != model.preprocessor.state:
        raise ValueError("Final preprocessing mismatch")
    if model.epochs_run != inner["selected_epochs"]:
        raise ValueError("Final refit epoch mismatch")
    log_price = np.log(final_train.sold_price.to_numpy())
    if model.target_mean != float(log_price.mean()) or model.target_scale != float(log_price.std()):
        raise ValueError("Final target scaling mismatch")
    predicted = model.predict(history.transform(data.hide_answers(test, "off_market")))
    path = out / "final_predictions/submission_off_market_resnet.csv"
    submission = pd.read_csv(path)
    if check_submission.check_file(path, test.ml_num) or not np.array_equal(submission.ml_num, test.ml_num) or not np.array_equal(submission.predicted_price, predicted):
        raise ValueError("Final predictions or format differ")
    checks["final"] = dict(**inner, predictions_n=len(test), reload_exact_match=True, submission_format_valid=True,
                           training_device=model.execution["device"], parameter_count=sum(p.numel() for p in model.network.parameters()))
    checks["passed"] = True
    write_json(out / "independent_verification.json", checks)
    print(f"Verified {len(all_rows):,} prediction rows, {groups_checked} metric groups and all five GPU model replays", flush=True)


if __name__ == "__main__":
    main()
