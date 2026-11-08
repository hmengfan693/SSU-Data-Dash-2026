"""Independently recompute metrics, temporal preprocessing and saved predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.plan_a.features import HistoryFeatures
from .models import MLPPriceModel, Preprocessor, sha, write_json
from .run_experiment import REPO, split_window


def independent_metrics(frame):
    # Deliberately do not call the experiment's metric implementation.
    error = (frame.predicted_price.to_numpy() - frame.sold_price.to_numpy()) / frame.sold_price.to_numpy() * 100
    return dict(n=len(frame), mdape=float(np.median(np.abs(error))),
                mape=float(np.mean(np.abs(error))), median_signed_error=float(np.median(error)),
                over_30_pct=float(np.mean(np.abs(error) > 30) * 100))


def same_metrics(actual, expected):
    for key, value in actual.items():
        if not np.isclose(value, expected[key], atol=1e-10, rtol=1e-12):
            raise ValueError(f"Metric mismatch: {key}: {value} != {expected[key]}")


def verify_inner(train, cutoff, folder, params):
    audit = json.loads((folder / "audit.json").read_text())
    inner = pd.Timestamp(cutoff) - pd.DateOffset(months=2)
    earlier, holdout = split_window(train, inner, cutoff)
    for filename, expected in [("train_ids.csv", earlier.ml_num), ("holdout_ids.csv", holdout.ml_num)]:
        actual = pd.read_csv(folder / filename).ml_num
        if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
            raise ValueError("Inner train/holdout IDs differ")
    if set(earlier.ml_num).intersection(holdout.ml_num):
        raise ValueError("Inner train/holdout overlap")
    if audit["inner_cutoff_exclusive"] != str(inner.date()) or audit["outer_cutoff_exclusive"] != cutoff:
        raise ValueError("Inner cutoff differs")
    features = HistoryFeatures(earlier, inner).transform(earlier)
    preprocessor = Preprocessor().fit(features)
    if preprocessor.state != audit["preprocessing"]:
        raise ValueError("Preprocessing did not fit only on inner training rows")
    log_prices = np.log(earlier.sold_price.to_numpy())
    if not np.isclose(log_prices.mean(), audit["target_mean"]) or not np.isclose(max(log_prices.std(), 1e-8), audit["target_scale"]):
        raise ValueError("Inner target scaling mismatch")
    curve = pd.read_csv(folder / "learning_curve.csv")
    best, best_epoch, stale = np.inf, None, 0
    for row in curve.itertuples():
        if stale >= params["patience"]:
            raise ValueError("Training continued after early stopping")
        if row.inner_mdape < best - params["min_delta_mdape"]:
            best, best_epoch, stale = row.inner_mdape, row.epoch, 0
        else:
            stale += 1
    if best_epoch != audit["best_epoch"] or len(curve) != audit["epochs_run"]:
        raise ValueError("Selected early stopping epoch differs")
    return dict(inner_train_n=len(earlier), inner_holdout_n=len(holdout), selected_epochs=int(best_epoch),
                preprocessing_exact_match=True, targets_training_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
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
            same_metrics(independent_metrics(rows), entry["overall"])
            groups_checked += 1
            for dimension, column in [("by_month", "month"), ("by_property_type", "house_type_name"), ("by_community", "community")]:
                for expected in entry[dimension]:
                    frame = rows.loc[rows[column].astype(str) == expected["group"]]
                    same_metrics(independent_metrics(frame), expected)
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
        if len(train) != audit["train_n"] or len(homes) != audit["validation_n"]:
            raise ValueError("Outer window counts differ")
        history = HistoryFeatures(train, cutoff)
        x_train = history.transform(train)
        x_valid = history.transform(data.hide_answers(homes, "off_market"))
        model = MLPPriceModel.load(folder / "model")
        if Preprocessor().fit(x_train).state != model.preprocessor.state:
            raise ValueError("Outer preprocessing mismatch")
        if model.epochs_run != inner["selected_epochs"] or audit["selected_epochs"] != model.epochs_run:
            raise ValueError("Refit epochs differ from inner selection")
        log_prices = np.log(train.sold_price.to_numpy())
        if model.target_mean != float(log_prices.mean()) or model.target_scale != float(log_prices.std()):
            raise ValueError("Outer target scaling mismatch")
        rows = pd.read_csv(folder / "mlp_predictions.csv")
        if not np.array_equal(rows.ml_num, homes.ml_num) or not np.array_equal(rows.sold_price, homes.sold_price):
            raise ValueError("Outer labels/IDs differ")
        predicted = model.predict(x_valid)
        if not np.array_equal(rows.predicted_price, predicted):
            raise ValueError("Outer saved model replay differs")
        for name in ["mlp", "lightgbm", "catboost"]:
            filename = "mlp_predictions.csv" if name == "mlp" else f"{name}_archived_predictions.csv"
            frame = pd.read_csv(folder / filename)
            same_metrics(independent_metrics(frame), audit["scores"][name])
            pooled = all_rows.loc[(all_rows.protocol == audit["protocol"]) & (all_rows.cutoff_exclusive == cutoff) & (all_rows.model == name)]
            if not np.array_equal(frame.ml_num, pooled.ml_num) or not np.array_equal(frame.predicted_price, pooled.predicted_price):
                raise ValueError("Pooled predictions differ from window predictions")
        key = f"{audit['protocol']}/{cutoff}"
        checks["windows"][key] = dict(**inner, evaluation_n=len(homes), reload_exact_match=True)
        print(f"Verified {key}: {len(homes):,} exact predictions", flush=True)
    final_train = split_window(listings, "2026-06-01")
    inner = verify_inner(final_train, "2026-06-01", out / "final_inner_selection", manifest["params"])
    test = pd.read_parquet(args.data_dir / "test_off_market.parquet")
    test["date_listed"] = pd.to_datetime(test.date_listed)
    history = HistoryFeatures(final_train, "2026-06-01")
    final_model = MLPPriceModel.load(out / "trained_model")
    if Preprocessor().fit(history.transform(final_train)).state != final_model.preprocessor.state:
        raise ValueError("Final preprocessing mismatch")
    if final_model.epochs_run != inner["selected_epochs"]:
        raise ValueError("Final refit epoch mismatch")
    predicted = final_model.predict(history.transform(data.hide_answers(test, "off_market")))
    path = out / "final_predictions/submission_off_market_mlp.csv"
    submission = pd.read_csv(path)
    if check_submission.check_file(path, test.ml_num) or not np.array_equal(submission.ml_num, test.ml_num) or not np.array_equal(submission.predicted_price, predicted):
        raise ValueError("Final predictions or submission format differ")
    checks["final"] = dict(**inner, predictions_n=len(test), reload_exact_match=True, submission_format_valid=True)
    checks["passed"] = True
    write_json(out / "independent_verification.json", checks)
    print(f"Verified {len(all_rows):,} prediction rows, {groups_checked} metric groups and all five model replays", flush=True)


if __name__ == "__main__":
    main()
