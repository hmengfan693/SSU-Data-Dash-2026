"""Independent brute-force kernel checks, metric recomputation and replays."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.plan_a.features import HistoryFeatures, property_features
from experiments.plan_a.run_experiment import sha, write_json
from .features import KERNEL_COLUMNS, KernelHistoryFeatures
from .run_experiment import REPO, load_model, split


def direct_kernel(reference, homes, cutoff, config):
    """Full eligible-pool scan and explicit weights; no KD-tree helper."""
    ref = reference.reset_index(drop=True)
    base, targets = property_features(ref), property_features(homes)
    parent = HistoryFeatures(ref, cutoff).transform(homes)
    output, tied_boundaries = [], []
    ref_dates = ref.date_sold.to_numpy(dtype="datetime64[ns]")
    ref_types = ref.house_type_name.to_numpy()
    for j in range(len(homes)):
        row = targets.iloc[j]
        q = min(pd.Timestamp(homes.date_listed.iloc[j]), pd.Timestamp(cutoff))
        eligible = ((ref_dates < q.to_datetime64()) & (ref_dates >= (q - pd.Timedelta(days=config["lookback_days"])).to_datetime64())
                    & (ref_dates < pd.Timestamp(cutoff).to_datetime64()) & (ref_types == row.house_type_name))
        ids = np.flatnonzero(eligible)
        d2 = np.zeros(len(ids))
        for column, bandwidth in [("log_area", "log_area_bandwidth"), ("washrooms", "washrooms_bandwidth"),
                                  ("latitude", "latitude_bandwidth"), ("longitude", "longitude_bandwidth")]:
            d2 += np.square((base[column].to_numpy()[ids] - row[column]) / config[bandwidth])
        finite = np.isfinite(d2)
        ids, d2 = ids[finite], d2[finite]
        order = np.argsort(d2, kind="stable")
        k = int(config["max_neighbors"])
        tied = len(order) > k and abs(d2[order[k]] - d2[order[k - 1]]) < 1e-11
        tied_boundaries.append(tied)
        order = order[:k]
        ids, d2 = ids[order], d2[order]
        keep = d2 <= config["max_distance"] ** 2
        ids, d2 = ids[keep], d2[keep]
        age = np.asarray((q.to_datetime64() - ref_dates[ids]) / np.timedelta64(1, "D"), dtype=float)
        weights = np.exp(-d2 / 2 - age / config["time_efold_days"])
        effective = weights.sum() ** 2 / np.square(weights).sum() if len(ids) else 0.
        good = effective >= config["min_effective_n"]
        if good:
            prices = np.log(ref.sold_price.to_numpy()[ids])
            price = np.average(prices, weights=weights)
            ppsf = np.average(prices - base.log_area.to_numpy()[ids], weights=weights)
            std = np.sqrt(np.average(np.square(prices - price), weights=weights))
            mean_age = np.average(age, weights=weights)
        else:
            price, ppsf = parent.iloc[j][["hist_community_type_365_log_price", "hist_community_type_365_log_ppsf"]]
            std, mean_age = np.nan, np.nan
        output.append([price, ppsf, std, effective, len(ids), int(not good), mean_age,
                       np.sqrt(d2.min()) if len(ids) else np.nan, ppsf + row.log_area])
    return pd.DataFrame(output, columns=KERNEL_COLUMNS), np.asarray(tied_boundaries)


def independent_metrics(rows):
    signed = (rows.predicted_price.to_numpy() - rows.sold_price.to_numpy()) / rows.sold_price.to_numpy() * 100
    return dict(n=len(rows), mdape=float(np.median(abs(signed))), mape=float(np.mean(abs(signed))),
                median_signed_error=float(np.median(signed)), over_30_pct=float(100 * np.mean(abs(signed) > 30)))


def check_metrics(rows, expected):
    for name, value in independent_metrics(rows).items():
        if not np.isclose(value, expected[name], atol=1e-10, rtol=1e-12):
            raise ValueError(f"Metric differs: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    args = parser.parse_args()
    out = args.artifacts.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    for relative, digest in manifest["outputs_sha256"].items():
        if sha(out / relative) != digest:
            raise ValueError(f"Artifact changed: {relative}")
    for path, digest in manifest["protected_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"Protected file changed: {path}")
    for relative, digest in manifest["source_sha256"].items():
        if sha(REPO / relative) != digest or sha(out / "source_snapshot" / relative) != digest:
            raise ValueError(f"Source changed: {relative}")
    for name, digest in manifest["inputs_sha256"].items():
        if sha(args.data_dir / name) != digest:
            raise ValueError(f"Data changed: {name}")
    listings = pd.read_parquet(args.data_dir / "listings.parquet")
    for c in ["date_sold", "date_listed"]:
        listings[c] = pd.to_datetime(listings[c])
    rows = pd.read_csv(out / "all_predictions.csv")
    scores = json.loads((out / "metrics.json").read_text())
    metric_count = 0
    for protocol, models in scores.items():
        for name, dimensions in models.items():
            subset = rows.loc[(rows.protocol == protocol) & (rows.model == name)]
            check_metrics(subset, dimensions["overall"])
            metric_count += 1
            for dimension, column in [("by_month", "month"), ("by_property_type", "house_type_name"), ("by_community", "community")]:
                for expected in dimensions[dimension]:
                    check_metrics(subset.loc[subset[column].astype(str) == expected["group"]], expected)
                    metric_count += 1
            band = pd.cut(subset.sold_price, [0, 5e5, 1e6, 1.5e6, 2e6, np.inf],
                          labels=["<500k", "500k-1m", "1m-1.5m", "1.5m-2m", ">=2m"], right=False)
            for expected in dimensions["by_price_band"]:
                check_metrics(subset.loc[band == expected["group"]], expected)
                metric_count += 1
    results, direct_count, skipped_ties = {}, 0, 0
    for audit_path in sorted(out.glob("*_A/*/audit.json")):
        folder = audit_path.parent
        audit = json.loads(audit_path.read_text())
        cutoff, end = audit["cutoff_exclusive"], audit["end_exclusive"]
        train, homes = split(listings, cutoff, end)
        if len(train) != audit["training_n"] or len(homes) != audit["validation_n"]:
            raise ValueError("Temporal window counts differ")
        for label, frame in [("training", train), ("validation", homes)]:
            saved = pd.read_parquet(folder / f"{label}_kernel_features.parquet")
            if not np.array_equal(saved.ml_num, frame.ml_num):
                raise ValueError("Stored kernel feature IDs differ")
            ids = np.unique(np.r_[np.arange(min(5, len(frame))),
                                  np.random.default_rng(42).choice(len(frame), min(100, len(frame)), replace=False)])
            sample = frame.iloc[ids].reset_index(drop=True)
            expected, ties = direct_kernel(train, data.hide_answers(sample, "off_market"), cutoff, manifest["kernel_config"])
            actual = saved.iloc[ids][KERNEL_COLUMNS].reset_index(drop=True)
            # Kth exact distance ties permit different, equally valid neighbors;
            # exclude those from value comparison and count them explicitly.
            np.testing.assert_allclose(actual.loc[~ties].to_numpy(), expected.loc[~ties].to_numpy(),
                                       atol=1e-9, rtol=1e-9, equal_nan=True)
            direct_count += int((~ties).sum())
            skipped_ties += int(ties.sum())
        history = KernelHistoryFeatures(train, cutoff, **manifest["kernel_config"])
        x_valid = history.transform(data.hide_answers(homes, "off_market"))
        saved = pd.read_parquet(folder / "validation_kernel_features.parquet")
        np.testing.assert_allclose(x_valid[KERNEL_COLUMNS], saved[KERNEL_COLUMNS], atol=0, rtol=0, equal_nan=True)
        model, schema = load_model(folder / "model")
        if schema["params"] != manifest["params"] or schema["kernel_config"] != manifest["kernel_config"]:
            raise ValueError("Saved configuration mismatch")
        prediction = model.predict(x_valid)
        predicted_rows = pd.read_csv(folder / "lightgbm_kernel_predictions.csv")
        if not np.array_equal(predicted_rows.ml_num, homes.ml_num) or not np.array_equal(prediction, predicted_rows.predicted_price):
            raise ValueError("Saved model does not reproduce predictions")
        for name in ["lightgbm", "lightgbm_kernel"]:
            filename = "lightgbm_archived_predictions.csv" if name == "lightgbm" else "lightgbm_kernel_predictions.csv"
            frame = pd.read_csv(folder / filename)
            if not np.array_equal(frame.sold_price, homes.sold_price):
                raise ValueError("Validation label mismatch")
            check_metrics(frame, audit["scores"][name])
            pooled = rows.loc[(rows.model == name) & (rows.protocol == audit["protocol"]) & (rows.cutoff_exclusive == cutoff)]
            if not np.array_equal(frame.ml_num, pooled.ml_num) or not np.array_equal(frame.predicted_price, pooled.predicted_price):
                raise ValueError("Pooled predictions differ")
        results[f"{audit['protocol']}/{cutoff}"] = dict(training_n=len(train), validation_n=len(homes),
               full_validation_feature_replay_exact=True, model_replay_exact=True)
        print(f"Verified {cutoff}: {len(homes):,} predictions and brute-force kernel samples", flush=True)
    final_train = split(listings, "2026-06-01")
    test = pd.read_parquet(args.data_dir / "test_off_market.parquet")
    test.date_listed = pd.to_datetime(test.date_listed)
    history = KernelHistoryFeatures(final_train, "2026-06-01", **manifest["kernel_config"])
    x_test = history.transform(data.hide_answers(test, "off_market"))
    expected_features = pd.read_parquet(out / "final_test_kernel_features.parquet")
    np.testing.assert_allclose(x_test[KERNEL_COLUMNS], expected_features[KERNEL_COLUMNS], atol=0, rtol=0, equal_nan=True)
    model, _ = load_model(out / "trained_model")
    path = out / "final_predictions/submission_off_market_kernel.csv"
    submission = pd.read_csv(path)
    if (check_submission.check_file(path, test.ml_num) or not np.array_equal(submission.ml_num, test.ml_num)
            or not np.array_equal(model.predict(x_test), submission.predicted_price)):
        raise ValueError("Final candidate replay or format failed")
    verification = dict(passed=True, metric_groups_checked=metric_count, prediction_rows_checked=len(rows),
                        brute_force_kernel_rows_checked=direct_count, kth_neighbor_tie_rows_skipped=skipped_ties,
                        all_five_model_replays_exact=True, final_predictions_n=len(test),
                        final_submission_format_valid=True, output_hashes_checked=len(manifest["outputs_sha256"]),
                        protected_hashes_checked=len(manifest["protected_sha256"]), windows=results,
                        verifier_sha256=sha(Path(__file__)))
    write_json(out / "independent_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
