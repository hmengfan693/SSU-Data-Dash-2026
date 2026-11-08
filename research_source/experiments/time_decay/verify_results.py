"""Independently recompute scores, date boundaries and row-aligned weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/run_2026-10-16_h12_v1")
    parser.add_argument("--data-dir", type=Path, default=repo.parent / "data")
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    rows = pd.read_csv(out / "all_predictions.csv")
    source = pd.read_parquet(args.data_dir / "listings.parquet")
    source["date_sold"] = pd.to_datetime(source.date_sold)
    labels = source.set_index("ml_num")
    assert not rows.duplicated(["protocol", "cutoff_exclusive", "model", "scheme", "ml_num"]).any()
    assert np.isfinite(rows.predicted_price).all() and rows.predicted_price.gt(0).all()
    windows = pd.read_csv(out / "窗口对照.csv")
    checked_windows = 0
    for keys, frame in rows.groupby(["protocol", "cutoff_exclusive", "model", "scheme"]):
        protocol, cutoff, model, scheme = keys
        audit = json.loads((out / protocol / cutoff[:7] / "audit.json").read_text())
        start, end = pd.Timestamp(cutoff), pd.Timestamp(audit["end_exclusive"])
        expected = source.loc[(source.date_sold >= start) & (source.date_sold < end)]
        assert set(frame.ml_num) == set(expected.ml_num)
        assert np.array_equal(frame.sold_price.to_numpy(), labels.loc[frame.ml_num, "sold_price"].to_numpy())
        assert audit["training_n"] == int(source.date_sold.lt(start).sum())
        assert pd.Timestamp(audit["max_training_date"]) < start
        assert audit["validation_labels_used_for_training_or_early_stopping"] is False
        assert audit["model_and_history_frozen_for_four_months"] is True
        result = windows.loc[(windows.protocol == protocol) & (windows.cutoff_exclusive == cutoff) & (windows.model == model) & (windows.scheme == scheme)].iloc[0]
        absolute = np.abs(frame.predicted_price.to_numpy() / frame.sold_price.to_numpy() - 1) * 100
        assert int(result.n) == len(frame)
        assert np.isclose(result.mdape, np.median(absolute), rtol=0, atol=1e-10)
        assert np.isclose(result.mape, np.mean(absolute), rtol=0, atol=1e-10)
        checked_windows += 1
    scores = json.loads((out / "metrics.json").read_text())
    checked_aggregates = 0
    for (scheme, protocol, model), frame in rows.groupby(["scheme", "protocol", "model"]):
        expected = scores[scheme][protocol][model]["overall"]
        absolute = np.abs(frame.predicted_price.to_numpy() / frame.sold_price.to_numpy() - 1) * 100
        assert expected["n"] == len(frame)
        assert np.isclose(expected["mdape"], np.median(absolute), rtol=0, atol=1e-10)
        assert np.isclose(expected["mape"], np.mean(absolute), rtol=0, atol=1e-10)
        checked_aggregates += 1
    weights = pd.read_csv(out / "selection_A/2026-02/training_weights.csv")
    training = source.loc[source.date_sold < "2026-02-01"].sort_values(["date_sold", "ml_num"])
    assert np.array_equal(weights.ml_num.to_numpy(), training.ml_num.to_numpy())
    ages_days = (pd.Timestamp("2026-02-01") - pd.to_datetime(weights.date_sold)).dt.total_seconds().to_numpy() / 86400
    expected_weights = np.exp2(-ages_days / 365.25)
    expected_weights /= expected_weights.mean()
    np.testing.assert_allclose(weights.sample_weight, expected_weights, rtol=1e-12, atol=1e-12)
    checks = json.loads((out / "checks.json").read_text())
    assert all(v["exact_match"] for v in checks["unweighted_main_reproduction"].values())
    for path, expected_hash in checks["protected_files_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash, path
    result = {"status": "passed", "prediction_rows_recomputed": len(rows),
              "window_model_scheme_scores_recomputed": checked_windows,
              "aggregate_scores_recomputed": checked_aggregates,
              "row_aligned_training_weights_recomputed": len(weights),
              "protected_files_unchanged": checks["protected_files_n"],
              "unweighted_control_exact_reproduction": True}
    (out / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
