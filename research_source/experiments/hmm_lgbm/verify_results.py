"""Independent scoring and prefix/as-of-state feature reconciliation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/run_2026-10-16_v1")
    parser.add_argument("--data-dir", type=Path, default=repo.parent / "data")
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    source = pd.read_parquet(args.data_dir / "listings.parquet")
    for c in ["date_sold", "date_listed"]:
        source[c] = pd.to_datetime(source[c])
    indexed = source.set_index("ml_num")
    rows = pd.read_csv(out / "all_predictions.csv")
    assert not rows.duplicated(["protocol", "cutoff_exclusive", "scheme", "ml_num"]).any()
    assert np.isfinite(rows.predicted_price).all() and rows.predicted_price.gt(0).all()
    assert np.array_equal(rows.sold_price.to_numpy(), indexed.loc[rows.ml_num, "sold_price"].to_numpy())
    scores = json.loads((out / "metrics.json").read_text())
    aggregates = 0
    for (scheme, protocol), frame in rows.groupby(["scheme", "protocol"]):
        error = 100 * (frame.predicted_price.to_numpy() / frame.sold_price.to_numpy() - 1)
        expected = scores[scheme][protocol]["lightgbm"]["overall"]
        assert expected["n"] == len(frame)
        assert np.isclose(expected["mdape"], np.median(np.abs(error)), rtol=0, atol=1e-10)
        assert np.isclose(expected["mape"], np.mean(np.abs(error)), rtol=0, atol=1e-10)
        aggregates += 1
    fitted_prefixes, window_scores, query_rows = 0, 0, 0
    for audit_path in out.glob("*/*/audit.json"):
        folder = audit_path.parent
        audit = json.loads(audit_path.read_text())
        cutoff, end = pd.Timestamp(audit["cutoff_exclusive"]), pd.Timestamp(audit["end_exclusive"])
        homes = source.loc[(source.date_sold >= cutoff) & (source.date_sold < end)]
        assert int((source.date_sold < cutoff).sum()) == audit["training_n"]
        assert pd.Timestamp(audit["max_training_sale"]) < cutoff
        for prediction in folder.glob("*_predictions.csv"):
            frame = pd.read_csv(prediction)
            assert set(frame.ml_num) == set(homes.ml_num)
            error = 100 * np.abs(frame.predicted_price.to_numpy() / frame.sold_price.to_numpy() - 1)
            expected = audit["scores"][frame.scheme.iloc[0]]
            assert np.isclose(expected["mdape"], np.median(error), atol=1e-10, rtol=0)
            assert np.isclose(expected["mape"], np.mean(error), atol=1e-10, rtol=0)
            window_scores += 1
        cache = pd.read_csv(folder / "point_in_time_feature_cache.csv", index_col="available_from")
        cache.index = pd.to_datetime(cache.index)
        features = pd.read_csv(folder / "validation_market_features.csv")
        assert set(features.ml_num) == set(homes.ml_num)
        listed = indexed.loc[features.ml_num, "date_listed"].reset_index(drop=True)
        boundary = listed.dt.to_period("M").dt.start_time.clip(upper=cutoff)
        expected_features = cache.reindex(pd.DatetimeIndex(boundary)).reset_index(drop=True)
        expected_features["hmm_available"] = expected_features.hmm_available.fillna(0)
        np.testing.assert_allclose(features[expected_features.columns].to_numpy(), expected_features.to_numpy(), atol=1e-12, rtol=1e-12, equal_nan=True)
        query_rows += len(features)
        fits = json.loads((folder / "hmm_fit_audits.json").read_text())
        for boundary, info in fits.items():
            assert pd.Period(info["last_observed_month"], freq="M").end_time < pd.Timestamp(boundary) <= cutoff
            if info["status"] != "fitted":
                continue
            assert info["observation_n"] >= 24
            p, transition, means = map(np.asarray, [info["filtered_probability"], info["transition"], info["state_means_log_change"]])
            assert np.all(np.diff(means) >= 0) and np.all(transition >= 0)
            np.testing.assert_allclose(transition.sum(axis=1), 1)
            assert np.isclose(p.sum(), 1)
            feature = cache.loc[pd.Timestamp(boundary)]
            assert np.isclose(feature.hmm_prob_high_current, p[1])
            cumulative = 0.
            for k in range(1, 5):
                pk = p @ np.linalg.matrix_power(transition, k)
                expected_change = pk @ means
                assert np.isclose(feature[f"hmm_prob_high_plus_{k}"], pk[1], rtol=0, atol=1e-12)
                assert np.isclose(feature[f"hmm_expected_change_plus_{k}"], expected_change, rtol=0, atol=1e-12)
                cumulative += expected_change
            assert np.isclose(feature.hmm_expected_cumulative_change_4, cumulative, rtol=0, atol=1e-12)
            fitted_prefixes += 1
    checks = json.loads((out / "checks.json").read_text())
    assert checks["baseline_main_exact_reproduction"] and checks["new_main_models_reload_exact"]
    for path, expected in checks["protected_files_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    result = {"status": "passed", "prediction_rows_verified": len(rows), "aggregate_scores_recomputed": aggregates,
              "window_scores_recomputed": window_scores, "state_forecast_prefixes_recomputed": fitted_prefixes,
              "validation_asof_feature_rows_verified": query_rows, "protected_files_unchanged": checks["protected_files_n"]}
    (out / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
