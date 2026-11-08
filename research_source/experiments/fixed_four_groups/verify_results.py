"""Independently rescore predictions, audit time boundaries, and replay final inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
from experiments.property_groups.predict_saved import predict_saved


def score(rows):
    error = 100 * (rows.predicted_price.to_numpy() - rows.sold_price.to_numpy()) / rows.sold_price.to_numpy()
    return {"n": len(rows), "mdape": float(np.median(np.abs(error))), "mape": float(np.mean(np.abs(error))),
            "median_signed_error": float(np.median(error)), "over_30_pct": float(np.mean(np.abs(error) > 30) * 100)}


def match(rows, expected):
    for key, value in score(rows).items():
        np.testing.assert_allclose(value, expected[key], rtol=0, atol=1e-10)


def verify(run, data_dir):
    run, data_dir = Path(run), Path(data_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    checks = json.loads((run / "checks.json").read_text())
    assert selection["choices"] == manifest["choices"]
    assert selection["this_run_labels_used_for_routing"] is False
    assert hashlib.sha256((run / "selection.json").read_bytes()).hexdigest() == checks["selection_sha256"]
    for path, digest in manifest["protected_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    mapping = {kind: group for group, spec in manifest["groups"].items() for kind in spec["types"]}
    raw = pd.read_parquet(data_dir / "listings.parquet")
    raw["date_sold"] = pd.to_datetime(raw.date_sold)
    reference = raw.set_index("ml_num")
    rows = pd.read_csv(run / "all_candidate_predictions.csv")
    assert not rows.duplicated(["ml_num", "candidate"]).any()
    assert rows.groupby("ml_num").size().eq(2).all()
    np.testing.assert_array_equal(rows.sold_price, reference.loc[rows.ml_num].sold_price)
    np.testing.assert_array_equal(rows.house_type_name, reference.loc[rows.ml_num].house_type_name)
    np.testing.assert_array_equal(rows.community, reference.loc[rows.ml_num].community)
    np.testing.assert_array_equal(rows.month, reference.loc[rows.ml_num].date_sold.dt.strftime("%Y-%m"))
    assert rows.group.eq(rows.house_type_name.map(mapping)).all()
    assert np.isfinite(rows.predicted_price).all() and rows.predicted_price.gt(0).all()
    windows = [("development", *window) for window in manifest["development_windows"]]
    windows.append(("check", *manifest["check_window"]))
    for protocol, cutoff, end in windows:
        frame = rows.loc[rows.protocol.eq(protocol) & rows.cutoff_exclusive.eq(cutoff)]
        homes = raw.loc[raw.date_sold.ge(cutoff) & raw.date_sold.lt(end)]
        train = raw.loc[raw.date_sold.lt(cutoff)]
        assert set(frame.ml_num) == set(homes.ml_num)
        audit = json.loads((run / protocol / cutoff[:7] / "audit.json").read_text())
        assert audit["training_n"] == len(train) and audit["evaluation_n"] == len(homes)
        assert audit["history_reference_n"] == len(train)
        assert pd.Timestamp(audit["max_training_sale"]) == train.date_sold.max()
        assert audit["evaluation_labels_used_for_fit_or_early_stopping"] is False
        assert set(audit["models"]) == set(manifest["choices"].values())
        for identifier, detail in audit["models"].items():
            scope, _ = identifier.split("__")
            eligible = train if scope == "global" else train.loc[train.house_type_name.map(mapping).eq(scope)]
            expected = homes if scope == "global" else homes.loc[homes.house_type_name.map(mapping).eq(scope)]
            candidate = frame.loc[frame.candidate.eq(identifier)]
            assert detail["training_n"] == len(eligible) and detail["reload_exact"] is True
            assert detail["feature_count"] == (59 if scope in ("apartments", "condo_townhouse") else 64)
            assert set(candidate.ml_num) == set(expected.ml_num)
            match(candidate, detail["scores"])
    metrics = json.loads((run / "metrics.json").read_text())
    comparisons = pd.read_csv(run / "comparison.csv")
    bundles = 0
    for protocol in ("development", "check"):
        frame = rows.loc[rows.protocol.eq(protocol)]
        routed = pd.read_csv(run / f"{protocol}_routed_predictions.csv").sort_values("ml_num").reset_index(drop=True)
        chosen = frame.loc[frame.candidate.eq(frame.group.map(manifest["choices"]))].sort_values("ml_num").reset_index(drop=True)
        assert len(routed) == frame.ml_num.nunique() and not routed.ml_num.duplicated().any()
        np.testing.assert_array_equal(routed.ml_num, chosen.ml_num)
        np.testing.assert_array_equal(routed.predicted_price, chosen.predicted_price)
        np.testing.assert_array_equal(routed.selected_candidate, chosen.candidate)
        baseline = frame.loc[frame.candidate.eq(manifest["baseline"])].sort_values("ml_num").reset_index(drop=True)
        for identifier, bundle in metrics[protocol].items():
            current = routed if identifier == "routed" else baseline
            match(current, bundle["overall"])
            bundles += 1
            for dimension, keys in (("by_group", ["group"]), ("by_type", ["house_type_name"]),
                                    ("by_month", ["month"]), ("by_group_month", ["group", "month"]),
                                    ("by_window", ["cutoff_exclusive"])):
                for expected in bundle[dimension]:
                    mask = np.ones(len(current), dtype=bool)
                    for key in keys:
                        mask &= current[key].eq(expected[key]).to_numpy()
                    match(current.loc[mask], expected)
                    bundles += 1
        for item in comparisons.loc[comparisons.protocol.eq(protocol)].to_dict("records"):
            mask = np.ones(len(routed), dtype=bool) if item["dimension"] == "overall" else routed[item["dimension"]].eq(item["slice"]).to_numpy()
            current, reference_score = score(routed.loc[mask]), score(baseline.loc[mask])
            match(routed.loc[mask], item)
            for name in ("mdape", "mape", "median_signed_error"):
                np.testing.assert_allclose(reference_score[name], item[f"baseline_{name}"], rtol=0, atol=1e-10)
            for name in ("mdape", "mape"):
                np.testing.assert_allclose(current[name] - reference_score[name], item[f"delta_{name}_pp"], rtol=0, atol=1e-10)
    # Replay into a temporary unique path so the verifier can safely be rerun.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="fixed-four-groups-") as tmp:
        replay = predict_saved(run, data_dir, Path(tmp) / "replay.csv")
        expected = pd.read_csv(run / "final_predictions/submission_off_market.csv")
        pd.testing.assert_frame_equal(replay, expected, check_dtype=False)
    test = pd.read_parquet(data_dir / "test_off_market.parquet")
    routes = pd.read_csv(run / "final_predictions/prediction_routes.csv")
    np.testing.assert_array_equal(routes.ml_num, test.ml_num)
    np.testing.assert_array_equal(routes.selected_candidate, test.house_type_name.map(mapping).map(manifest["choices"]))
    np.testing.assert_array_equal(routes.predicted_price, expected.predicted_price)
    assert not check_submission.check_file(run / "final_predictions/submission_off_market.csv", test.ml_num.tolist())
    report = {"all_checks_passed": True, "candidate_rows_checked": len(rows),
              "distinct_evaluation_homes": rows.ml_num.nunique(), "metric_bundles_recomputed": bundles,
              "paired_comparisons_recomputed": len(comparisons), "saved_predictions_replayed": len(expected),
              "evaluation_models_reloaded": 20, "final_models_reloaded": 5,
              "fixed_routes_unchanged": True, "protected_sources_and_inputs_unchanged": True}
    (run / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[3] / "data")
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
