"""Independent NumPy scoring and routing verification, plus saved inference replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .predict_saved import predict_saved


def score(frame):
    error = 100. * (frame.predicted_price.to_numpy() - frame.sold_price.to_numpy()) / frame.sold_price.to_numpy()
    return {"n": len(frame), "mdape": float(np.median(np.abs(error))),
            "mape": float(np.mean(np.abs(error))), "median_signed_error": float(np.median(error)),
            "over_30_pct": float(np.mean(np.abs(error) > 30) * 100)}


def match(actual, expected):
    for key, value in score(actual).items():
        if not np.isclose(value, expected[key], rtol=0, atol=1e-10):
            raise AssertionError((key, value, expected[key]))


def verify(run, data_dir):
    run, data_dir = Path(run), Path(data_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    choices = selection["choices"]
    groups = manifest["groups"]
    mapping = {kind: group for group, item in groups.items() for kind in item["types"]}
    listings = pd.read_parquet(data_dir / "listings.parquet")
    listings["date_sold"] = pd.to_datetime(listings.date_sold)
    raw = listings.set_index("ml_num")
    rows = pd.read_csv(run / "all_candidate_predictions.csv")
    if rows.duplicated(["ml_num", "candidate"]).any():
        raise AssertionError("Duplicate candidate predictions")
    assert rows.groupby("ml_num").size().eq(4).all()
    assert rows.group.eq(rows.house_type_name.map(mapping)).all()
    reference = raw.loc[rows.ml_num]
    np.testing.assert_array_equal(rows.sold_price, reference.sold_price)
    np.testing.assert_array_equal(rows.house_type_name, reference.house_type_name)
    np.testing.assert_array_equal(rows.month, reference.date_sold.dt.strftime("%Y-%m"))
    assert np.isfinite(rows.predicted_price).all() and (rows.predicted_price > 0).all()
    windows = [("development", *window) for window in manifest["development_windows"]]
    windows.append(("check", *manifest["check_window"]))
    for protocol, cutoff, end in windows:
        frame = rows.loc[(rows.protocol == protocol) & (rows.cutoff_exclusive == cutoff)]
        expected = listings.loc[(listings.date_sold >= cutoff) & (listings.date_sold < end)]
        assert set(frame.ml_num) == set(expected.ml_num)
        audit = json.loads((run / protocol / cutoff[:7] / "audit.json").read_text())
        train = listings.loc[listings.date_sold < cutoff]
        assert audit["training_n"] == len(train)
        assert pd.Timestamp(audit["max_training_sale"]) == train.date_sold.max()
        assert audit["evaluation_labels_used_for_fit_or_early_stopping"] is False
        for identifier, item in audit["models"].items():
            scope, algorithm = identifier.split("__")
            eligible_train = train if scope == "global" else train.loc[train.house_type_name.map(mapping).eq(scope)]
            eligible_valid = expected if scope == "global" else expected.loc[expected.house_type_name.map(mapping).eq(scope)]
            candidate = frame.loc[frame.candidate == identifier]
            assert item["training_n"] == len(eligible_train)
            assert set(candidate.ml_num) == set(eligible_valid.ml_num)
            assert item["feature_count"] == (59 if scope in ("apartments", "condo_townhouse") else 64)
            assert item["reload_exact"] is True
            match(candidate, item["scores"])
    development = rows.loc[rows.protocol == "development"]
    for group in groups:
        candidate_scores = {identifier: score(frame) for identifier, frame in development.loc[
            development.group == group].groupby("candidate")}
        best = min(candidate_scores, key=lambda i: (candidate_scores[i]["mdape"], candidate_scores[i]["mape"], i))
        assert choices[group] == best
        for identifier, frame in development.loc[development.group == group].groupby("candidate"):
            match(frame, selection["scores"]["by_group"][group][identifier])
    global_scores = {identifier: score(frame) for identifier, frame in development.loc[
        development.scope == "global"].groupby("candidate")}
    assert choices["__default__"] == min(global_scores, key=lambda i: (
        global_scores[i]["mdape"], global_scores[i]["mape"], i))
    metrics = json.loads((run / "metrics.json").read_text())
    bundles_checked = 0
    for protocol in ("development", "check"):
        frame = rows.loc[rows.protocol == protocol]
        routed = pd.read_csv(run / f"{protocol}_routed_predictions.csv")
        assert len(routed) == frame.ml_num.nunique() and not routed.ml_num.duplicated().any()
        routed = routed.sort_values("ml_num").reset_index(drop=True)
        chosen = frame.loc[frame.candidate.eq(frame.group.map(choices))].sort_values("ml_num").reset_index(drop=True)
        np.testing.assert_array_equal(routed.ml_num, chosen.ml_num)
        np.testing.assert_array_equal(routed.predicted_price, chosen.predicted_price)
        np.testing.assert_array_equal(routed.selected_candidate, chosen.candidate)
        for identifier, bundle in metrics[protocol].items():
            result = routed if identifier == "routed" else frame.loc[frame.candidate == identifier]
            match(result, bundle["overall"])
            bundles_checked += 1
            for dimension, keys in (("by_group", ["group"]), ("by_type", ["house_type_name"]),
                ("by_month", ["month"]), ("by_group_month", ["group", "month"]),
                ("by_window", ["cutoff_exclusive"])):
                for item in bundle[dimension]:
                    mask = np.ones(len(result), dtype=bool)
                    for key in keys:
                        mask &= result[key].eq(item[key]).to_numpy()
                    match(result.loc[mask], item)
                    bundles_checked += 1
    comparison = pd.read_csv(run / "group_comparison.csv")
    for row in comparison.to_dict("records"):
        match(rows.loc[(rows.protocol == row["protocol"]) & (rows.group == row["group"])
                       & (rows.candidate == row["candidate"])], row)
        assert row["selected"] == (choices[row["group"]] == row["candidate"])
    checks = json.loads((run / "checks.json").read_text())
    assert hashlib.sha256((run / "selection.json").read_bytes()).hexdigest() == checks["selection_sha256"]
    for path, digest in manifest["protected_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    archive = Path(__file__).resolve().parents[1] / "plan_a/artifacts/run_2026-10-16_v1"
    baseline_reproductions = []
    for protocol, cutoff, _ in windows:
        old_protocol = "development_A" if protocol == "development" else "selection_A"
        for name in ("lightgbm", "catboost"):
            old_path = archive / old_protocol / cutoff[:7] / f"{name}_predictions.csv"
            if old_path.exists():
                old = pd.read_csv(old_path).sort_values("ml_num")
                current = rows.loc[(rows.protocol == protocol) & (rows.cutoff_exclusive == cutoff)
                                   & (rows.candidate == f"global__{name}")].sort_values("ml_num")
                np.testing.assert_array_equal(old.ml_num, current.ml_num)
                np.testing.assert_array_equal(old.predicted_price, current.predicted_price)
                baseline_reproductions.append(f"{protocol}/{cutoff}/{name}")
    replay_path = run / "verification_replay.csv"
    replay = predict_saved(run, data_dir, replay_path)
    expected = pd.read_csv(run / "final_predictions/submission_off_market.csv")
    pd.testing.assert_frame_equal(replay, expected, check_dtype=False)
    report = {"all_checks_passed": True, "candidate_rows_checked": len(rows),
        "distinct_validation_homes": rows.ml_num.nunique(), "metric_bundles_recomputed": bundles_checked,
        "group_candidate_scores_recomputed": len(comparison),
        "saved_final_predictions_reproduced": len(replay), "frozen_selection_recomputed": True,
        "original_global_baselines_exactly_reproduced": baseline_reproductions,
        "protected_sources_and_inputs_unchanged": True}
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
