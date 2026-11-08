"""Recreate the candidate submission using saved preprocessing and weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.plan_a.features import HistoryFeatures
from .models import MLPPriceModel, sha
from .run_experiment import REPO, split_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.artifacts / "trained_model"
    metadata = json.loads((folder / "schema.json").read_text())["metadata"]
    for name, digest in metadata["input_sha256"].items():
        if sha(args.data_dir / name) != digest:
            raise ValueError(f"Input data changed: {name}")
    if sha(REPO / "experiments/plan_a/features.py") != metadata["feature_source_sha256"]:
        raise ValueError("Feature code changed")
    listings = pd.read_parquet(args.data_dir / "listings.parquet")
    test = pd.read_parquet(args.data_dir / "test_off_market.parquet")
    for c in ["date_sold", "date_listed"]:
        listings[c] = pd.to_datetime(listings[c])
    test["date_listed"] = pd.to_datetime(test.date_listed)
    cutoff = metadata["training_cutoff_exclusive"]
    train = split_window(listings, cutoff)
    history = HistoryFeatures(train, cutoff)
    x_test = history.transform(data.hide_answers(test, "off_market"))
    predicted = MLPPriceModel.load(folder).predict(x_test)
    archived = pd.read_csv(args.artifacts / "final_predictions/submission_off_market_mlp.csv")
    if not np.array_equal(archived.ml_num, test.ml_num) or not np.array_equal(archived.predicted_price, predicted):
        raise ValueError("Saved model does not reproduce archived predictions")
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "submission_off_market_mlp.csv"
    pd.DataFrame(dict(ml_num=test.ml_num, predicted_price=predicted)).to_csv(path, index=False)
    errors = check_submission.check_file(path, test.ml_num)
    if errors:
        raise ValueError(errors)
    print(f"Exact replay and format check passed: {len(test):,} rows; {path}")


if __name__ == "__main__":
    main()
