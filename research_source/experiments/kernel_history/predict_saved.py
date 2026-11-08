"""Replay the final kernel candidate, rebuilding history from original data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.plan_a.run_experiment import sha
from .features import KernelHistoryFeatures
from .run_experiment import REPO, load_model, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, schema = load_model(args.artifacts / "trained_model")
    for filename, digest in schema["inputs_sha256"].items():
        if sha(args.data_dir / filename) != digest:
            raise ValueError(f"Input changed: {filename}")
    for filename, digest in schema["source_sha256"].items():
        if sha(REPO / filename) != digest:
            raise ValueError(f"Feature/model source changed: {filename}")
    listings = pd.read_parquet(args.data_dir / "listings.parquet")
    test = pd.read_parquet(args.data_dir / "test_off_market.parquet")
    for column in ["date_listed", "date_sold"]:
        listings[column] = pd.to_datetime(listings[column])
    test["date_listed"] = pd.to_datetime(test.date_listed)
    cutoff = schema["cutoff_exclusive"]
    history = KernelHistoryFeatures(split(listings, cutoff), cutoff, **schema["kernel_config"])
    features = history.transform(data.hide_answers(test, "off_market"))
    prices = model.predict(features)
    archive = pd.read_csv(args.artifacts / "final_predictions/submission_off_market_kernel.csv")
    if not np.array_equal(archive.ml_num, test.ml_num) or not np.array_equal(archive.predicted_price, prices):
        raise ValueError("Final checkpoint replay differs")
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "submission_off_market_kernel.csv"
    pd.DataFrame(dict(ml_num=test.ml_num, predicted_price=prices)).to_csv(path, index=False)
    if check_submission.check_file(path, test.ml_num):
        raise ValueError("Invalid candidate submission")
    print(f"Exact replay and format check passed: {len(test):,} rows; {path}")


if __name__ == "__main__":
    main()
