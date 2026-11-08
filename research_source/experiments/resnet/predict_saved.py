"""Replay a final ResNet candidate with saved preprocessing and weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.plan_a.features import HistoryFeatures
from experiments.mlp.run_experiment import split_window
from .models import ResNetPriceModel, sha
from .run_experiment import REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    folder = args.artifacts / "trained_model"
    metadata = json.loads((folder / "schema.json").read_text())["metadata"]
    for name, digest in metadata["input_sha256"].items():
        if sha(args.data_dir / name) != digest:
            raise ValueError(f"Data changed: {name}")
    if sha(REPO / "experiments/plan_a/features.py") != metadata["feature_source_sha256"]:
        raise ValueError("Feature source changed")
    listings = pd.read_parquet(args.data_dir / "listings.parquet")
    test = pd.read_parquet(args.data_dir / "test_off_market.parquet")
    for c in ["date_listed", "date_sold"]:
        listings[c] = pd.to_datetime(listings[c])
    test["date_listed"] = pd.to_datetime(test.date_listed)
    cutoff = metadata["training_cutoff_exclusive"]
    history = HistoryFeatures(split_window(listings, cutoff), cutoff)
    x = history.transform(data.hide_answers(test, "off_market"))
    prediction = ResNetPriceModel.load(folder, device=args.device).predict(x)
    archived = pd.read_csv(args.artifacts / "final_predictions/submission_off_market_resnet.csv")
    if not np.array_equal(test.ml_num, archived.ml_num):
        raise ValueError("Archived IDs differ")
    if not np.array_equal(prediction, archived.predicted_price):
        raise ValueError("Exact replay differs; use the training backend for exact reproduction")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "submission_off_market_resnet.csv"
    pd.DataFrame(dict(ml_num=test.ml_num, predicted_price=prediction)).to_csv(output, index=False)
    errors = check_submission.check_file(output, test.ml_num)
    if errors:
        raise ValueError(errors)
    print(f"Exact replay and submission check passed: {len(test):,} rows; {output}")


if __name__ == "__main__":
    main()
