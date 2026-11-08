"""Reload saved final models and reproduce predictions without retraining."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import Booster

import check_submission
from .features import HistoryFeatures
from .models import PriceModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Run is not complete")
    for name in ["listings.parquet", "test_off_market.parquet"]:
        if sha(args.data_dir / name) != manifest["inputs_sha256"][name]:
            raise ValueError(f"Data fingerprint mismatch: {name}")
    if sha(Path(__file__).with_name("features.py")) != manifest["code_sha256"]["features.py"]:
        raise ValueError("Feature definitions changed since training")
    train = pd.read_parquet(args.data_dir / "listings.parquet")
    test = pd.read_parquet(args.data_dir / "test_off_market.parquet")
    train = train.sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    x = HistoryFeatures(train, "2026-06-01").transform(test)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, extension in [("catboost", ".cbm"), ("lightgbm", ".txt")]:
        path = args.artifacts / "trained_models" / (name + extension)
        if sha(path) != manifest["trained_models_sha256"][path.name]:
            raise ValueError("Saved model fingerprint mismatch")
        schema = json.loads((path.parent / f"{name}_schema.json").read_text())
        model = PriceModel(name)
        model.feature_names, model.categories = schema["features"], schema["categories"]
        if name == "catboost":
            model.model = CatBoostRegressor()
            model.model.load_model(str(path))
        else:
            model.model = Booster(model_file=str(path))
        output = args.output / f"submission_off_market_{name}.csv"
        pd.DataFrame({"ml_num": test.ml_num, "predicted_price": model.predict(x)}).to_csv(output, index=False)
        assert not check_submission.check_file(output, test.ml_num.tolist())
        reference = args.artifacts / "final_predictions" / output.name
        if sha(output) != sha(reference):
            raise ValueError(f"Reloaded predictions differ: {name}")
        print(f"{name}: saved model reload reproduced all {len(test):,} predictions byte-for-byte")


if __name__ == "__main__":
    main()
