"""Replay the saved, frozen group choices without fitting or reading test labels."""
import argparse
import json
from pathlib import Path

import pandas as pd

import check_submission
from experiments.plan_a.features import HistoryFeatures
from .models import GroupModel, routed_predict


def predict_saved(run, data_dir, output):
    run, data_dir, output = Path(run), Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    state = json.loads((run / "final_models/routes.json").read_text())
    listings = pd.read_parquet(data_dir / "listings.parquet")
    homes = pd.read_parquet(data_dir / "test_off_market.parquet")
    x = HistoryFeatures(listings, state["cutoff_exclusive"]).transform(homes)
    models = {identifier: GroupModel.load_bundle(run / "final_models" / identifier)
              for identifier in set(state["choices"].values())}
    price, _ = routed_predict(x, homes, state["choices"], models)
    result = pd.DataFrame({"ml_num": homes.ml_num, "predicted_price": price})
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    errors = check_submission.check_file(output, homes.ml_num.tolist())
    if errors:
        raise ValueError(errors)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[3] / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = predict_saved(args.run, args.data_dir, args.output)
    print(f"Replayed {len(result):,} predictions: {args.output}")


if __name__ == "__main__":
    main()
