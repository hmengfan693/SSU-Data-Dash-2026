"""Prediction sensitivity when only exact-location category becomes unknown.

This intervention holds historical/geographic numeric features fixed. It is a
network mechanism probe, not a valid new housing dataset or deployable score.
"""
import pandas as pd
import numpy as np
import torch

from experiments.mlp.models import MLPPriceModel
from experiments.resnet.models import ResNetPriceModel
from .audit import OUT, RESNET, MLP, inference


def main():
    torch.set_num_threads(1)
    records = []
    for name, model in [("resnet", ResNetPriceModel.load(RESNET / "selection_A/2026-02/model")),
                        ("mlp", MLPPriceModel.load(MLP / "selection_A/2026-02/model"))]:
        for partition in ["train", "outer"]:
            x = pd.read_parquet(OUT / "cache" / f"{partition}_x.parquet")
            y = pd.read_parquet(OUT / "cache" / f"{partition}_labels.parquet")
            normal = inference(model, x, "mps")
            masked = inference(model, x.assign(location_group="__DIAGNOSTIC_UNKNOWN__"), "mps")
            known = x.location_group.isin(model.preprocessor.state["categories"]["location_group"])
            for typ, idx in y.groupby("house_type_name").groups.items():
                change = 100 * (masked[idx] / normal[idx] - 1)
                records.append(dict(model=name, partition=partition, property_type=typ,
                                    n=len(idx), known_pct=float(known.loc[idx].mean() * 100),
                                    median_prediction_change_pct=float(np.median(change)),
                                    mean_prediction_change_pct=float(np.mean(change))))
    result = pd.DataFrame(records)
    result.to_csv(OUT / "embedding_zero_intervention.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
