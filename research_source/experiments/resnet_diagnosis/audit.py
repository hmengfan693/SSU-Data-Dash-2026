"""Rebuild main-window inputs and diagnose saved models without retraining."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data
from experiments.mlp.models import MLPPriceModel, sha, write_json
from experiments.mlp.run_experiment import split_window
from experiments.plan_a.features import HistoryFeatures
from experiments.resnet.models import ResNetPriceModel

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "artifacts/run_2026-10-19_v1"
RESNET = REPO / "experiments/resnet/artifacts/run_2026-10-19_v1"
MLP = REPO / "experiments/mlp/artifacts/run_2026-10-17_v2"


def metrics(price, actual):
    error = 100 * (np.asarray(price) / np.asarray(actual) - 1)
    return dict(n=len(error), mdape=float(np.median(abs(error))), mape=float(np.mean(abs(error))),
                bias=float(np.median(error)), log_mae=float(np.mean(abs(np.log(price) - np.log(actual)))),
                over30=float(np.mean(abs(error) > 30) * 100))


def inference(model, x, device):
    model.network.to(device).eval()
    numeric, categorical = model.preprocessor.transform(x)
    z = []
    with torch.inference_mode():
        for start in range(0, len(x), 4096):
            z.append(model.network(numeric[start:start + 4096].to(device),
                                   categorical[start:start + 4096].to(device)).cpu().numpy())
    return np.exp(model.target_mean + model.target_scale * np.concatenate(z).astype(float))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    cache = OUT / "cache"
    cache.mkdir()
    torch.set_num_threads(1)
    listings = pd.read_parquet(REPO.parent / "data/listings.parquet")
    for c in ["date_sold", "date_listed"]:
        listings[c] = pd.to_datetime(listings[c])
    train, outer = split_window(listings, "2026-02-01", "2026-06-01")
    earlier, inner = split_window(train, "2025-12-01", "2026-02-01")
    h = HistoryFeatures(train, "2026-02-01")
    ih = HistoryFeatures(earlier, "2025-12-01")
    frames = dict(train=h.transform(train), outer=h.transform(data.hide_answers(outer, "off_market")),
                  inner_train=ih.transform(earlier), inner_valid=ih.transform(data.hide_answers(inner, "off_market")))
    labels = dict(train=train, outer=outer, inner_train=earlier, inner_valid=inner)
    for key, x in frames.items():
        x.to_parquet(cache / f"{key}_x.parquet", index=False)
        labels[key][["ml_num", "sold_price", "date_sold", "house_type_name", "community"]].to_parquet(cache / f"{key}_labels.parquet", index=False)
    cache_hashes = {p.name: sha(p) for p in cache.iterdir()}
    write_json(OUT / "cache_hashes.json", cache_hashes)
    models = dict(resnet=ResNetPriceModel.load(RESNET / "selection_A/2026-02/model", device="mps"),
                  mlp=MLPPriceModel.load(MLP / "selection_A/2026-02/model"))
    summary, rows, parity = {}, [], {}
    for name, model in models.items():
        summary[name] = {}
        predictions = {}
        for key in ["train", "outer"]:
            predictions[key] = inference(model, frames[key], "mps")
            summary[name][key] = metrics(predictions[key], labels[key].sold_price)
        p = predictions["outer"]
        cpu = inference(model, frames["outer"], "cpu")
        parity[name] = dict(max_relative_difference=float(np.max(abs(cpu / p - 1))),
                            mean_absolute_dollar_difference=float(np.mean(abs(cpu - p))),
                            cpu_mdape=metrics(cpu, outer.sold_price)["mdape"],
                            gpu_mdape=summary[name]["outer"]["mdape"])
        for key in ["train", "outer"]:
            x, y = frames[key], labels[key]
            freq = x.location_group.map(frames["train"].location_group.value_counts()).fillna(0).astype(int)
            bins = pd.cut(freq, [-1, 0, 1, 4, 19, np.inf], labels=["unseen", "1", "2-4", "5-19", "20+"])
            price_bins = pd.cut(y.sold_price, [0, 5e5, 1e6, 1.5e6, 2e6, np.inf], right=False,
                                labels=["<500k", "500k-1m", "1m-1.5m", "1.5m-2m", ">=2m"])
            for dimension, groups in [("property_type", y.house_type_name), ("location_frequency", bins),
                                       ("price_band", price_bins), ("month", y.date_sold.dt.strftime("%Y-%m"))]:
                for group in groups.dropna().unique():
                    mask = np.asarray(groups == group)
                    rows.append(dict(model=name, partition=key, dimension=dimension, group=str(group),
                                     **metrics(predictions[key][mask], y.sold_price.to_numpy()[mask])))
    pd.DataFrame(rows).to_csv(OUT / "saved_model_groups.csv", index=False)
    write_json(OUT / "saved_model_summary.json", summary)
    write_json(OUT / "cpu_gpu_inference_parity.json", parity)
    freq = frames["outer"].location_group.map(frames["train"].location_group.value_counts()).fillna(0)
    coverage = []
    for typ, positions in outer.groupby("house_type_name").groups.items():
        f = freq.loc[positions]
        coverage.append(dict(property_type=typ, n=len(positions), unseen_pct=float((f == 0).mean() * 100),
                             low_frequency_0_to_4_pct=float((f <= 4).mean() * 100),
                             median_training_location_count=float(f.median())))
    pd.DataFrame(coverage).to_csv(OUT / "location_coverage.csv", index=False)
    # MAPE is additive across samples; MdAPE is not. Attribute excess mean APE.
    r = pd.read_csv(RESNET / "selection_A/2026-02/resnet_predictions.csv").set_index("ml_num")
    m = pd.read_csv(MLP / "selection_A/2026-02/mlp_predictions.csv").set_index("ml_num").loc[r.index]
    assert np.array_equal(r.sold_price, m.sold_price)
    diff = (abs(r.predicted_price / r.sold_price - 1) - abs(m.predicted_price / m.sold_price - 1)) * 100
    contribution = [dict(property_type=typ, n=len(g), mean_ape_difference=float(diff.loc[g.index].mean()),
                         overall_mape_contribution=float(diff.loc[g.index].sum() / len(r)))
                    for typ, g in r.groupby("house_type_name")]
    pd.DataFrame(contribution).to_csv(OUT / "mape_contribution.csv", index=False)
    print(json.dumps(dict(saved_models=summary, inference_parity=parity, location_coverage=coverage,
                          mape_contribution=contribution), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
