"""Prespecified GPU controls; all epoch selection uses Dec-Jan inner labels only.

Outer Feb-May metrics are examined once after every configuration has been
frozen. This is diagnosis on an already-used development window, not a new test.
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.mlp.models import DEFAULTS as MLP_DEFAULTS, Network as MLPNetwork, Preprocessor, sha, write_json
from experiments.resnet.models import ResNetPriceModel
from .audit import OUT, RESNET, MLP, metrics


SPECS = [
    dict(name="resnet_repeat42", kind="resnet", overrides={}),
    dict(name="resnet_seed43", kind="resnet", overrides=dict(seed=43)),
    dict(name="resnet_two_blocks", kind="resnet", overrides=dict(n_blocks=2)),
    dict(name="resnet_lower_lr", kind="resnet", overrides=dict(learning_rate=0.0003)),
    dict(name="resnet_no_location", kind="resnet", overrides={}, no_location=True),
    dict(name="resnet_location_dropout", kind="resnet", overrides={}, location_dropout=0.5),
    dict(name="resnet_batchnorm", kind="resnet", overrides={}, normalization="batchnorm"),
    dict(name="mlp_gpu", kind="mlp", overrides={}),
]


class DiagnosticModel(ResNetPriceModel):
    def __init__(self, spec, device="mps"):
        self.spec = spec
        super().__init__(device=device, **spec["overrides"])

    def initialize(self):
        super().initialize()
        if self.spec["kind"] == "mlp":
            torch.manual_seed(self.params["seed"])
            self.network = MLPNetwork(self.preprocessor.state, MLP_DEFAULTS).to(self.device)
        if self.spec.get("normalization") == "batchnorm":
            for block in self.network.blocks:
                block.branch[0] = nn.BatchNorm1d(self.params["width"]).to(self.device)
            self.network.head[0] = nn.BatchNorm1d(self.params["width"]).to(self.device)
        if self.spec.get("no_location"):
            # Keep vocabulary, initialization and all other parameters identical.
            # The one removed signal is the exact-location category embedding.
            with torch.no_grad():
                self.network.embeddings[4].weight.zero_()
            self.network.embeddings[4].weight.requires_grad_(False)
        if self.spec.get("location_dropout"):
            p = self.spec["location_dropout"]

            def mask_location(module, inputs, output):
                if module.training:
                    keep = (torch.rand((len(output), 1), device=output.device) >= p).to(output.dtype)
                    return output * keep
                return output

            self.network.embeddings[4].register_forward_hook(mask_location)


def save_snapshot(model, folder):
    folder.mkdir(parents=True, exist_ok=False)
    torch.save({k: v.detach().cpu() for k, v in model.network.state_dict().items()}, folder / "weights.pt")
    write_json(folder / "schema.json", dict(
        format_version=1, diagnostic_spec=model.spec, trainer_params=model.params,
        preprocessing=model.preprocessor.state, target_mean=model.target_mean, target_scale=model.target_scale,
        best_epoch=model.best_epoch, epochs_run=model.epochs_run, execution=model.execution,
        parameter_count=sum(p.numel() for p in model.network.parameters()),
        trainable_parameter_count=sum(p.numel() for p in model.network.parameters() if p.requires_grad),
        weights_sha256=sha(folder / "weights.pt")))
    write_json(folder / "training_history.json", model.training_history)


def load_snapshot(folder, device="mps"):
    s = json.loads((folder / "schema.json").read_text())
    if sha(folder / "weights.pt") != s["weights_sha256"]:
        raise ValueError("Corrupted diagnostic weights")
    m = DiagnosticModel(s["diagnostic_spec"], device=device)
    m.preprocessor = Preprocessor()
    m.preprocessor.state = s["preprocessing"]
    m.target_mean, m.target_scale = s["target_mean"], s["target_scale"]
    m.initialize()
    m.network.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
    m.network.eval()
    return m


def release():
    gc.collect()
    torch.mps.empty_cache()


def main():
    target = OUT / "ablations"
    target.mkdir(parents=True, exist_ok=False)
    cache = OUT / "cache"
    hashes = json.loads((OUT / "cache_hashes.json").read_text())
    if not all(sha(cache / name) == digest for name, digest in hashes.items()):
        raise ValueError("Cached input changed")
    sources = [Path(__file__), Path(__file__).with_name("audit.py")]
    protected = {str(p): sha(p) for root in [RESNET, MLP] for p in root.rglob("*") if p.is_file()}
    write_json(target / "manifest.json", dict(
        specs=SPECS, source_sha256={p.name: sha(p) for p in sources}, protected_sha256=protected,
        inner_train_cutoff_exclusive="2025-12-01", inner_valid_end_exclusive="2026-02-01",
        outer_end_exclusive="2026-06-01", selection_metric="inner unrounded MdAPE",
        outer_labels_used_for_selection=False, final_test_labels_used=False, device="mps"))
    for p in sources:
        (target / p.name).write_bytes(p.read_bytes())
    x_inner = pd.read_parquet(cache / "inner_train_x.parquet")
    x_valid = pd.read_parquet(cache / "inner_valid_x.parquet")
    y_inner = pd.read_parquet(cache / "inner_train_labels.parquet")
    y_valid = pd.read_parquet(cache / "inner_valid_labels.parquet")
    x_train = pd.read_parquet(cache / "train_x.parquet")
    y_train = pd.read_parquet(cache / "train_labels.parquet")
    # Do not load outer labels until all configurations have selected epochs.
    x_outer = pd.read_parquet(cache / "outer_x.parquet")
    records, epoch_records = [], []
    for spec in SPECS:
        name = spec["name"]
        folder = target / name
        folder.mkdir()
        started = time.monotonic()
        print(f"[{name}] inner selection on GPU", flush=True)
        model = DiagnosticModel(spec)
        valid_tensors = []

        def log(record):
            if not valid_tensors:
                valid_tensors.extend(t.to(model.device) for t in model.preprocessor.transform(x_valid))
            prediction = model.predict_tensors(*valid_tensors)
            groups = []
            for typ, idx in y_valid.groupby("house_type_name").groups.items():
                s = metrics(prediction[idx], y_valid.sold_price.to_numpy()[idx])
                epoch_records.append(dict(name=name, epoch=record["epoch"], group=typ, **s))
                if len(idx) >= 30:
                    groups.append(s["mdape"])
            record["inner_log_mae"] = metrics(prediction, y_valid.sold_price)["log_mae"]
            record["inner_macro_mdape_n30"] = float(np.mean(groups))
            if record["epoch"] == 1 or record["epoch"] % 5 == 0:
                print(f"  epoch {record['epoch']}: train={record['train_log_mae']:.4f}, "
                      f"inner MdAPE={record['inner_mdape']:.3f}%, macro={record['inner_macro_mdape_n30']:.3f}%", flush=True)

        model.fit(x_inner, y_inner.sold_price, validation=(x_valid, y_valid.sold_price), logger=log)
        epochs = model.best_epoch
        inner_prediction = model.predict_tensors(*valid_tensors)
        inner_score = metrics(inner_prediction, y_valid.sold_price)
        save_snapshot(model, folder / "inner_model")
        pd.DataFrame(model.training_history).to_csv(folder / "learning_curve.csv", index=False)
        pd.DataFrame(dict(ml_num=y_valid.ml_num, sold_price=y_valid.sold_price,
                          house_type_name=y_valid.house_type_name, predicted_price=inner_prediction)).to_csv(folder / "inner_predictions.csv", index=False)
        if not np.array_equal(np.round(inner_prediction), load_snapshot(folder / "inner_model").predict(x_valid)):
            raise ValueError("Inner snapshot replay differs")
        print(f"  selected {epochs} epochs; inner MdAPE={inner_score['mdape']:.3f}%", flush=True)
        del model, valid_tensors
        release()
        model = DiagnosticModel(spec).fit(x_train, y_train.sold_price, epochs=epochs)
        prediction = model.predict(x_outer)
        save_snapshot(model, folder / "refit_model")
        if not np.array_equal(prediction, load_snapshot(folder / "refit_model").predict(x_outer)):
            raise ValueError("Outer snapshot replay differs")
        pd.DataFrame(dict(predicted_price=prediction)).to_csv(folder / "outer_predictions_unscored.csv", index=False)
        train_pred = model.predict_tensors(*model.preprocessor.transform(x_train))
        train_score = metrics(train_pred, y_train.sold_price)
        records.append(dict(name=name, selected_epochs=epochs, epochs_run_inner=model.params["max_epochs"],
                            inner=inner_score, refit_train=train_score, seconds=time.monotonic() - started,
                            execution=model.execution))
        # Correct the recorded inner epoch count from its saved checkpoint.
        records[-1]["epochs_run_inner"] = json.loads((folder / "inner_model/schema.json").read_text())["epochs_run"]
        write_json(target / "progress.json", records)
        print(f"  refit finished ({records[-1]['seconds']:.1f}s); outer predictions saved without scoring", flush=True)
        del model
        release()
    # All epoch decisions are now frozen. Score the existing development window.
    y_outer = pd.read_parquet(cache / "outer_labels.parquet")
    groups = []
    for record in records:
        folder = target / record["name"]
        prediction = pd.read_csv(folder / "outer_predictions_unscored.csv").predicted_price.to_numpy()
        record["outer"] = metrics(prediction, y_outer.sold_price)
        for typ, idx in y_outer.groupby("house_type_name").groups.items():
            groups.append(dict(name=record["name"], group=typ,
                               **metrics(prediction[idx], y_outer.sold_price.to_numpy()[idx])))
        pd.DataFrame(dict(ml_num=y_outer.ml_num, sold_price=y_outer.sold_price,
                          house_type_name=y_outer.house_type_name, predicted_price=prediction)).to_csv(folder / "outer_predictions.csv", index=False)
    write_json(target / "results.json", records)
    pd.DataFrame(groups).to_csv(target / "outer_by_property_type.csv", index=False)
    pd.DataFrame(epoch_records).to_csv(target / "inner_epoch_by_property_type.csv", index=False)
    if not all(sha(p) == h for p, h in protected.items()):
        raise ValueError("Original artifact changed")
    if not all(sha(p) == json.loads((target / "manifest.json").read_text())["source_sha256"][p.name] for p in sources):
        raise ValueError("Running source changed")
    write_json(target / "checks.json", dict(model_replays_exact=2 * len(SPECS), all_fits_mps=True,
               protected_files_unchanged=True, protected_file_count=len(protected),
               configs_selected_before_outer_scoring=True, source_unchanged=True))
    print(json.dumps([{k: v for k, v in r.items() if k != "execution"} for r in records], indent=2), flush=True)


if __name__ == "__main__":
    main()
