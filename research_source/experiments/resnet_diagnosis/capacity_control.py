"""One additional control: bring the dense backbone close to MLP capacity.

Prespecified from parameter counts, before looking at new outer scores.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.mlp.models import sha, write_json
from .ablations import DiagnosticModel, save_snapshot, load_snapshot
from .audit import OUT, metrics

SPEC = dict(name="resnet_capacity_matched", kind="resnet",
            overrides=dict(width=64, n_blocks=2))


def main():
    folder = OUT / "ablations" / SPEC["name"]
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "prespecified.json", dict(spec=SPEC, source_sha256=sha(Path(__file__)),
               rationale="Original ResNet dense backbone is 43x MLP; this control is ~1.6x, embeddings unchanged",
               outer_labels_used_for_selection=False))
    cache = OUT / "cache"
    xi = pd.read_parquet(cache / "inner_train_x.parquet")
    xv = pd.read_parquet(cache / "inner_valid_x.parquet")
    yi = pd.read_parquet(cache / "inner_train_labels.parquet")
    yv = pd.read_parquet(cache / "inner_valid_labels.parquet")
    model = DiagnosticModel(SPEC)
    tensors, groups = [], []

    def log(r):
        if not tensors:
            tensors.extend(t.to(model.device) for t in model.preprocessor.transform(xv))
        pred = model.predict_tensors(*tensors)
        scores = []
        for typ, idx in yv.groupby("house_type_name").groups.items():
            s = metrics(pred[idx], yv.sold_price.to_numpy()[idx])
            groups.append(dict(name=SPEC["name"], epoch=r["epoch"], group=typ, **s))
            if len(idx) >= 30:
                scores.append(s["mdape"])
        r["inner_macro_mdape_n30"] = float(np.mean(scores))
        r["inner_log_mae"] = metrics(pred, yv.sold_price)["log_mae"]
        if r["epoch"] == 1 or r["epoch"] % 5 == 0:
            print(f"epoch {r['epoch']}: inner MdAPE={r['inner_mdape']:.3f}%", flush=True)

    started = time.monotonic()
    model.fit(xi, yi.sold_price, validation=(xv, yv.sold_price), logger=log)
    epochs, epochs_run = model.best_epoch, model.epochs_run
    pred = model.predict_tensors(*tensors)
    score = metrics(pred, yv.sold_price)
    save_snapshot(model, folder / "inner_model")
    pd.DataFrame(model.training_history).to_csv(folder / "learning_curve.csv", index=False)
    pd.DataFrame(dict(ml_num=yv.ml_num, sold_price=yv.sold_price, house_type_name=yv.house_type_name,
                      predicted_price=pred)).to_csv(folder / "inner_predictions.csv", index=False)
    assert np.array_equal(np.round(pred), load_snapshot(folder / "inner_model").predict(xv))
    xt = pd.read_parquet(cache / "train_x.parquet")
    yt = pd.read_parquet(cache / "train_labels.parquet")
    xo = pd.read_parquet(cache / "outer_x.parquet")
    model = DiagnosticModel(SPEC).fit(xt, yt.sold_price, epochs=epochs)
    pred = model.predict(xo)
    save_snapshot(model, folder / "refit_model")
    assert np.array_equal(pred, load_snapshot(folder / "refit_model").predict(xo))
    train_score = metrics(model.predict_tensors(*model.preprocessor.transform(xt)), yt.sold_price)
    yo = pd.read_parquet(cache / "outer_labels.parquet")
    pd.DataFrame(dict(ml_num=yo.ml_num, sold_price=yo.sold_price, house_type_name=yo.house_type_name,
                      predicted_price=pred)).to_csv(folder / "outer_predictions.csv", index=False)
    pd.DataFrame(groups).to_csv(folder / "inner_epoch_by_property_type.csv", index=False)
    record = dict(name=SPEC["name"], selected_epochs=epochs, epochs_run_inner=epochs_run,
                  inner=score, refit_train=train_score, outer=metrics(pred, yo.sold_price),
                  execution=model.execution, seconds=time.monotonic() - started)
    write_json(folder / "result.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
