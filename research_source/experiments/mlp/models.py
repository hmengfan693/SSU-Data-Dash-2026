"""Train-only preprocessing, categorical embeddings and standardized log-MAE.

CPU inference, fixed seeds and deterministic algorithms support same-runtime
reproduction. No guarantee of bitwise reproduction across PyTorch/platforms.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.plan_a.features import CATEGORICAL, FORBIDDEN

DEFAULTS = dict(hidden_sizes=[128, 64, 32], embedding_dims=[4, 4, 4, 8, 8],
                dropout=0.1, learning_rate=0.001, weight_decay=0.0001,
                batch_size=512, max_epochs=100, patience=10,
                min_delta_mdape=0.0001, seed=42, threads=1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


class Preprocessor:
    """All fitted state is JSON-serializable and learned from fit rows only."""

    @staticmethod
    def numeric(x, columns, log_columns):
        frame = x[columns].apply(pd.to_numeric, errors="coerce").astype(float)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        for column in log_columns:
            frame[column] = np.log1p(frame[column].where(frame[column] >= 0))
        return frame

    def fit(self, x):
        if x.empty or not x.columns.is_unique or FORBIDDEN.intersection(x.columns):
            raise ValueError("Expected nonempty, unique, allowlisted feature columns")
        if not set(CATEGORICAL).issubset(x.columns):
            raise ValueError("Missing categorical features")
        columns = [c for c in x if c not in CATEGORICAL]
        area = {"area_low", "area_high", "area_mid", "lot_front", "lot_depth", "lot_area"}
        log_columns = [c for c in columns if c in area or (c.startswith("hist_") and c.endswith("_n"))]
        frame = self.numeric(x, columns, log_columns)
        missing = [c for c in columns if frame[c].isna().any()]
        binary = [c for c in columns if not frame[c].dropna().empty
                  and frame[c].dropna().isin([0., 1.]).all()]
        medians = frame.median().fillna(0.)
        filled = frame.fillna(medians)
        mean, scale = filled.mean(), filled.std(ddof=0)
        mean.loc[binary], scale.loc[binary] = 0., 1.
        scale = scale.where(scale > 1e-12, 1.)
        categories = {c: sorted(x[c].fillna("__MISSING__").astype(str).unique().tolist())
                      for c in CATEGORICAL}
        self.state = dict(feature_names=list(x.columns), numeric_columns=columns,
                          categorical_columns=list(CATEGORICAL), categories=categories,
                          log1p_columns=log_columns, binary_columns=binary,
                          missing_columns=missing, medians=medians.to_dict(),
                          means=mean.to_dict(), scales=scale.to_dict(),
                          unknown_category_id=0,
                          numeric_output_dim=len(columns) + len(missing))
        return self

    def transform(self, x):
        s = self.state
        if set(x.columns) != set(s["feature_names"]) or not x.columns.is_unique:
            raise ValueError("Prediction feature schema differs from training")
        frame = self.numeric(x, s["numeric_columns"], s["log1p_columns"])
        missing = frame[s["missing_columns"]].isna().to_numpy(dtype=np.float32)
        filled = frame.fillna(s["medians"])
        numeric = ((filled - pd.Series(s["means"])) / pd.Series(s["scales"]))[s["numeric_columns"]]
        numeric = np.concatenate([numeric.to_numpy(dtype=np.float32), missing], axis=1)
        categorical = []
        for column in s["categorical_columns"]:
            vocabulary = {value: i + 1 for i, value in enumerate(s["categories"][column])}
            ids = x[column].fillna("__MISSING__").astype(str).map(vocabulary).fillna(0)
            categorical.append(ids.to_numpy(dtype=np.int64))
        if not np.isfinite(numeric).all():
            raise ValueError("Preprocessing produced nonfinite inputs")
        return torch.from_numpy(numeric), torch.from_numpy(np.stack(categorical, axis=1))


class Network(nn.Module):
    def __init__(self, state, params):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(state["categories"][column]) + 1, width, padding_idx=0)
            for column, width in zip(state["categorical_columns"], params["embedding_dims"])
        ])
        # UNK stays zero: never rely on an untrained random embedding at inference.
        width = state["numeric_output_dim"] + sum(params["embedding_dims"])
        layers = []
        for hidden in params["hidden_sizes"]:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(params["dropout"])])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, numeric, categorical):
        parts = [numeric] + [embedding(categorical[:, i])
                             for i, embedding in enumerate(self.embeddings)]
        return self.layers(torch.cat(parts, dim=1)).squeeze(1)


class MLPPriceModel:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown MLP parameters: {unknown}")
        self.params = copy.deepcopy(DEFAULTS)
        self.params.update(params)
        if len(self.params["embedding_dims"]) != len(CATEGORICAL):
            raise ValueError("One embedding dimension is required per categorical field")
        for name in ["batch_size", "max_epochs", "patience", "threads"]:
            if self.params[name] < 1:
                raise ValueError(f"{name} must be positive")

    def initialize(self):
        torch.set_num_threads(self.params["threads"])
        torch.manual_seed(self.params["seed"])
        torch.use_deterministic_algorithms(True)
        self.network = Network(self.preprocessor.state, self.params)

    @staticmethod
    def prices(prices, n):
        values = np.asarray(prices, dtype=np.float64)
        if values.shape != (n,) or not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError("Targets must be aligned, finite and positive")
        return values

    def fit(self, x, prices, *, epochs=None, validation=None, logger=None):
        if epochs is not None and validation is not None:
            raise ValueError("Fixed refit epochs and early stopping are mutually exclusive")
        limit = self.params["max_epochs"] if epochs is None else int(epochs)
        if limit < 1:
            raise ValueError("epochs must be positive")
        values = self.prices(prices, len(x))
        self.preprocessor = Preprocessor().fit(x)
        numeric, categorical = self.preprocessor.transform(x)
        log_price = np.log(values)
        self.target_mean = float(log_price.mean())
        self.target_scale = max(float(log_price.std()), 1e-8)
        self.initialize()
        # Configure CPU threads BEFORE the first tensor copy/parallel operation.
        # A large torch.tensor() before set_num_threads can initialize competing
        # OpenMP pools when the archived tree libraries are also imported.
        target = torch.from_numpy(np.asarray((log_price - self.target_mean) / self.target_scale,
                                            dtype=np.float32))
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.params["learning_rate"],
                                      weight_decay=self.params["weight_decay"])
        loss_fn = nn.L1Loss()
        generator = torch.Generator().manual_seed(self.params["seed"])
        valid_tensors, valid_prices = None, None
        if validation is not None:
            valid_x, valid_prices = validation
            if valid_x.empty:
                raise ValueError("Early stopping needs a nonempty temporal validation set")
            valid_prices = self.prices(valid_prices, len(valid_x))
            valid_tensors = self.preprocessor.transform(valid_x)
        best, best_state, stale = np.inf, None, 0
        self.training_history, self.best_epoch = [], limit
        for epoch in range(1, limit + 1):
            self.network.train()
            order = torch.randperm(len(x), generator=generator)
            total = 0.
            for idx in order.split(self.params["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                predicted = self.network(numeric[idx], categorical[idx])
                loss = loss_fn(predicted, target[idx])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(idx)
            record = dict(epoch=epoch, train_log_mae=total / len(x) * self.target_scale)
            if valid_tensors is not None:
                predicted = self.predict_tensors(*valid_tensors)
                score = float(100 * np.median(np.abs(predicted / valid_prices - 1)))
                record["inner_mdape"] = score
                if score < best - self.params["min_delta_mdape"]:
                    best, stale, self.best_epoch = score, 0, epoch
                    best_state = copy.deepcopy(self.network.state_dict())
                else:
                    stale += 1
            self.training_history.append(record)
            if logger is not None:
                logger(record)
            if valid_tensors is not None and stale >= self.params["patience"]:
                break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        self.epochs_run = len(self.training_history)
        return self

    def predict_tensors(self, numeric, categorical):
        self.network.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(numeric), 4096):
                outputs.append(self.network(numeric[start:start + 4096], categorical[start:start + 4096]).numpy())
        if not outputs:
            return np.empty(0, dtype=np.float64)
        z = np.concatenate(outputs).astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            price = np.exp(self.target_mean + self.target_scale * z)
        if not np.isfinite(price).all() or not (price > 0).all():
            raise ValueError("Predicted prices must be finite and positive")
        return price

    def predict(self, x):
        price = np.round(self.predict_tensors(*self.preprocessor.transform(x)))
        if not (price > 0).all():
            raise ValueError("Rounded predictions must be positive")
        return price

    def save(self, folder, metadata=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "weights.pt"
        torch.save(self.network.state_dict(), path)
        write_json(folder / "schema.json", dict(
            format_version=1, params=self.params, preprocessing=self.preprocessor.state,
            target_mean=self.target_mean, target_scale=self.target_scale,
            best_epoch=self.best_epoch, epochs_run=self.epochs_run,
            weights_sha256=sha(path), torch_version=torch.__version__,
            metadata={} if metadata is None else metadata))
        write_json(folder / "training_history.json", self.training_history)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        schema = json.loads((folder / "schema.json").read_text())
        if schema["format_version"] != 1 or sha(folder / "weights.pt") != schema["weights_sha256"]:
            raise ValueError("Unsupported or corrupted MLP checkpoint")
        model = cls(**schema["params"])
        model.preprocessor = Preprocessor()
        model.preprocessor.state = schema["preprocessing"]
        model.target_mean, model.target_scale = schema["target_mean"], schema["target_scale"]
        model.best_epoch, model.epochs_run = schema["best_epoch"], schema["epochs_run"]
        model.training_history = json.loads((folder / "training_history.json").read_text())
        model.initialize()
        model.network.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        model.network.eval()
        return model
