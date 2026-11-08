"""Pre-activation tabular ResNet with explicit CUDA/MPS execution.

Uses the existing MLP's train-only preprocessing and standardized log-price L1
objective. Architecture is inspired by Gorishniy et al. (NeurIPS 2021), with
LayerNorm, separate field embeddings, and a deeper/wider residual backbone.
"""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.mlp.models import Preprocessor, sha, write_json


DEFAULTS = dict(width=256, n_blocks=4, hidden_factor=2,
                embedding_dims=[4, 4, 4, 8, 8], hidden_dropout=0.1,
                residual_dropout=0.0, learning_rate=0.001, weight_decay=0.0001,
                batch_size=512, max_epochs=100, patience=10,
                min_delta_mdape=0.0001, seed=42, threads=1)


def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    available = requested == "cpu" or (requested == "cuda" and torch.cuda.is_available()) or (
        requested == "mps" and torch.backends.mps.is_available())
    if not available:
        raise RuntimeError(f"Requested GPU backend {requested} is unavailable; no silent CPU fallback")
    if requested == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK to require GPU execution")
    return torch.device(requested)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class ResidualBlock(nn.Module):
    def __init__(self, width, hidden, hidden_dropout, residual_dropout):
        super().__init__()
        self.branch = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                    nn.ReLU(), nn.Dropout(hidden_dropout),
                                    nn.Linear(hidden, width), nn.Dropout(residual_dropout))

    def forward(self, x):
        return x + self.branch(x)


class Network(nn.Module):
    def __init__(self, state, params):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(state["categories"][column]) + 1, dim, padding_idx=0)
            for column, dim in zip(state["categorical_columns"], params["embedding_dims"])])
        input_dim = state["numeric_output_dim"] + sum(params["embedding_dims"])
        width = params["width"]
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*[
            ResidualBlock(width, int(width * params["hidden_factor"]),
                          params["hidden_dropout"], params["residual_dropout"])
            for _ in range(params["n_blocks"])])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, numeric, categorical):
        values = [numeric] + [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.head(self.blocks(self.input(torch.cat(values, dim=1)))).squeeze(1)


class ResNetPriceModel:
    def __init__(self, *, device="auto", **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}")
        self.params = copy.deepcopy(DEFAULTS)
        self.params.update(params)
        self.device = resolve_device(device)
        for key in ["width", "n_blocks", "hidden_factor", "batch_size", "max_epochs", "patience", "threads"]:
            if self.params[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if len(self.params["embedding_dims"]) != 5 or any(d < 1 for d in self.params["embedding_dims"]):
            raise ValueError("Five positive embedding dimensions required")
        for key in ["hidden_dropout", "residual_dropout"]:
            if not 0 <= self.params[key] < 1:
                raise ValueError(f"Invalid {key}")

    def initialize(self):
        torch.set_num_threads(self.params["threads"])
        torch.manual_seed(self.params["seed"])
        # CUDA/CPU request deterministic kernels; MPS has no general guarantee.
        # Do not silently switch to CPU for unsupported operations.
        torch.use_deterministic_algorithms(self.device.type != "mps")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
        self.network = Network(self.preprocessor.state, self.params).to(self.device)

    @staticmethod
    def validate_prices(prices, n):
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
        torch.set_num_threads(self.params["threads"])
        values = self.validate_prices(prices, len(x))
        self.preprocessor = Preprocessor().fit(x)
        log_price = np.log(values)
        self.target_mean = float(log_price.mean())
        self.target_scale = max(float(log_price.std()), 1e-8)
        self.initialize()
        numeric, categorical = (t.to(self.device) for t in self.preprocessor.transform(x))
        target = torch.from_numpy(((log_price - self.target_mean) / self.target_scale).astype(np.float32)).to(self.device)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.params["learning_rate"],
                                      weight_decay=self.params["weight_decay"])
        generator = torch.Generator().manual_seed(self.params["seed"])
        valid_tensors, valid_prices = None, None
        if validation is not None:
            valid_x, valid_prices = validation
            if valid_x.empty:
                raise ValueError("Early stopping needs a nonempty temporal validation set")
            valid_prices = self.validate_prices(valid_prices, len(valid_x))
            valid_tensors = tuple(t.to(self.device) for t in self.preprocessor.transform(valid_x))
        best, best_state, stale = np.inf, None, 0
        self.training_history, self.best_epoch = [], limit
        self.execution = dict(device=str(self.device), parameter_device=str(next(self.network.parameters()).device),
                              numeric_device=str(numeric.device), categorical_device=str(categorical.device),
                              target_device=str(target.device), gpu_batches=0, cpu_batches=0,
                              cpu_fallback_enabled=False, deterministic_algorithms=self.device.type != "mps")
        for epoch in range(1, limit + 1):
            synchronize(self.device)
            started = time.monotonic()
            self.network.train()
            order = torch.randperm(len(x), generator=generator).to(self.device)
            # Accumulate on device: per-batch .item() would serialize GPU work.
            total = torch.zeros((), device=self.device)
            for idx in order.split(self.params["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                predicted = self.network(numeric[idx], categorical[idx])
                loss = nn.functional.l1_loss(predicted, target[idx])
                loss.backward()
                optimizer.step()
                total += loss.detach() * len(idx)
                self.execution["cpu_batches" if self.device.type == "cpu" else "gpu_batches"] += 1
            train_loss = float(total.cpu()) / len(x)
            if not np.isfinite(train_loss):
                raise ValueError("Nonfinite training loss")
            record = dict(epoch=epoch, train_log_mae=train_loss * self.target_scale)
            if valid_tensors is not None:
                prediction = self.predict_tensors(*valid_tensors)
                score = float(100 * np.median(np.abs(prediction / valid_prices - 1)))
                record["inner_mdape"] = score
                if score < best - self.params["min_delta_mdape"]:
                    best, stale, self.best_epoch = score, 0, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                else:
                    stale += 1
            synchronize(self.device)
            record["seconds"] = time.monotonic() - started
            self.training_history.append(record)
            if logger is not None:
                logger(record)
            if valid_tensors is not None and stale >= self.params["patience"]:
                break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        self.epochs_run = len(self.training_history)
        if self.device.type == "mps":
            self.execution["allocated_bytes_at_end"] = torch.mps.current_allocated_memory()
        return self

    def predict_tensors(self, numeric, categorical):
        self.network.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(numeric), 4096):
                outputs.append(self.network(numeric[start:start + 4096].to(self.device),
                                            categorical[start:start + 4096].to(self.device)).cpu().numpy())
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
        torch.save({k: v.detach().cpu() for k, v in self.network.state_dict().items()}, folder / "weights.pt")
        write_json(folder / "schema.json", dict(
            format_version=1, architecture="tabular_resnet", params=self.params,
            preprocessing=self.preprocessor.state, target_mean=self.target_mean,
            target_scale=self.target_scale, best_epoch=self.best_epoch, epochs_run=self.epochs_run,
            execution=self.execution, parameter_count=sum(p.numel() for p in self.network.parameters()),
            weights_sha256=sha(folder / "weights.pt"), torch_version=torch.__version__,
            metadata={} if metadata is None else metadata))
        write_json(folder / "training_history.json", self.training_history)

    @classmethod
    def load(cls, folder, *, device="auto"):
        folder = Path(folder)
        s = json.loads((folder / "schema.json").read_text())
        if (s["format_version"] != 1 or s["architecture"] != "tabular_resnet"
                or sha(folder / "weights.pt") != s["weights_sha256"]):
            raise ValueError("Unsupported or corrupted ResNet checkpoint")
        model = cls(device=device, **s["params"])
        model.preprocessor = Preprocessor()
        model.preprocessor.state = s["preprocessing"]
        model.target_mean, model.target_scale = s["target_mean"], s["target_scale"]
        model.best_epoch, model.epochs_run = s["best_epoch"], s["epochs_run"]
        model.training_history = json.loads((folder / "training_history.json").read_text())
        model.execution = s["execution"]
        model.initialize()
        model.network.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        model.network.eval()
        return model
