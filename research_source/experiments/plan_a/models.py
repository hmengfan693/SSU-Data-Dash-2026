"""[A3] Same feature definitions and log-MAE objective for both tree libraries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from .features import CATEGORICAL

PARAMS = {
    "catboost": dict(loss_function="MAE", iterations=600, depth=6,
                     learning_rate=0.05, l2_leaf_reg=5, random_seed=42,
                     thread_count=6, has_time=True, allow_writing_files=False,
                     verbose=False),
    "lightgbm": dict(objective="regression_l1", n_estimators=600,
                     num_leaves=31, max_depth=-1, learning_rate=0.05,
                     min_child_samples=40, reg_lambda=5, colsample_bytree=1.0,
                     subsample=1.0, random_state=42, n_jobs=6,
                     deterministic=True, force_col_wise=True, verbosity=-1),
}


class PriceModel:
    def __init__(self, name):
        self.name = name
        self.model = (CatBoostRegressor if name == "catboost" else LGBMRegressor)(**PARAMS[name])

    def encode(self, x):
        x = x[self.feature_names].copy()
        if self.name == "lightgbm":
            for c, values in self.categories.items():
                x[c] = pd.Categorical(x[c], categories=values)
        return x

    def fit(self, x, prices):
        self.feature_names = list(x.columns)
        self.categories = {c: sorted(x[c].unique().tolist()) for c in CATEGORICAL}
        # Fixed iterations: no early stopping on the four-month selection
        # labels. Categories are learned ONLY from this training fold.
        options = {"cat_features": CATEGORICAL} if self.name == "catboost" else {}
        self.model.fit(self.encode(x), np.log(np.asarray(prices)), **options)
        return self

    def predict(self, x):
        price = np.exp(self.model.predict(self.encode(x)))
        if not np.isfinite(price).all() or not (price > 0).all():
            raise ValueError("Predictions must be finite and positive")
        return np.round(price)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (self.name + (".cbm" if self.name == "catboost" else ".txt"))
        if self.name == "catboost":
            self.model.save_model(str(path))
        else:
            self.model.booster_.save_model(str(path))
        (folder / f"{self.name}_schema.json").write_text(json.dumps({
            "model": self.name, "params": PARAMS[self.name], "features": self.feature_names,
            "categories": self.categories, "target": "log(sold_price)",
            "history_cutoff_exclusive": "2026-06-01",
        }, ensure_ascii=False, indent=2) + "\n")
        return path
