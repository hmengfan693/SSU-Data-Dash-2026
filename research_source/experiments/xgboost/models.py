"""Native categorical XGBoost, log-L1, and complete inference persistence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from experiments.plan_a.features import CATEGORICAL, FORBIDDEN

PARAMS = dict(objective='reg:absoluteerror', booster='gbtree', tree_method='hist',
              device='cpu', grow_policy='depthwise', enable_categorical=True,
              n_estimators=600, learning_rate=0.05, max_depth=5,
              min_child_weight=20, reg_lambda=5., reg_alpha=0., gamma=0.,
              subsample=1., colsample_bytree=1., max_bin=256,
              max_cat_to_onehot=4, max_cat_threshold=64,
              random_state=42, n_jobs=6, verbosity=0)


class XGBoostPriceModel:
    def __init__(self, **overrides):
        self.params = dict(PARAMS, **overrides)

    def encode(self, x):
        if FORBIDDEN.intersection(x.columns):
            raise ValueError('Forbidden feature')
        x = x[self.feature_names].copy()
        # Fixed training vocabulary: unseen values become missing, never
        # independently factorize prediction rows or impose numeric ordering.
        for column, values in self.categories.items():
            x[column] = pd.Categorical(x[column], categories=values)
        return x

    def fit(self, x, prices):
        prices = np.asarray(prices, dtype=float)
        if (x.empty or len(x) != len(prices) or not np.isfinite(prices).all()
                or not (prices > 0).all()):
            raise ValueError('Training prices must be aligned, finite and positive')
        if FORBIDDEN.intersection(x.columns) or x.columns.duplicated().any():
            raise ValueError('Forbidden feature or duplicate feature names')
        self.feature_names = list(x)
        self.categories = {c: sorted(x[c].dropna().astype(str).unique().tolist())
                           for c in CATEGORICAL if c in x}
        y = np.log(prices)
        # An explicit training-only starting constant makes model replay and
        # version behavior inspectable. No validation/early-stop input exists.
        self.init_log_price = float(np.median(y))
        self.model = XGBRegressor(**self.params, base_score=self.init_log_price)
        self.model.fit(self.encode(x), y)
        return self

    def predict(self, x):
        # Float64 exponentiation follows the existing price pipeline even
        # though native XGBoost margins are float32.
        price = np.round(np.exp(np.asarray(self.model.predict(self.encode(x)), dtype=float)))
        if not np.isfinite(price).all() or not (price > 0).all():
            raise ValueError('Predictions must be finite and positive')
        return price

    def save(self, folder, **audit):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        self.model.save_model(folder/'model.ubj')
        schema = dict(params=self.params, features=self.feature_names,
                      categories=self.categories, target='log(sold_price)',
                      init_log_price=self.init_log_price,
                      unknown_category='missing', **audit)
        (folder/'schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2)+'\n')

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        schema = json.loads((folder/'schema.json').read_text())
        result = cls(**schema['params'])
        result.feature_names = schema['features']
        result.categories = schema['categories']
        result.init_log_price = schema['init_log_price']
        result.model = XGBRegressor()
        result.model.load_model(folder/'model.ubj')
        return result
