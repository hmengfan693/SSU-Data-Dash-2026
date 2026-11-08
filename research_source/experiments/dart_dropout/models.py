"""Keep the baseline objective and capacity; vary tree dropout only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMRegressor

from experiments.plan_a.features import CATEGORICAL, FORBIDDEN
from experiments.plan_a.models import PARAMS

DROP_SEEDS = (42, 43, 44)
PRIMARY_SEED = 42
DROPOUT = dict(boosting_type='dart', drop_rate=0.01, skip_drop=0.0,
               uniform_drop=True, max_drop=-1, xgboost_dart_mode=False)


def parameters(drop_seed=None):
    params = dict(PARAMS['lightgbm'])
    params['boosting_type'] = 'gbdt'
    if drop_seed is not None:
        params.update(DROPOUT, drop_seed=int(drop_seed))
    return params


class DropoutModel:
    def __init__(self, drop_seed=None):
        self.params = parameters(drop_seed)
        self.model = LGBMRegressor(**self.params)

    def encode(self, x):
        x = x[self.feature_names].copy()
        for column, values in self.categories.items():
            x[column] = pd.Categorical(x[column], categories=values)
        return x

    def fit(self, x, prices):
        prices = np.asarray(prices, dtype=float)
        if len(x) != len(prices) or not np.isfinite(prices).all() or not (prices > 0).all():
            raise ValueError('Training prices must be finite, positive and aligned')
        if FORBIDDEN.intersection(x.columns):
            raise ValueError('Forbidden feature')
        self.feature_names = list(x)
        self.categories = {c: sorted(x[c].unique().tolist()) for c in CATEGORICAL if c in x}
        # Keep the native L1 starting constant outside the dropped ensemble.
        # Otherwise dropping the first, bias-carrying tree can leave a constant
        # gradient, stop tree growth and leave dropped trees unnormalised in
        # the installed DART implementation. A constant-only native fit uses
        # exactly the baseline library's label precision and median convention.
        y = np.log(prices)
        initial = LGBMRegressor(objective='regression_l1', n_estimators=1,
                               num_leaves=2, n_jobs=1, verbosity=-1)
        initial.fit(np.zeros((len(y), 1)), y)
        self.init_log_price = float(initial.predict(np.zeros((1, 1)))[0])
        # No evaluation labels or early stopping callbacks are accepted.
        self.model.fit(self.encode(x), y, init_score=np.full(len(y), self.init_log_price))
        return self

    def predict(self, x):
        price = np.round(np.exp(self.init_log_price + self.model.predict(self.encode(x))))
        if not np.isfinite(price).all() or not (price > 0).all():
            raise ValueError('Invalid prediction')
        return price

    @property
    def booster(self):
        return self.model if isinstance(self.model, Booster) else self.model.booster_

    def save(self, folder, **audit):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        self.booster.save_model(str(folder/'model.txt'))
        schema = dict(params=self.params, features=self.feature_names, categories=self.categories,
                      target='log(sold_price)', init_log_price=self.init_log_price,
                      initial_constant_outside_dropout=True, **audit)
        (folder/'schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2)+'\n')

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        schema = json.loads((folder/'schema.json').read_text())
        model = cls.__new__(cls)
        model.params = schema['params']
        model.feature_names = schema['features']
        model.categories = schema['categories']
        model.init_log_price = schema['init_log_price']
        model.model = Booster(model_file=str(folder/'model.txt'))
        return model
