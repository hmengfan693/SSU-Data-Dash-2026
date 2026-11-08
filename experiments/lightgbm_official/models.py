"""The official Dataset/train/Booster API, with baseline-equivalent parameters."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.plan_a.features import CATEGORICAL, FORBIDDEN

VERSION = '4.7.0'
ROUNDS = 600
# Explicitly translate sklearn wrapper defaults as well as the existing settings.
# No example-dataset parameters, validation labels, or early stopping are used.
PARAMS = dict(boosting='gbdt', objective='regression_l1', metric='l1',
              num_leaves=31, max_depth=-1, learning_rate=0.05,
              min_data_in_leaf=40, min_sum_hessian_in_leaf=0.001,
              min_gain_to_split=0., lambda_l1=0., lambda_l2=5.,
              feature_fraction=1., bagging_fraction=1., bagging_freq=0,
              bin_construct_sample_cnt=200000, seed=42, num_threads=6,
              deterministic=True, force_col_wise=True, verbosity=-1)


class OfficialLightGBM:
    def __init__(self, num_boost_round=ROUNDS, **overrides):
        if lgb.__version__ != VERSION:
            raise ValueError(f'Pinned official lightgbm=={VERSION} is required')
        self.params = dict(PARAMS, **overrides)
        self.num_boost_round = num_boost_round

    def encode(self, x):
        if FORBIDDEN.intersection(x.columns) or x.columns.duplicated().any():
            raise ValueError('Forbidden feature or duplicate feature names')
        x = x[self.feature_names].copy()
        for column, values in self.categories.items():
            x[column] = pd.Categorical(x[column], categories=values)
        return x

    def fit(self, x, prices):
        prices = np.asarray(prices, dtype=float)
        if (x.empty or prices.ndim != 1 or len(x) != len(prices)
                or not np.isfinite(prices).all() or not (prices > 0).all()):
            raise ValueError('Training prices must be aligned, finite and positive')
        self.feature_names = list(x)
        self.categories = {c: sorted(x[c].dropna().unique().tolist())
                           for c in CATEGORICAL if c in x}
        training = lgb.Dataset(self.encode(x), label=np.log(prices),
                               feature_name=self.feature_names,
                               categorical_feature=list(self.categories))
        self.booster = lgb.train(self.params, training,
                                 num_boost_round=self.num_boost_round)
        return self

    def predict(self, x):
        price = np.round(np.exp(self.booster.predict(self.encode(x),
                               num_threads=self.params['num_threads'])))
        if not np.isfinite(price).all() or not (price > 0).all():
            raise ValueError('Predictions must be finite and positive')
        return price

    def save(self, folder, **audit):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        self.booster.save_model(str(folder/'model.txt'))
        schema = dict(params=self.params, num_boost_round=self.num_boost_round,
                      features=self.feature_names, categories=self.categories,
                      target='log(sold_price)', lightgbm_version=VERSION,
                      api='lightgbm.Dataset -> lightgbm.train -> lightgbm.Booster',
                      unknown_category='missing', **audit)
        (folder/'schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2)+'\n')

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        schema = json.loads((folder/'schema.json').read_text())
        if schema['lightgbm_version'] != VERSION:
            raise ValueError('Saved model version differs from pinned version')
        result = cls(num_boost_round=schema['num_boost_round'], **schema['params'])
        result.feature_names, result.categories = schema['features'], schema['categories']
        result.booster = lgb.Booster(model_file=str(folder/'model.txt'))
        return result
