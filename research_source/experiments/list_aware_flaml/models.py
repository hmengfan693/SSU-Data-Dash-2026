"""Same seven-parameter search as off-market, with the list-aware ratio target."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.flaml_tuning.models import BASE_CONFIG, SEARCH_SPEC, canonical_config, search_space
from experiments.list_aware_v1.models import RatioModel
from experiments.list_aware_v1.features import list_prices
from experiments.plan_a.features import CATEGORICAL


def validate_config(config):
    if set(config) != set(BASE_CONFIG):
        raise ValueError('Exactly the seven predeclared parameters are required')
    config = canonical_config(config)
    for key, (low, high) in SEARCH_SPEC.items():
        if not np.isfinite(config[key]) or not low <= config[key] <= high:
            raise ValueError(f'Out-of-range parameter: {key}')
    return config


class TunedRatioModel(RatioModel):
    def __init__(self, config=None, threads=6):
        super().__init__(threads)
        self.config = validate_config(BASE_CONFIG if config is None else config)
        c = self.config
        self.rounds = c['n_estimators']
        self.params.update(num_leaves=c['num_leaves'], learning_rate=c['learning_rate'],
                           min_data_in_leaf=c['min_child_samples'], lambda_l1=c['reg_alpha'],
                           lambda_l2=c['reg_lambda'], feature_fraction=c['colsample_bytree'])

    def fit(self, x, homes):
        sold = homes.sold_price.to_numpy(float)
        if x.empty or len(x) != len(homes) or not np.isfinite(sold).all() or not (sold > 0).all():
            raise ValueError('Invalid training rows/prices')
        self.features = list(x)
        self.categories = {c: sorted(x[c].dropna().unique().tolist()) for c in CATEGORICAL if c in x}
        train = lgb.Dataset(self.encode(x), label=np.log(sold/list_prices(homes)),
                            categorical_feature=list(self.categories))
        self.booster = lgb.train(self.params, train, num_boost_round=self.rounds)
        return self

    def save(self, folder, **audit):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        self.booster.save_model(str(folder/'model.txt'))
        schema = dict(config=self.config, features=self.features, categories=self.categories,
                      params=self.params, num_boost_round=self.rounds, target='log(sold_price/list_price)',
                      lightgbm_version=lgb.__version__, **audit)
        (folder/'schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2)+'\n')
        pd.DataFrame(dict(feature=self.features, gain=self.booster.feature_importance('gain'))).sort_values(
            'gain', ascending=False).to_csv(folder/'importance.csv', index=False)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        s = json.loads((folder/'schema.json').read_text())
        if s['lightgbm_version'] != lgb.__version__:
            raise ValueError('Model library version mismatch')
        result = cls(s['config'], s['params']['num_threads'])
        if result.rounds != s['num_boost_round'] or result.params != s['params']:
            raise ValueError('Saved config and parameters disagree')
        result.features, result.categories = s['features'], s['categories']
        result.booster = lgb.Booster(model_file=str(folder/'model.txt'))
        return result
