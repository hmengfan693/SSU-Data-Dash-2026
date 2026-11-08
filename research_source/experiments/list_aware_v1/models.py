"""Deterministic LightGBM ratio models with train-only category vocabularies."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.plan_a.features import CATEGORICAL
from experiments.lightgbm_official.models import PARAMS
from .features import FORBIDDEN, list_prices

GROUPS = {'apartments': ['Condo Apt', 'Co-Op Apt'],
          'condo_townhouse': ['Condo Townhouse'], 'detached': ['Detached'],
          'attached_freehold': ['Semi-Detached', 'Freehold Townhouse']}
TYPE_TO_GROUP = {kind: group for group, kinds in GROUPS.items() for kind in kinds}
ROUNDS = 600


class RatioModel:
    def __init__(self, threads=6):
        self.params = dict(PARAMS, num_threads=threads)

    def encode(self, x):
        if FORBIDDEN.intersection(x.columns) or x.columns.duplicated().any():
            raise ValueError('Invalid model inputs')
        result = x[self.features].copy()
        for c, values in self.categories.items():
            result[c] = pd.Categorical(result[c], categories=values)
        return result

    def fit(self, x, homes):
        sold = homes.sold_price.to_numpy(float)
        if len(x) != len(homes) or not np.isfinite(sold).all() or not (sold > 0).all():
            raise ValueError('Invalid or misaligned training prices')
        self.features = list(x)
        self.categories = {c: sorted(x[c].dropna().unique().tolist()) for c in CATEGORICAL if c in x}
        training = lgb.Dataset(self.encode(x), label=np.log(sold/list_prices(homes)),
                               categorical_feature=list(self.categories))
        self.booster = lgb.train(self.params, training, num_boost_round=ROUNDS)
        return self

    def predict(self, x, homes):
        if len(x) != len(homes):
            raise ValueError('Prediction rows differ')
        correction = self.booster.predict(self.encode(x), num_threads=self.params['num_threads'])
        result = np.round(list_prices(homes)*np.exp(correction))
        if not np.isfinite(result).all() or not (result > 0).all():
            raise ValueError('Invalid predictions')
        return result

    def save(self, folder, **audit):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        self.booster.save_model(str(folder/'model.txt'))
        (folder/'schema.json').write_text(json.dumps(dict(features=self.features,
            categories=self.categories, params=self.params, num_boost_round=ROUNDS,
            target='log(sold_price/list_price)', lightgbm_version=lgb.__version__, **audit),
            ensure_ascii=False, indent=2)+'\n')
        pd.DataFrame({'feature': self.features,
                      'gain': self.booster.feature_importance(importance_type='gain')}).sort_values(
                          'gain', ascending=False).to_csv(folder/'importance.csv', index=False)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        schema = json.loads((folder/'schema.json').read_text())
        if schema['lightgbm_version'] != lgb.__version__:
            raise ValueError('LightGBM version mismatch')
        result = cls(schema['params']['num_threads'])
        result.params = schema['params']
        result.features, result.categories = schema['features'], schema['categories']
        result.booster = lgb.Booster(model_file=str(folder/'model.txt'))
        return result


def grouped_predict(models, x, homes):
    groups = homes.house_type_name.map(TYPE_TO_GROUP).to_numpy()
    if pd.isna(groups).any():
        raise ValueError('Unrecognized property type')
    result = np.full(len(homes), np.nan)
    for group in GROUPS:
        ix = np.flatnonzero(groups == group)
        if len(ix):
            result[ix] = models[group].predict(x.iloc[ix], homes.iloc[ix])
    return result
