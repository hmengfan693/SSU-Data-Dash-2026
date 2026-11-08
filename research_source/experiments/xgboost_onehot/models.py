"""Only house_type_name changes; keep all other features and parameters."""
from pathlib import Path

import pandas as pd

from experiments.xgboost.models import XGBoostPriceModel

TYPE_COLUMN = 'house_type_name'
PREFIX = 'house_type_ohe__'


class OneHotPriceModel(XGBoostPriceModel):
    def encode(self, x):
        encoded = super().encode(x)
        if TYPE_COLUMN not in self.categories:
            raise ValueError('house_type_name required')
        kinds = encoded.pop(TYPE_COLUMN)
        # Dense, explicit 0/1 values: an absent sparse element is missing to
        # XGBoost trees, which would change the meaning of a valid zero.
        for index, value in enumerate(self.categories[TYPE_COLUMN]):
            name = f'{PREFIX}{index}'
            if name in encoded:
                raise ValueError('One-hot feature name collision')
            encoded[name] = kinds.eq(value).astype('float32')
        return encoded

    def save(self, folder, **audit):
        encoded = [c for c in self.feature_names if c != TYPE_COLUMN]
        encoded += [f'{PREFIX}{i}' for i in range(len(self.categories[TYPE_COLUMN]))]
        super().save(folder, house_type_encoding='dense_full_one_hot',
                     encoded_features=encoded, unknown_house_type='all_zero', **audit)

    @classmethod
    def load(cls, folder):
        import json
        schema = json.loads((Path(folder)/'schema.json').read_text())
        if schema.get('house_type_encoding') != 'dense_full_one_hot':
            raise ValueError('Expected a house-type one-hot model')
        result = super().load(folder)
        if schema['encoded_features'] != result.model.get_booster().feature_names:
            raise ValueError('Persisted encoded feature schema mismatch')
        return result
