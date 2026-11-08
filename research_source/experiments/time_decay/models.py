"""Training-only time weights, with the plan-A features and parameters unchanged."""
import numpy as np
import pandas as pd

from experiments.plan_a.features import CATEGORICAL
from experiments.plan_a.models import PriceModel

MONTH_DAYS = 365.25 / 12


def time_weights(sale_dates, cutoff, half_life_months=12.0):
    """Return mean-one weights; cutoff is exclusive and never uses test labels."""
    if not np.isfinite(half_life_months) or half_life_months <= 0:
        raise ValueError("Half-life must be finite and positive")
    dates = pd.DatetimeIndex(pd.to_datetime(sale_dates))
    boundary = pd.Timestamp(cutoff)
    if not len(dates) or dates.isna().any() or (dates >= boundary).any():
        raise ValueError("Every training sale must have a date strictly before cutoff")
    age_months = np.asarray((boundary - dates).total_seconds()) / (86400 * MONTH_DAYS)
    raw = np.exp2(-age_months / half_life_months)
    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("Invalid or underflowed training weights")
    return raw / raw.mean()


class WeightedPriceModel(PriceModel):
    def fit(self, x, prices, sample_weight=None):
        if sample_weight is None:
            return super().fit(x, prices)
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(x),) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("One finite positive weight is required per training row")
        self.feature_names = list(x.columns)
        self.categories = {c: sorted(x[c].unique().tolist()) for c in CATEGORICAL}
        options = {"cat_features": CATEGORICAL} if self.name == "catboost" else {}
        self.model.fit(self.encode(x), np.log(np.asarray(prices)),
                       sample_weight=weights, **options)
        return self
