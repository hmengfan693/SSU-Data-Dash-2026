"""Reuse off-market features and add strictly past sold-to-list information."""
import numpy as np
import pandas as pd

from experiments.plan_a.features import HistoryFeatures, group_keys, property_features
from experiments.data_processing_round1.features import ProcessingFeatures, assemble, text_features

LEVELS = [('market', []), ('type', ['house_type_name']),
          ('community_type', ['community', 'house_type_name'])]
WINDOWS = (90, 180, 365)
FORBIDDEN = {'sold_price', 'date_sold', 'days_on_market', 'external_avm_estimate',
             'external_avm_estimate_date', 'ml_num', 'list_price'}


def list_prices(homes):
    prices = pd.to_numeric(homes['list_price'], errors='coerce').to_numpy(float)
    if not np.isfinite(prices).all() or not (prices > 0).all():
        raise ValueError('List prices must be finite and positive')
    return prices


class RatioHistory:
    """Frozen pool; every query excludes same-day and future sales, including self.

    Features are hierarchical smoothed means of log(sold/list). Their exponentials
    are geometric ratios, not medians. Empty market history defaults to ratio 1.
    """
    def __init__(self, reference, cutoff):
        self.cutoff = pd.Timestamp(cutoff)
        ref = reference.loc[pd.to_datetime(reference.date_sold) < self.cutoff].reset_index(drop=True)
        if ref.empty or (pd.to_datetime(ref.date_sold) < pd.to_datetime(ref.date_listed)).any():
            raise ValueError('Require nonempty, chronologically valid reference sales')
        sold = ref.sold_price.to_numpy(float)
        if not np.isfinite(sold).all() or not (sold > 0).all():
            raise ValueError('Invalid historical sold price')
        ratio = np.log(sold / list_prices(ref))
        dates = pd.to_datetime(ref.date_sold).to_numpy(dtype='datetime64[ns]')
        self.tables = {}
        for name, columns in LEVELS:
            keys = group_keys(ref, columns)
            self.tables[name] = {}
            for key, positions in keys.groupby(keys, sort=False).groups.items():
                ix = np.asarray(positions)
                ix = ix[np.argsort(dates[ix], kind='stable')]
                self.tables[name][key] = (dates[ix], np.r_[0., np.cumsum(ratio[ix])])

    def transform(self, homes):
        h = homes.reset_index(drop=True)
        listed = pd.to_datetime(h.date_listed).to_numpy(dtype='datetime64[ns]')
        if np.isnat(listed).any():
            raise ValueError('Listing dates required')
        query = np.minimum(listed, self.cutoff.to_datetime64())
        x = pd.DataFrame(index=h.index)
        for name, columns in LEVELS:
            keys = group_keys(h, columns)
            for days in WINDOWS:
                n, total = np.zeros(len(h)), np.zeros(len(h))
                age = np.full(len(h), np.nan)
                for key, positions in keys.groupby(keys, sort=False).groups.items():
                    history = self.tables[name].get(key)
                    if history is None:
                        continue
                    dates, prefix = history
                    ix = np.asarray(positions)
                    right = np.searchsorted(dates, query[ix], side='left')
                    left = np.searchsorted(dates, query[ix]-np.timedelta64(days, 'D'), side='left')
                    n[ix] = right-left
                    total[ix] = prefix[right]-prefix[left]
                    good = right > left
                    age[ix[good]] = (query[ix[good]]-dates[right[good]-1])/np.timedelta64(1, 'D')
                parent = np.zeros(len(h)) if name == 'market' else x[
                    f'ratio_{"market" if name == "type" else "type"}_{days}_log_mean'].to_numpy()
                strength = 0. if name == 'market' else (20. if name == 'type' else 30.)
                denom = n+strength
                mean = np.divide(total+strength*parent, denom, out=parent.copy(), where=denom > 0)
                prefix = f'ratio_{name}_{days}'
                x[prefix+'_log_mean'], x[prefix+'_n'], x[prefix+'_age_days'] = mean, n, age
            x[f'ratio_{name}_trend_90_vs_365'] = x[f'ratio_{name}_90_log_mean']-x[f'ratio_{name}_365_log_mean']
        return x


class FeatureBuilder:
    def __init__(self, reference, cutoff):
        self.base = HistoryFeatures(reference, cutoff)
        self.processing = ProcessingFeatures(reference, cutoff)
        self.ratios = RatioHistory(reference, cutoff)

    def transform(self, homes):
        # Select exactly the test-time fields. Labels and unavailable fields never
        # reach any query feature, even if the caller passes a training table.
        from experiments.plan_a.features import INPUT_COLUMNS
        h = homes[[*INPUT_COLUMNS, 'list_price']].reset_index(drop=True)
        price = list_prices(h)
        x = assemble(self.base.transform(h), self.processing.history(h),
                     self.processing.land(h), text_features(h), 'all_three')
        ratios = self.ratios.transform(h)
        x = pd.concat([x, ratios], axis=1)
        x['la_log_list_price'] = np.log(price)
        x['la_log_list_ppsf'] = np.log(price)-x.log_area
        for days in WINDOWS:
            x[f'la_list_vs_community_price_{days}'] = np.log(price)-x[f'hist_community_type_{days}_log_price']
            x[f'la_list_ppsf_vs_community_{days}'] = x.la_log_list_ppsf-x[f'hist_community_type_{days}_log_ppsf']
        if FORBIDDEN.intersection(x.columns) or x.columns.duplicated().any():
            raise ValueError('Forbidden or duplicated feature')
        return x.replace([np.inf, -np.inf], np.nan)


def historical_prediction(homes, ratio_features):
    return np.round(list_prices(homes)*np.exp(ratio_features['ratio_community_type_180_log_mean'].to_numpy()))
