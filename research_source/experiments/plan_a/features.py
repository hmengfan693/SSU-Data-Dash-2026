"""[A2] Allowlisted property features and strictly earlier historical statistics.

Each historical query uses [query_date - lookback, query_date), where
query_date = min(date_listed, fixed_training_cutoff). No validation label enters
the reference pool. Prefix sums make these point-in-time queries inexpensive.
"""
import re

import numpy as np
import pandas as pd

CATEGORICAL = ["house_type_name", "house_style", "municipality", "community", "location_group"]
NUMERIC = ["bedrooms", "washrooms", "parking_total", "garage", "lot_front", "lot_depth", "latitude", "longitude"]
INPUT_COLUMNS = ["house_area", "date_listed", "remarks", *NUMERIC, *CATEGORICAL[:-1]]
FORBIDDEN = {"sold_price", "list_price", "date_sold", "days_on_market", "external_avm_estimate", "external_avm_estimate_date", "ml_num"}


def property_features(homes):
    """Only named test-time fields can influence model input."""
    h = homes[INPUT_COLUMNS].reset_index(drop=True)
    x = h[NUMERIC].apply(pd.to_numeric, errors="coerce").copy()
    for c in CATEGORICAL[:-1]:
        x[c] = h[c].fillna("__MISSING__").astype(str)
    x["location_group"] = (h.latitude.round(5).astype(str) + "|" + h.longitude.round(5).astype(str))
    area = h.house_area.astype(str).str.extract(r"^\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*$").astype(float)
    x["area_low"] = area[0]
    x["area_high"] = area[1].fillna(area[0])
    x["area_mid"] = (x.area_low + x.area_high) / 2
    x["area_is_range"] = area[1].notna().astype(int)
    x["log_area"] = np.log(x.area_mid.where(x.area_mid > 0))
    x["lot_area"] = x.lot_front * x.lot_depth
    x["lot_missing"] = (x.lot_front.isna() | x.lot_depth.isna()).astype(int)
    x["log_lot_area"] = np.log1p(x.lot_area.where(x.lot_area >= 0))
    date = pd.to_datetime(h.date_listed)
    x["listed_days_since_2021"] = (date - pd.Timestamp("2021-01-01")).dt.days
    x["listed_month"] = date.dt.month
    x["month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
    x["month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)
    # No prices or arbitrary numbers are extracted from remarks. These are
    # lexical indicators, not verified condition labels. Negated positives
    # are removed before matching (e.g. "not recently renovated").
    text = h.remarks.fillna("").str.lower()
    text = text.str.replace(r"\b(?:not|never|no)\s+(?:(?:\w+)\s+){0,3}(?:renovated|updated|remodelled|remodeled|redone)\b", "", regex=True)
    patterns = {
        "condition_full_renovation": r"fully (?:renovated|redone|remodelled|remodeled)|top to bottom|gut renovat|complete(?:ly)? renovat",
        "condition_update": r"renovat|updated|refreshed|new (?:kitchen|bathroom|flooring)|fresh paint|quartz|hardwood",
        "condition_needs_work": r"needs? (?:work|updat|replac|renovat)|fixer|handyman|as[ -]is|original finishes|dated|project for",
        "condition_ready": r"turnkey|move[ -]in ready|nothing to do|nothing to .{0,12}unpack",
        "feature_basement": r"finished basement|basement apartment|separate entrance",
        "feature_view": r"unobstructed|ravine|waterfront|lake view",
    }
    for name, pattern in patterns.items():
        x[name] = text.str.contains(pattern, regex=True, flags=re.I).astype(int)
    assert not FORBIDDEN.intersection(x.columns)
    return x.replace([np.inf, -np.inf], np.nan)


def group_keys(x, columns):
    if not columns:
        return pd.Series("__ALL__", index=x.index)
    return x[columns].astype(str).agg("\x1f".join, axis=1)


class HistoryFeatures:
    """[A2] One immutable reference pool per training cutoff."""

    LEVELS = [("market", [], [90, 180, 365]),
              ("type", ["house_type_name"], [90, 180, 365]),
              ("community_type", ["community", "house_type_name"], [90, 180, 365]),
              ("location_type", ["location_group", "house_type_name"], [365])]

    def __init__(self, reference, cutoff):
        self.cutoff = pd.Timestamp(cutoff)
        ref = reference.loc[pd.to_datetime(reference.date_sold) < self.cutoff].reset_index(drop=True)
        if ref.empty or not (ref.sold_price > 0).all():
            raise ValueError("Historical pool must contain positive prices before cutoff")
        if (pd.to_datetime(ref.date_sold) < pd.to_datetime(ref.date_listed)).any():
            raise ValueError("Sale before listing would permit self-target leakage")
        self.reference_n = len(ref)
        self.latest_sale = pd.to_datetime(ref.date_sold).max()
        base = property_features(ref)
        dates = pd.to_datetime(ref.date_sold).to_numpy(dtype="datetime64[ns]")
        log_price = np.log(ref.sold_price.to_numpy())
        log_ppsf = log_price - base.log_area.to_numpy()
        self.tables = {}
        for name, columns, _ in self.LEVELS:
            keys = group_keys(base, columns)
            table = {}
            for key, positions in keys.groupby(keys, sort=False).groups.items():
                idx = np.asarray(positions)
                idx = idx[np.argsort(dates[idx], kind="stable")]
                valid = np.isfinite(log_ppsf[idx])
                table[key] = (dates[idx],
                              np.r_[0., np.cumsum(log_price[idx])],
                              np.r_[0., np.cumsum(np.where(valid, log_ppsf[idx], 0.))],
                              np.r_[0, np.cumsum(valid)])
            self.tables[name] = table

    def transform(self, homes):
        x = property_features(homes)
        date = pd.to_datetime(homes.date_listed).to_numpy(dtype="datetime64[ns]")
        query = np.minimum(date, self.cutoff.to_datetime64())
        for name, columns, windows in self.LEVELS:
            keys = group_keys(x, columns)
            for days in windows:
                n = np.zeros(len(x))
                sums = np.zeros(len(x))
                ppsf_sums = np.zeros(len(x))
                ppsf_n = np.zeros(len(x))
                for key, positions in keys.groupby(keys, sort=False).groups.items():
                    history = self.tables[name].get(key)
                    if history is None:
                        continue
                    idx = np.asarray(positions)
                    dates, prices, ppsf, ppsf_counts = history
                    # side=left excludes every sale at the query timestamp,
                    # including same-day sales and the training row itself.
                    right = np.searchsorted(dates, query[idx], side="left")
                    left = np.searchsorted(dates, query[idx] - np.timedelta64(days, "D"), side="left")
                    n[idx] = right - left
                    sums[idx] = prices[right] - prices[left]
                    ppsf_sums[idx] = ppsf[right] - ppsf[left]
                    ppsf_n[idx] = ppsf_counts[right] - ppsf_counts[left]
                if name == "market":
                    parent_price = np.full(len(x), np.nan)
                    parent_ppsf = parent_price
                    strength = 0
                else:
                    parent = "market" if name == "type" else "type"
                    parent_price = x[f"hist_{parent}_{days}_log_price"].to_numpy()
                    parent_ppsf = x[f"hist_{parent}_{days}_log_ppsf"].to_numpy()
                    strength = 10 if name == "type" else 20
                # Hierarchical smoothing only uses statistics available at the
                # SAME query time. Empty history remains NaN, never a full-
                # training target mean (which would leak into early rows).
                for suffix, total, count, parent in [("log_price", sums, n, parent_price),
                                                     ("log_ppsf", ppsf_sums, ppsf_n, parent_ppsf)]:
                    weight = np.where(np.isfinite(parent), strength, 0)
                    denom = count + weight
                    mean = np.divide(total + weight * np.nan_to_num(parent), denom,
                                     out=np.full(len(x), np.nan), where=denom > 0)
                    x[f"hist_{name}_{days}_{suffix}"] = mean
                x[f"hist_{name}_{days}_n"] = n
            if 90 in windows:
                x[f"hist_{name}_trend_90_vs_365"] = x[f"hist_{name}_90_log_ppsf"] - x[f"hist_{name}_365_log_ppsf"]
        return x
