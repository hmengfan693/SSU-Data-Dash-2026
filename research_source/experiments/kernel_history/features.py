"""Exact date-filtered neighbor retrieval before Gaussian kernel averaging.

The 365-day eligible pool is determined before building each KD-tree. No
future/self labels can displace valid neighbors. No fitted global imputation
or scaling uses later feature values. Bandwidths are fixed before scoring.
"""
import copy

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from experiments.plan_a.features import HistoryFeatures, property_features

CONFIG = dict(lookback_days=365, max_neighbors=64, min_effective_n=5.,
              log_area_bandwidth=0.25, washrooms_bandwidth=1.,
              latitude_bandwidth=0.02, longitude_bandwidth=0.03,
              time_efold_days=180., max_distance=3.)
KERNEL_COLUMNS = ["kernel_log_price", "kernel_log_ppsf", "kernel_log_price_std",
                  "kernel_effective_n", "kernel_neighbor_n", "kernel_fallback",
                  "kernel_mean_age_days", "kernel_nearest_distance",
                  "kernel_area_adjusted_log_price"]


def coordinates(base, config):
    """Fixed physical/ratio bandwidths, not statistics fitted on future rows."""
    return np.column_stack([
        base.log_area.to_numpy() / config["log_area_bandwidth"],
        base.washrooms.to_numpy() / config["washrooms_bandwidth"],
        (base.latitude.to_numpy() - 43.65) / config["latitude_bandwidth"],
        (base.longitude.to_numpy() + 79.4) / config["longitude_bandwidth"],
    ]).astype(np.float64)


def weighted_statistics(distances, ages, log_prices, log_ppsf, config):
    """Rows are queries; columns are eligible selected neighbors, padding=inf."""
    valid = np.isfinite(distances) & (distances <= config["max_distance"])
    log_weights = np.where(valid, -0.5 * distances ** 2 - ages / config["time_efold_days"], -np.inf)
    peak = log_weights.max(axis=1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.)
    weights = np.exp(log_weights - peak)
    total = weights.sum(axis=1)
    denominator = np.where(total > 0, total, 1.)
    normal = weights / denominator[:, None]
    mean = np.sum(normal * log_prices, axis=1)
    mean_ppsf = np.sum(normal * log_ppsf, axis=1)
    variance = np.sum(normal * (log_prices - mean[:, None]) ** 2, axis=1)
    square_total = np.square(weights).sum(axis=1)
    effective = np.divide(total ** 2, square_total, out=np.zeros_like(total), where=square_total > 0)
    nearest = np.min(np.where(valid, distances, np.inf), axis=1)
    return dict(mean=mean, mean_ppsf=mean_ppsf, std=np.sqrt(np.maximum(variance, 0.)),
                effective=effective, n=valid.sum(axis=1), age=np.sum(normal * ages, axis=1),
                nearest=np.where(np.isfinite(nearest), nearest, np.nan))


class KernelHistoryFeatures:
    def __init__(self, reference, cutoff, **config):
        if set(config) - set(CONFIG):
            raise ValueError("Unknown kernel configuration")
        self.config = copy.deepcopy(CONFIG)
        self.config.update(config)
        if any(not np.isfinite(v) or v <= 0 for v in self.config.values()):
            raise ValueError("Kernel settings must be finite and positive")
        if int(self.config["max_neighbors"]) != self.config["max_neighbors"]:
            raise ValueError("max_neighbors must be an integer")
        self.cutoff = pd.Timestamp(cutoff)
        self.reference = reference.loc[pd.to_datetime(reference.date_sold) < self.cutoff].sort_values(
            ["date_sold", "ml_num"], kind="stable").reset_index(drop=True)
        self.history = HistoryFeatures(self.reference, self.cutoff)
        base = property_features(self.reference)
        self.points = coordinates(base, self.config)
        self.complete = np.isfinite(self.points).all(axis=1)
        self.dates = pd.to_datetime(self.reference.date_sold).to_numpy(dtype="datetime64[ns]")
        self.log_prices = np.log(self.reference.sold_price.to_numpy(dtype=float))
        self.log_ppsf = self.log_prices - base.log_area.to_numpy()
        self.types = base.house_type_name.to_numpy()
        self.reference_n = len(self.reference)
        self.last_audit = None

    def transform(self, homes, logger=None):
        x = self.history.transform(homes)
        points = coordinates(x, self.config)
        complete = np.isfinite(points).all(axis=1)
        query = np.minimum(pd.to_datetime(homes.date_listed).to_numpy(dtype="datetime64[ns]"),
                           self.cutoff.to_datetime64())
        if np.isnat(query).any():
            raise ValueError("Every query needs a valid listing date")
        c = self.config
        kernel = pd.DataFrame(index=x.index)
        # Parent statistics are available at exactly the same query date.
        # If no parent history exists, NaN stays NaN (no full-target fallback).
        kernel["kernel_log_price"] = x.hist_community_type_365_log_price
        kernel["kernel_log_ppsf"] = x.hist_community_type_365_log_ppsf
        kernel["kernel_log_price_std"] = np.nan
        kernel["kernel_effective_n"] = 0.
        kernel["kernel_neighbor_n"] = 0
        kernel["kernel_fallback"] = 1
        kernel["kernel_mean_age_days"] = np.nan
        kernel["kernel_nearest_distance"] = np.nan
        query_types = x.house_type_name.to_numpy()
        trees_built, queries_with_history = 0, 0
        for kind in sorted(set(query_types)):
            ref_ids = np.flatnonzero((self.types == kind) & self.complete)
            q_ids = np.flatnonzero((query_types == kind) & complete)
            if not len(ref_ids) or not len(q_ids):
                continue
            dates = self.dates[ref_ids]
            # A fresh tree per exact query timestamp makes temporal selection
            # exact, including cutoff and rolling-window boundaries.
            groups = pd.Series(q_ids).groupby(query[q_ids], sort=True)
            for query_date, positions in groups:
                ids = positions.to_numpy(dtype=int)
                timestamp = np.datetime64(query_date, "ns")
                lo = np.searchsorted(dates, timestamp - np.timedelta64(c["lookback_days"], "D"), side="left")
                hi = np.searchsorted(dates, timestamp, side="left")
                eligible = ref_ids[lo:hi]
                if not len(eligible):
                    continue
                tree = cKDTree(self.points[eligible])
                trees_built += 1
                k = min(int(c["max_neighbors"]), len(eligible))
                distances, local = tree.query(points[ids], k=list(range(1, k + 1)), workers=1)
                selected = eligible[local]
                age = (timestamp - self.dates[selected]) / np.timedelta64(1, "D")
                if not (age > 0).all() or not (age <= c["lookback_days"]).all():
                    raise ValueError("Neighbor temporal boundary violation")
                stat = weighted_statistics(distances, age, self.log_prices[selected], self.log_ppsf[selected], c)
                good = stat["effective"] >= c["min_effective_n"]
                kernel.loc[ids, "kernel_effective_n"] = stat["effective"]
                kernel.loc[ids, "kernel_neighbor_n"] = stat["n"]
                kernel.loc[ids, "kernel_nearest_distance"] = stat["nearest"]
                for column, key in [("kernel_log_price", "mean"), ("kernel_log_ppsf", "mean_ppsf"),
                                    ("kernel_log_price_std", "std"), ("kernel_mean_age_days", "age")]:
                    kernel.loc[ids[good], column] = stat[key][good]
                kernel.loc[ids[good], "kernel_fallback"] = 0
                queries_with_history += len(ids)
            if logger is not None:
                logger(f"  kernel features: {kind}: {len(q_ids):,} complete queries")
        kernel["kernel_area_adjusted_log_price"] = kernel.kernel_log_ppsf + x.log_area
        kernel = kernel[KERNEL_COLUMNS]
        if np.isinf(kernel.to_numpy(dtype=float)).any():
            raise ValueError("Nonfinite kernel outputs")
        self.last_audit = dict(query_n=len(x), reference_n=self.reference_n,
                               reference_complete_n=int(self.complete.sum()),
                               query_missing_geometry_n=int((~complete).sum()),
                               queries_with_eligible_history=queries_with_history,
                               trees_built=trees_built,
                               fallback_n=int(kernel.kernel_fallback.sum()),
                               fallback_pct=float(100 * kernel.kernel_fallback.mean()) if len(x) else 0.,
                               mean_effective_n=float(kernel.kernel_effective_n.mean()) if len(x) else 0.,
                               median_effective_n=float(kernel.kernel_effective_n.median()) if len(x) else 0.,
                               query_cutoff_exclusive=str(self.cutoff),
                               temporal_filter_before_neighbor_search=True,
                               feature_columns=KERNEL_COLUMNS)
        return pd.concat([x, kernel], axis=1)
