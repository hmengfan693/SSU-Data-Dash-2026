"""Expanding-prefix HMM fits: neither parameters nor filters see future months.

Only completed months before min(listing date, reference cutoff) are available.
Emission: matched-cell log-price-per-square-foot change (one dimension).
Two states, ordered by fitted emission mean, are not named bull/bear ex ante.
"""
import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from experiments.plan_a.features import property_features

MIN_MONTHS = 24
SEEDS = (11, 29, 47)
RAW_COLUMNS = ["market_matched_change", "market_change_mean_3", "market_change_std_6",
               "market_volume_yoy", "market_matched_cells", "market_match_share"]
STATE_COLUMNS = ["hmm_available", "hmm_prob_high_current"] + [
    name for k in range(1, 5) for name in (f"hmm_prob_high_plus_{k}", f"hmm_expected_change_plus_{k}")
] + ["hmm_expected_cumulative_change_4"]


def monthly_observations(reference, cutoff):
    """Same-community/type/400-sqft-band matched-cell change, not a fixed-quality index."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff != cutoff.to_period("M").start_time:
        raise ValueError("Reference cutoff must be a month start")
    ref = reference.loc[pd.to_datetime(reference.date_sold) < cutoff].reset_index(drop=True)
    if ref.empty:
        raise ValueError("No reference observations")
    x = property_features(ref)
    month = pd.to_datetime(ref.date_sold).dt.to_period("M")
    cells = pd.DataFrame({"month": month, "community": ref.community,
                          "type": ref.house_type_name,
                          "area_band": np.floor(x.area_mid / 400),
                          "log_ppsf": np.log(ref.sold_price) - x.log_area})
    grouped = cells.groupby(["month", "community", "type", "area_band"], observed=True).log_ppsf.agg(["median", "count"])
    periods = pd.period_range(month.min(), cutoff.to_period("M") - 1, freq="M")
    records = []
    previous = None
    counts = month.value_counts()
    for period in periods:
        current = grouped.xs(period, level="month") if period in grouped.index.get_level_values("month") else pd.DataFrame(columns=["median", "count"])
        change, matches, matched_n = np.nan, 0, 0
        if previous is not None and len(previous) and len(current):
            joined = current.join(previous, lsuffix="_current", rsuffix="_previous", how="inner")
            joined = joined.loc[(joined.count_current >= 3) & (joined.count_previous >= 3)]
            matches = len(joined)
            if matches >= 5:
                weights = np.minimum(joined.count_current, joined.count_previous).clip(upper=20)
                change = float(np.average(joined.median_current - joined.median_previous, weights=weights))
                matched_n = int(joined.count_current.sum())
        n = int(counts.get(period, 0))
        records.append({"month": str(period), "matched_change": change, "volume": n,
                        "matched_cells": matches, "match_share": matched_n / n if n else 0.})
        previous = current
    result = pd.DataFrame(records).set_index("month")
    result["change_mean_3"] = result.matched_change.rolling(3, min_periods=3).mean()
    result["change_std_6"] = result.matched_change.rolling(6, min_periods=6).std(ddof=0)
    result["volume_yoy"] = np.log(result.volume.where(result.volume > 0) / result.volume.shift(12).where(result.volume.shift(12) > 0))
    return result


def filtered_last(model, observations):
    """Explicit forward filtering; no smoothed/Viterbi path is used for features."""
    values = np.asarray(observations, dtype=float).reshape(-1, 1)
    means = model.means_.reshape(-1)
    variances = np.asarray(model.covars_).reshape(model.n_components, -1)[:, 0]
    log_transition = np.log(np.maximum(model.transmat_, 1e-300))
    log_probability = np.log(np.maximum(model.startprob_, 1e-300))
    for i, value in enumerate(values[:, 0]):
        if i:
            log_probability = logsumexp(log_probability[:, None] + log_transition, axis=0)
        log_probability += -.5 * (np.log(2 * np.pi * variances) + (value - means) ** 2 / variances)
        log_probability -= logsumexp(log_probability)
    return np.exp(log_probability)


def state_forecast(probability, transition, emission_means, steps=4):
    p = np.asarray(probability, dtype=float)
    forecasts = []
    for _ in range(steps):
        p = p @ transition
        forecasts.append((p.copy(), float(p @ emission_means)))
    return forecasts


def fit_prefix(observations):
    """Fit only one historical prefix; seeds selected by its own likelihood."""
    values = np.asarray(observations, dtype=float).reshape(-1, 1)
    center, scale = float(values.mean()), float(values.std())
    scale = max(scale, 1e-4)
    standardized = (values - center) / scale
    candidates, diagnostics = [], []
    logger = logging.getLogger("hmmlearn.base")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        for seed in SEEDS:
            model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=300,
                                tol=1e-4, min_covar=1e-3, random_state=seed,
                                transmat_prior=1.1, startprob_prior=1.1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    model.fit(standardized)
                    likelihood = float(model.score(standardized))
                    iterations = int(model.monitor_.iter)
                    trajectory = list(model.monitor_.history)
                    last_gain = float(trajectory[-1] - trajectory[-2]) if len(trajectory) > 1 else None
                    occupancy = model.predict_proba(standardized).sum(axis=0)
                    # Smoothed occupancy is only a training-fit diagnostic, never an input feature.
                    valid = (np.isfinite(likelihood) and np.isfinite(model.transmat_).all()
                             and np.isfinite(model.covars_).all() and np.all(occupancy >= 2)
                             and iterations < 300 and last_gain is not None and last_gain >= -1e-3)
                    diagnostics.append({"seed": seed, "log_likelihood": likelihood,
                                        "iterations": iterations, "last_gain": last_gain,
                                        "occupancy": occupancy.tolist(), "accepted": bool(valid)})
                    if valid:
                        candidates.append((likelihood, seed, model))
                except (ValueError, FloatingPointError) as error:
                    diagnostics.append({"seed": seed, "accepted": False, "error": str(error)})
    finally:
        logger.setLevel(old_level)
    if not candidates:
        return None, {"status": "no_valid_fit", "candidates": diagnostics}
    likelihood, seed, model = max(candidates, key=lambda item: item[0])
    raw_means = model.means_.ravel() * scale + center
    order = np.argsort(raw_means)
    probability = filtered_last(model, standardized)[order]
    transition = model.transmat_[np.ix_(order, order)]
    means = raw_means[order]
    forecasts = state_forecast(probability, transition, means)
    features = {"hmm_available": 1., "hmm_prob_high_current": float(probability[1])}
    cumulative = 0.
    for k, (p, change) in enumerate(forecasts, 1):
        features[f"hmm_prob_high_plus_{k}"] = float(p[1])
        features[f"hmm_expected_change_plus_{k}"] = change
        cumulative += change
    features["hmm_expected_cumulative_change_4"] = cumulative
    return features, {"status": "fitted", "seed": seed, "log_likelihood": likelihood,
                      "observation_n": len(values), "standardizer_mean": center,
                      "standardizer_scale": scale, "state_means_log_change": means.tolist(),
                      "transition": transition.tolist(), "filtered_probability": probability.tolist(),
                      "candidates": diagnostics}


class CausalMarketFeatures:
    def __init__(self, reference, cutoff):
        self.cutoff = pd.Timestamp(cutoff)
        self.monthly = monthly_observations(reference, cutoff)
        self.rows, self.audits = {}, {}
        # Each row corresponds to an information boundary at the next month start.
        for month in self.monthly.index:
            period = pd.Period(month, freq="M")
            boundary = (period + 1).start_time
            prefix = self.monthly.loc[:month]
            last = prefix.iloc[-1]
            feature = dict(zip(RAW_COLUMNS, [last.matched_change, last.change_mean_3,
                           last.change_std_6, last.volume_yoy, last.matched_cells, last.match_share]))
            feature.update({c: (0. if c == "hmm_available" else np.nan) for c in STATE_COLUMNS})
            observations = prefix.matched_change.dropna()
            audit = {"status": "warmup", "observation_n": len(observations)}
            # Gaps cannot silently be treated as adjacent monthly transitions.
            if len(observations) >= MIN_MONTHS and not prefix.matched_change.iloc[1:].isna().any():
                states, audit = fit_prefix(observations.to_numpy())
                if states is not None:
                    feature.update(states)
            self.rows[boundary] = feature
            self.audits[str(boundary.date())] = {"last_observed_month": month, **audit}
        self.cache = pd.DataFrame.from_dict(self.rows, orient="index").reindex(columns=RAW_COLUMNS + STATE_COLUMNS)

    def transform(self, homes):
        dates = pd.to_datetime(homes.date_listed).dt.to_period("M").dt.start_time
        # Same cutoff for all future listings. Four-month forecasts are anchored
        # here, not assigned using hidden date_sold or an assumed closing month.
        boundaries = dates.clip(upper=self.cutoff)
        result = self.cache.reindex(pd.DatetimeIndex(boundaries)).reset_index(drop=True)
        result["hmm_available"] = result.hmm_available.fillna(0)
        return result
