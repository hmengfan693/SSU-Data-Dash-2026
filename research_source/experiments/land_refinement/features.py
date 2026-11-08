"""Comparable sales queried from completed months; no fitted full-period scaler."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from experiments.plan_a.features import property_features

GEOMETRY = ["land_to_floor_ratio", "front_to_depth_ratio", "floor_per_bedroom",
            "washrooms_per_1000_sqft"]
COMPARABLES = ["comp_log_price", "comp_log_ppsf", "comp_log_price_std", "comp_n",
               "comp_mean_distance", "comp_mean_age_days", "comp_type_fallback", "comp_available"]
CONFIG = {"lookback_days": 365, "shortlist": 100, "neighbors": 20, "min_local": 5,
          "log_area_scale": .35, "log_lot_scale": .7, "style_penalty": .75,
          "query": "completed month before min(listed date, fixed training cutoff)"}


def geometry_features(homes):
    x = property_features(homes)
    result = pd.DataFrame(index=x.index)
    result["land_to_floor_ratio"] = x.lot_area / x.area_mid.where(x.area_mid > 0)
    result["front_to_depth_ratio"] = x.lot_front / x.lot_depth.where(x.lot_depth > 0)
    result["floor_per_bedroom"] = x.area_mid / x.bedrooms.where(x.bedrooms > 0)
    result["washrooms_per_1000_sqft"] = 1000 * x.washrooms / x.area_mid.where(x.area_mid > 0)
    return result.replace([np.inf, -np.inf], np.nan)


def coordinates(homes):
    x = property_features(homes)
    return np.column_stack([x.log_area.to_numpy() / CONFIG["log_area_scale"],
        np.log(x.lot_area.where(x.lot_area > 0)).to_numpy() / CONFIG["log_lot_scale"]])


class ComparableFeatures:
    def __init__(self, reference, cutoff):
        self.cutoff = pd.Timestamp(cutoff)
        if self.cutoff != self.cutoff.to_period("M").start_time:
            raise ValueError("Cutoff must be a month start")
        self.ref = reference.loc[pd.to_datetime(reference.date_sold) < self.cutoff].reset_index(drop=True)
        if (pd.to_datetime(self.ref.date_sold) < pd.to_datetime(self.ref.date_listed)).any():
            raise ValueError("Sale before listing")
        if not np.isfinite(self.ref.sold_price).all() or not (self.ref.sold_price > 0).all():
            raise ValueError("Comparable prices must be positive")
        self.coords = coordinates(self.ref)
        self.sold = pd.to_datetime(self.ref.date_sold).to_numpy(dtype="datetime64[ns]")
        self.log_price = np.log(self.ref.sold_price.to_numpy())
        self.log_area = property_features(self.ref).log_area.to_numpy()
        self.cache = {}

    def pool(self, month, kind, community):
        key = (month, kind, community)
        if key in self.cache:
            return self.cache[key]
        q = np.datetime64(month, "ns")
        eligible = ((self.sold < q) & (self.sold >= q - np.timedelta64(CONFIG["lookback_days"], "D"))
                    & self.ref.house_type_name.eq(kind).to_numpy() & np.isfinite(self.coords).all(axis=1))
        if community is not None:
            eligible &= self.ref.community.eq(community).to_numpy()
        positions = np.flatnonzero(eligible)
        tree = cKDTree(self.coords[positions]) if len(positions) else None
        value = (positions, tree)
        self.cache[key] = value
        return value

    def transform(self, homes):
        homes = homes.reset_index(drop=True)
        coords = coordinates(homes)
        date = pd.to_datetime(homes.date_listed)
        query = pd.Series(np.minimum(date.to_numpy(), self.cutoff.to_datetime64())).dt.to_period("M").dt.start_time
        result = pd.DataFrame(np.nan, index=homes.index, columns=COMPARABLES)
        result[["comp_n", "comp_available"]] = 0.
        group_table = pd.DataFrame({"month": query, "kind": homes.house_type_name, "community": homes.community})
        for (month, kind, community), idx in group_table.groupby(["month", "kind", "community"], sort=True).groups.items():
            positions, tree = self.pool(month, kind, community)
            fallback = len(positions) < CONFIG["min_local"]
            if fallback:
                positions, tree = self.pool(month, kind, None)
            query_idx = np.asarray(idx)[np.isfinite(coords[np.asarray(idx)]).all(axis=1)]
            if tree is None or not len(query_idx):
                continue
            distance, nearest = tree.query(coords[query_idx], k=min(CONFIG["shortlist"], len(positions)), workers=1)
            distance = np.asarray(distance).reshape(len(query_idx), -1)
            nearest = np.asarray(nearest).reshape(len(query_idx), -1)
            refs = positions[nearest]
            mismatch = self.ref.house_style.to_numpy()[refs] != homes.house_style.to_numpy()[query_idx, None]
            distance = np.sqrt(distance ** 2 + CONFIG["style_penalty"] ** 2 * mismatch)
            order = np.argsort(distance, axis=1, kind="stable")[:, :CONFIG["neighbors"]]
            refs = np.take_along_axis(refs, order, axis=1)
            distance = np.take_along_axis(distance, order, axis=1)
            weight = np.exp(-distance)
            weight /= weight.sum(axis=1, keepdims=True)
            log_price = self.log_price[refs]
            mean = np.sum(weight * log_price, axis=1)
            age = (date.to_numpy()[query_idx, None] - self.sold[refs]) / np.timedelta64(1, "D")
            values = np.column_stack([mean, np.sum(weight * (log_price - self.log_area[refs]), axis=1),
                np.sqrt(np.sum(weight * (log_price - mean[:, None]) ** 2, axis=1)),
                np.full(len(query_idx), refs.shape[1]), np.sum(weight * distance, axis=1),
                np.sum(weight * age, axis=1), np.full(len(query_idx), float(fallback)), np.ones(len(query_idx))])
            result.loc[query_idx, COMPARABLES] = values
        return result
