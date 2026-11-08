"""Shared model configuration; only training population and applicable columns differ."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import Booster

from experiments.plan_a.models import PriceModel

ALGORITHMS = ("lightgbm", "catboost")
TYPES = ("Co-Op Apt", "Condo Apt", "Condo Townhouse", "Detached",
         "Freehold Townhouse", "Semi-Detached")
APARTMENT_TYPES = frozenset(TYPES[:3])
LAND_TYPES = frozenset(TYPES[3:])
LOT_COLUMNS = ("lot_front", "lot_depth", "lot_area", "log_lot_area", "lot_missing")
MIN_SPECIALIST_TRAIN = 2000
SCHEMES = tuple(f"{level}__{algorithm}" for level in ("global", "family", "type")
                for algorithm in ALGORITHMS)


def family(kind):
    if kind in APARTMENT_TYPES:
        return "apartment"
    if kind in LAND_TYPES:
        return "land"
    return "unknown"


def type_slug(kind):
    return kind.lower().replace("-", "_").replace(" ", "_")


def component_id(scope, algorithm):
    return f"{scope}__{algorithm}"


def mask_for(homes, spec):
    if spec["level"] == "global":
        return np.ones(len(homes), dtype=bool)
    if spec["level"] == "family":
        return homes.house_type_name.map(family).eq(spec["group"]).to_numpy()
    return homes.house_type_name.eq(spec["group"]).to_numpy()


def component_specs(train, min_train=MIN_SPECIALIST_TRAIN):
    """Rare Co-Op never gets its own model, even if a future dataset grows."""
    scopes = [("global", "all", "global")]
    for group in ("apartment", "land"):
        if train.house_type_name.map(family).eq(group).sum() >= min_train:
            scopes.append(("family", group, f"family_{group}"))
    for kind in TYPES:
        if kind != "Co-Op Apt" and train.house_type_name.eq(kind).sum() >= min_train:
            scopes.append(("type", kind, f"type_{type_slug(kind)}"))
    return [{"id": component_id(scope, algorithm), "level": level, "group": group,
             "algorithm": algorithm,
             "drop_lot": (level == "family" and group == "apartment")
                         or (level == "type" and group in APARTMENT_TYPES)}
            for level, group, scope in scopes for algorithm in ALGORITHMS]


def resolve_component(kind, scheme, available):
    """Deterministic routing; unknown types always use the shared model."""
    level, algorithm = scheme.split("__")
    global_id = component_id("global", algorithm)
    if global_id not in available:
        raise ValueError(f"Missing global fallback: {global_id}")
    if kind not in TYPES or level == "global":
        return global_id
    if level == "type":
        specialist = component_id(f"type_{type_slug(kind)}", algorithm)
        if kind == "Co-Op Apt":
            return global_id
        if specialist in available:
            return specialist
    broader = component_id(f"family_{family(kind)}", algorithm)
    return broader if broader in available else global_id


class SpecialistModel(PriceModel):
    def __init__(self, spec):
        super().__init__(spec["algorithm"])
        self.spec = dict(spec)

    def fit(self, x, prices):
        x = x.drop(columns=list(LOT_COLUMNS), errors="ignore") if self.spec["drop_lot"] else x
        return super().fit(x, prices)

    def save_at(self, folder, cutoff, training_n):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / ("model.cbm" if self.name == "catboost" else "model.txt")
        if self.name == "catboost":
            self.model.save_model(str(path))
        else:
            self.model.booster_.save_model(str(path))
        schema = {"spec": self.spec, "features": self.feature_names,
                  "categories": self.categories, "history_cutoff_exclusive": cutoff,
                  "training_n": int(training_n), "target": "log(sold_price)",
                  "model_file": path.name}
        (folder / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        schema = json.loads((folder / "schema.json").read_text())
        result = cls(schema["spec"])
        result.feature_names, result.categories = schema["features"], schema["categories"]
        path = folder / schema["model_file"]
        if result.name == "catboost":
            result.model = CatBoostRegressor()
            result.model.load_model(str(path))
        else:
            result.model = Booster(model_file=str(path))
        return result


def predict_routed(x, homes, routes, available, get_model):
    if len(x) != len(homes):
        raise ValueError("Features and homes must have identical row order and length")
    default = routes["__default__"]
    identifiers = [resolve_component(k, routes.get(k, default), available)
                   for k in homes.house_type_name]
    values = np.full(len(homes), np.nan)
    for identifier in sorted(set(identifiers)):
        positions = np.flatnonzero(np.asarray(identifiers) == identifier)
        values[positions] = get_model(identifier).predict(x.iloc[positions])
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Routing left missing, invalid, or nonpositive predictions")
    return values, identifiers
