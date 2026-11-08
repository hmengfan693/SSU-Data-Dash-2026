"""Explicit group membership and serialized models; no label-dependent routing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import Booster

from experiments.plan_a.models import PARAMS, PriceModel

GROUPS = {
    "apartments": {"label": "普通公寓及 Co-Op", "types": ["Condo Apt", "Co-Op Apt"], "drop_lot": True},
    "condo_townhouse": {"label": "Condo Townhouse", "types": ["Condo Townhouse"], "drop_lot": True},
    "detached": {"label": "独立屋", "types": ["Detached"], "drop_lot": False},
    "attached_freehold": {"label": "半独立屋及 Freehold Townhouse", "types": ["Semi-Detached", "Freehold Townhouse"], "drop_lot": False},
}
TYPE_TO_GROUP = {kind: group for group, spec in GROUPS.items() for kind in spec["types"]}
ALGORITHMS = ("lightgbm", "catboost")
LOT_COLUMNS = ("lot_front", "lot_depth", "lot_area", "log_lot_area", "lot_missing")


def group_labels(homes, strict=True):
    result = homes.house_type_name.map(TYPE_TO_GROUP)
    if strict and result.isna().any():
        raise ValueError(f"Unknown property types: {homes.loc[result.isna(), 'house_type_name'].unique().tolist()}")
    return result.fillna("__unknown__")


def candidate_id(scope, algorithm):
    if scope not in ("global", *GROUPS) or algorithm not in ALGORITHMS:
        raise ValueError("Unknown scope or algorithm")
    return f"{scope}__{algorithm}"


class GroupModel(PriceModel):
    def __init__(self, scope, algorithm):
        candidate_id(scope, algorithm)
        super().__init__(algorithm)
        self.scope = scope

    def fit(self, x, prices):
        if self.scope != "global" and GROUPS[self.scope]["drop_lot"]:
            x = x.drop(columns=list(LOT_COLUMNS))
        return super().fit(x, prices)

    def save_bundle(self, folder, cutoff, training_n):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        model_file = "model.cbm" if self.name == "catboost" else "model.txt"
        path = folder / model_file
        if self.name == "catboost":
            self.model.save_model(str(path))
        else:
            self.model.booster_.save_model(str(path))
        schema = {"scope": self.scope, "algorithm": self.name, "features": self.feature_names,
                  "categories": self.categories, "cutoff_exclusive": str(cutoff),
                  "training_n": int(training_n), "target": "log(sold_price)",
                  "params": PARAMS[self.name], "model_file": model_file}
        (folder / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load_bundle(cls, folder):
        folder = Path(folder)
        schema = json.loads((folder / "schema.json").read_text())
        result = cls(schema["scope"], schema["algorithm"])
        result.feature_names, result.categories = schema["features"], schema["categories"]
        if result.name == "catboost":
            result.model = CatBoostRegressor()
            result.model.load_model(str(folder / schema["model_file"]))
        else:
            result.model = Booster(model_file=str(folder / schema["model_file"]))
        return result


def routed_predict(x, homes, choices, models):
    """Preserve positional order, including shuffled/nonconsecutive DataFrame indices."""
    if len(x) != len(homes):
        raise ValueError("Feature and home row counts differ")
    groups = group_labels(homes, strict=False)
    routes = np.asarray([choices.get(g, choices["__default__"]) for g in groups])
    price = np.full(len(homes), np.nan)
    for identifier in sorted(set(routes)):
        positions = np.flatnonzero(routes == identifier)
        price[positions] = models[identifier].predict(x.iloc[positions])
    if not np.isfinite(price).all() or not (price > 0).all():
        raise ValueError("Routed predictions must be positive and finite")
    return price, routes
