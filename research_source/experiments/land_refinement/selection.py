"""Small predeclared candidate set; chronological selection with stability gate."""
import numpy as np
import pandas as pd

from experiments.plan_a.run_experiment import metrics

TARGET_TYPES = ("Detached", "Semi-Detached", "Freehold Townhouse")
DEV_CUTOFFS = {"2025-01-01", "2025-05-01", "2025-09-01"}


def select(rows):
    if set(rows.protocol) != {"development"} or set(rows.cutoff_exclusive) != DEV_CUTOFFS:
        raise ValueError("Selection must only use the three development windows")
    if rows.duplicated(["ml_num", "candidate"]).any():
        raise ValueError("Duplicate candidate rows")
    choices, details = {}, {}
    for kind in TARGET_TYPES:
        frame = rows.loc[rows.house_type_name == kind]
        candidates = {identifier: data for identifier, data in frame.groupby("candidate")}
        if not {"previous", "global"}.issubset(candidates):
            raise ValueError("Missing reference candidates")
        for name, data in candidates.items():
            if set(data.ml_num) != set(frame.ml_num) or set(data.cutoff_exclusive) != DEV_CUTOFFS:
                raise ValueError(f"Incomplete candidate: {name}")
        summary = {name: metrics(data) for name, data in candidates.items()}
        anchor = min(("previous", "global"), key=lambda name: (summary[name]["mdape"], name))
        eligible = [anchor]
        detail = {}
        for name, data in candidates.items():
            wins = 0
            windows = {}
            for cutoff in sorted(DEV_CUTOFFS):
                current = metrics(data.loc[data.cutoff_exclusive == cutoff])
                baseline = metrics(candidates[anchor].loc[candidates[anchor].cutoff_exclusive == cutoff])
                wins += int(current["mdape"] < baseline["mdape"])
                windows[cutoff] = current
            passes = summary[name]["mdape"] < summary[anchor]["mdape"] and wins >= 2
            if passes:
                eligible.append(name)
            detail[name] = {"overall": summary[name], "by_window": windows,
                            "windows_better_than_anchor": wins, "eligible": passes or name == anchor}
        choices[kind] = min(eligible, key=lambda name: (summary[name]["mdape"], name))
        details[kind] = {"anchor": anchor, "selected": choices[kind], "candidates": detail}
    return choices, details


def route(rows, choices):
    expected = set(rows.ml_num)
    chosen = rows.loc[rows.candidate.eq(rows.house_type_name.map(choices))].copy()
    if set(chosen.ml_num) != expected or chosen.ml_num.duplicated().any():
        raise ValueError("Routing lost or duplicated target homes")
    chosen["selected_candidate"] = chosen.candidate
    chosen["candidate"] = "selected"
    return chosen


def mix_predictions(left, right):
    """Join by ID before arithmetic blending, never by incidental file order."""
    if left.ml_num.duplicated().any() or right.ml_num.duplicated().any() or set(left.ml_num) != set(right.ml_num):
        raise ValueError("Blend ID coverage mismatch")
    joined = left[["ml_num", "sold_price"]].merge(right[["ml_num", "sold_price", "predicted_price"]],
        on="ml_num", validate="one_to_one", suffixes=("", "_right"))
    if not np.array_equal(joined.sold_price, joined.sold_price_right):
        raise ValueError("Blend target mismatch")
    return np.round(.5 * left.predicted_price.to_numpy() + .5 * joined.predicted_price.to_numpy())
