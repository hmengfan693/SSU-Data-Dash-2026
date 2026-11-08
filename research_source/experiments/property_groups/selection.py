"""Selection only accepts the predeclared 2025 development predictions."""
import pandas as pd

from experiments.plan_a.run_experiment import metrics
from .models import ALGORITHMS, GROUPS, candidate_id

DEV_WINDOWS = (("2025-01-01", "2025-05-01"), ("2025-05-01", "2025-09-01"),
               ("2025-09-01", "2026-01-01"))
CHECK_WINDOW = ("2026-02-01", "2026-06-01")


def choose_routes(rows):
    allowed = {start for start, _ in DEV_WINDOWS}
    if set(rows.protocol) != {"development"} or set(rows.cutoff_exclusive) != allowed:
        raise ValueError("Selection requires exactly the three development cutoffs")
    if set(rows.month) != set(pd.period_range("2025-01", "2025-12", freq="M").astype(str)):
        raise ValueError("Selection must contain the complete 2025 development year")
    if rows.duplicated(["ml_num", "candidate"]).any():
        raise ValueError("Duplicate predictions would distort model selection")
    scores, choices = {}, {}
    for group in GROUPS:
        frame = rows.loc[rows.group == group]
        ids = [candidate_id(scope, name) for scope in ("global", group) for name in ALGORITHMS]
        scores[group] = {}
        expected = set(frame.ml_num)
        if not expected:
            raise ValueError(f"No development examples for {group}")
        for identifier in ids:
            candidate = frame.loc[frame.candidate == identifier]
            if set(candidate.ml_num) != expected:
                raise ValueError(f"Incomplete candidate coverage: {identifier}")
            scores[group][identifier] = metrics(candidate)
        choices[group] = min(ids, key=lambda i: (scores[group][i]["mdape"], scores[group][i]["mape"], i))
    global_scores = {candidate_id("global", name): metrics(rows.loc[rows.candidate == candidate_id("global", name)])
                     for name in ALGORITHMS}
    choices["__default__"] = min(global_scores, key=lambda i: (global_scores[i]["mdape"], global_scores[i]["mape"], i))
    return choices, {"by_group": scores, "global": global_scores}


def route_rows(rows, choices):
    parts = []
    for group in GROUPS:
        part = rows.loc[(rows.group == group) & (rows.candidate == choices[group])].copy()
        if part.ml_num.duplicated().any():
            raise ValueError("A routed home may only be scored once")
        part["selected_candidate"] = part.candidate
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if set(result.ml_num) != set(rows.ml_num) or result.ml_num.duplicated().any():
        raise ValueError("Routing lost or duplicated homes")
    result["candidate"] = "routed"
    return result
