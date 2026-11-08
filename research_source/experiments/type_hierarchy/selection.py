"""Choose routes on earlier development windows, never on the evaluation labels."""
import numpy as np
import pandas as pd

from experiments.plan_a.run_experiment import metrics
from .models import ALGORITHMS, SCHEMES, TYPES

MIN_DEVELOPMENT_N = 200
MIN_IMPROVEMENT_PP = 0.10
MIN_WINDOW_WINS = 2


def apply_routes(rows, routes, scheme_name):
    """Select existing predictions by ID/type; preserve labels only for scoring."""
    chosen = []
    for kind, group in rows.groupby("house_type_name", sort=False):
        scheme = routes.get(kind, routes["__default__"])
        selected = group.loc[group.scheme.eq(scheme)].copy()
        expected = group[["cutoff_exclusive", "ml_num"]].drop_duplicates()
        if len(selected) != len(expected) or selected.duplicated(["cutoff_exclusive", "ml_num"]).any():
            raise ValueError(f"Route does not cover exactly one prediction per home: {kind}")
        selected["source_scheme"] = selected.scheme
        selected["scheme"] = scheme_name
        chosen.append(selected)
    return pd.concat(chosen, ignore_index=True)


def select_routes(development):
    if set(development.protocol) != {"development_A"}:
        raise ValueError("Selection accepts earlier development windows only")
    cutoffs = sorted(development.cutoff_exclusive.unique())
    if len(cutoffs) != 3:
        raise ValueError("Exactly three predeclared development windows are required")
    baseline = min((f"global__{a}" for a in ALGORITHMS),
                   key=lambda s: (metrics(development.loc[development.scheme.eq(s)])["mdape"],
                                  metrics(development.loc[development.scheme.eq(s)])["mape"], s))
    proposed = {"__default__": baseline}
    decisions = []
    for kind in TYPES:
        group = development.loc[development.house_type_name.eq(kind)]
        reference = group.loc[group.scheme.eq(baseline)]
        if reference.empty:
            proposed[kind] = baseline
            decisions.append({"house_type_name": kind, "selected_scheme": baseline,
                              "reason": "no_development_examples", "candidates": []})
            continue
        ref_score = metrics(reference)
        candidates = []
        eligible = []
        for scheme in SCHEMES:
            frame = group.loc[group.scheme.eq(scheme)]
            if set(zip(frame.cutoff_exclusive, frame.ml_num)) != set(zip(reference.cutoff_exclusive, reference.ml_num)):
                raise ValueError("Candidate cohorts differ")
            score = metrics(frame)
            wins = sum(metrics(frame.loc[frame.cutoff_exclusive.eq(c)])["mdape"]
                       < metrics(reference.loc[reference.cutoff_exclusive.eq(c)])["mdape"]
                       for c in cutoffs)
            gain = ref_score["mdape"] - score["mdape"]
            allowed = (kind != "Co-Op Apt" and score["n"] >= MIN_DEVELOPMENT_N
                       and gain >= MIN_IMPROVEMENT_PP and wins >= MIN_WINDOW_WINS)
            entry = {"scheme": scheme, **score, "improvement_pp": gain,
                     "window_wins": int(wins), "eligible": bool(allowed)}
            candidates.append(entry)
            if allowed:
                eligible.append(entry)
        winner = min(eligible, key=lambda s: (s["mdape"], s["mape"], SCHEMES.index(s["scheme"]))) if eligible else None
        proposed[kind] = winner["scheme"] if winner else baseline
        decisions.append({"house_type_name": kind, "development_n": len(reference),
                          "baseline": ref_score, "selected_scheme": proposed[kind],
                          "reason": "eligible_improvement" if winner else "shared_model_fallback",
                          "candidates": candidates})
    hybrid = apply_routes(development, proposed, "hybrid_proposed")
    reference = development.loc[development.scheme.eq(baseline)]
    overall_wins = sum(metrics(hybrid.loc[hybrid.cutoff_exclusive.eq(c)])["mdape"]
                       < metrics(reference.loc[reference.cutoff_exclusive.eq(c)])["mdape"] for c in cutoffs)
    accept = (metrics(hybrid)["mdape"] < metrics(reference)["mdape"] and overall_wins >= MIN_WINDOW_WINS)
    recommended = dict(proposed) if accept else {"__default__": baseline, **{k: baseline for k in TYPES}}
    return {"baseline_scheme": baseline, "proposed_routes": proposed,
            "recommended_routes": recommended, "accept_hierarchy": bool(accept),
            "development_baseline": metrics(reference), "development_proposed": metrics(hybrid),
            "development_overall_window_wins": int(overall_wins), "by_property_type": decisions,
            "rules": {"min_development_n": MIN_DEVELOPMENT_N,
                      "min_improvement_pp": MIN_IMPROVEMENT_PP, "min_window_wins": MIN_WINDOW_WINS,
                      "rare_type": "Co-Op Apt always uses the overall selected global model",
                      "global_safeguard": "pooled MdAPE improves and at least two of three windows improve"},
            "selection_uses_evaluation_labels": False}
