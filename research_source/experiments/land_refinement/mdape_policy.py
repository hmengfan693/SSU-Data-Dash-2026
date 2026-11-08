"""Apply the user's updated MdAPE-only rule to frozen development predictions."""
import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.plan_a.run_experiment import metrics
from .selection import DEV_CUTOFFS, TARGET_TYPES


def select_mdape(rows):
    """MAPE is descriptive only, including ties; stability is based on MdAPE."""
    if set(rows.protocol) != {"development"} or set(rows.cutoff_exclusive) != DEV_CUTOFFS:
        raise ValueError("Only the three development windows may select models")
    if rows.duplicated(["ml_num", "candidate"]).any():
        raise ValueError("Duplicate candidate rows")
    choices, details = {}, {}
    for kind in TARGET_TYPES:
        frame = rows.loc[rows.house_type_name == kind]
        candidates = {name: data for name, data in frame.groupby("candidate")}
        if not {"previous", "global"}.issubset(candidates):
            raise ValueError("Missing reference candidates")
        for name, data in candidates.items():
            if set(data.ml_num) != set(frame.ml_num) or set(data.cutoff_exclusive) != DEV_CUTOFFS:
                raise ValueError(f"Incomplete candidate: {name}")
        scores = {name: metrics(data) for name, data in candidates.items()}
        anchor = min(("previous", "global"), key=lambda name: (scores[name]["mdape"], name))
        eligible, detail = [anchor], {}
        for name, data in candidates.items():
            wins, windows = 0, {}
            for cutoff in sorted(DEV_CUTOFFS):
                current = metrics(data.loc[data.cutoff_exclusive == cutoff])
                baseline = metrics(candidates[anchor].loc[candidates[anchor].cutoff_exclusive == cutoff])
                wins += int(current["mdape"] < baseline["mdape"])
                windows[cutoff] = current
            passes = scores[name]["mdape"] < scores[anchor]["mdape"] and wins >= 2
            if passes:
                eligible.append(name)
            detail[name] = {"mdape": scores[name]["mdape"], "windows_better_than_anchor": wins,
                            "eligible": passes or name == anchor,
                            "window_mdape": {k: v["mdape"] for k, v in windows.items()}}
        choices[kind] = min(eligible, key=lambda name: (scores[name]["mdape"], name))
        details[kind] = {"anchor": anchor, "selected": choices[kind], "candidates": detail}
    return choices, details


def audit_existing(run):
    run = Path(run)
    rows = pd.read_csv(run / "all_candidate_predictions.csv")
    # Check-period predictions are discarded before invoking the selector.
    choices, details = select_mdape(rows.loc[rows.protocol == "development"])
    frozen = json.loads((run / "selection.json").read_text())["choices"]
    audit = {"policy": "pooled development MdAPE only; >=2 of 3 window MdAPE improvements; ties by name",
        "mape_used_for_eligibility_or_ties": False, "check_labels_used_for_selection": False,
        "choices": choices, "details": details, "same_as_previously_frozen_choices": choices == frozen}
    if choices != frozen:
        raise ValueError("MdAPE-only choices differ; a separate refit is required before publishing predictions")
    (run / "mdape_only_selection.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_existing(args.run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
