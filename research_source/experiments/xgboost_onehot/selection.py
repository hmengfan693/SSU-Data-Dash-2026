"""One global encoding choice on the three earlier four-month windows only."""
from experiments.plan_a.run_experiment import metrics

NATIVE = 'xgboost_native'
ONEHOT = 'xgboost_house_type_onehot'
DEV_CUTOFFS = {'2025-01-01', '2025-05-01', '2025-09-01'}


def choose(rows):
    if set(rows.protocol) != {'development'} or set(rows.cutoff_exclusive) != DEV_CUTOFFS:
        raise ValueError('Selection accepts only three development windows')
    if set(rows.candidate) != {NATIVE, ONEHOT}:
        raise ValueError('Both encoding candidates required')
    if rows.duplicated(['candidate', 'cutoff_exclusive', 'ml_num']).any():
        raise ValueError('Duplicate selection cohort')
    cohorts, scores = {}, {}
    for name in (NATIVE, ONEHOT):
        part = rows.loc[rows.candidate.eq(name)]
        cohorts[name] = set(zip(part.cutoff_exclusive, part.ml_num))
        scores[name] = metrics(part)
    if cohorts[NATIVE] != cohorts[ONEHOT]:
        raise ValueError('Selection cohort mismatch')
    selected = min((NATIVE, ONEHOT), key=lambda name:
        (scores[name]['mdape'], scores[name]['mape'], name != NATIVE))
    return dict(selected_candidate=selected, candidate_scores=scores,
        rule='Pooled development MdAPE; MAPE breaks ties; exact ties retain native',
        selection_windows=sorted(DEV_CUTOFFS), outer_check_labels_used=False,
        global_choice=True, per_type_selection=False)
