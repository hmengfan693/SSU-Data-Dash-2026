"""Select predeclared candidates on exactly three earlier windows using MdAPE."""
import numpy as np
import pandas as pd
from experiments.plan_a.run_experiment import metrics

TYPES = ['Detached','Semi-Detached','Freehold Townhouse','Co-Op Apt']
DEV = ['2025-01-01','2025-05-01','2025-09-01']
LAND_CANDIDATES=['previous','shared_original','location','location_ridge','location_catboost','text_catboost']
COOP_CANDIDATES=['previous','apartment_ridge','apartment_text']


def route(rows, choices):
    parts=[]
    for kind,g in rows.groupby('house_type_name',sort=False):
        candidate=choices.get(kind,'previous')
        z=g.loc[g.candidate.eq(candidate)].copy()
        keys=['cutoff_exclusive','ml_num']
        expected=g[keys].drop_duplicates()
        if len(z)!=len(expected) or z.duplicated(keys).any():
            raise ValueError('Incomplete route coverage')
        z['source_candidate']=candidate
        parts.append(z)
    return pd.concat(parts,ignore_index=True)


def choose(rows):
    if set(rows.protocol)!={'development'} or set(rows.cutoff_exclusive)!=set(DEV):
        raise ValueError('Selection accepts only the three development windows')
    if rows.duplicated(['cutoff_exclusive','ml_num','candidate']).any():
        raise ValueError('Duplicate predictions')
    choices={}; details=[]
    for kind in TYPES:
        g=rows.loc[rows.house_type_name.eq(kind)]
        ref=g.loc[g.candidate.eq('previous')]
        candidates=COOP_CANDIDATES if kind=='Co-Op Apt' else LAND_CANDIDATES
        minimum_n, required_wins, gain = (100,3,.25) if kind=='Co-Op Apt' else (200,2,.10)
        eligible=[]
        for candidate in candidates:
            z=g.loc[g.candidate.eq(candidate)]
            if set(zip(z.cutoff_exclusive,z.ml_num))!=set(zip(ref.cutoff_exclusive,ref.ml_num)):
                raise ValueError('Incomplete candidate cohort')
            wins=sum(metrics(z[z.cutoff_exclusive.eq(c)])['mdape']<metrics(ref[ref.cutoff_exclusive.eq(c)])['mdape'] for c in DEV)
            score=metrics(z)
            improvement=metrics(ref)['mdape']-score['mdape']
            allowed=candidate!='previous' and len(z)>=minimum_n and wins>=required_wins and improvement>=gain
            details.append({'house_type_name':kind,'candidate':candidate,**score,'wins':int(wins),'gain_pp':improvement,'eligible':bool(allowed)})
            if allowed:eligible.append((score['mdape'],candidate))
        choices[kind]=min(eligible)[1] if eligible else 'previous'
    proposed=route(rows,choices)
    ref=rows.loc[rows.candidate.eq('previous')]
    wins=sum(metrics(proposed[proposed.cutoff_exclusive.eq(c)])['mdape']<metrics(ref[ref.cutoff_exclusive.eq(c)])['mdape'] for c in DEV)
    accepted=metrics(ref)['mdape']-metrics(proposed)['mdape']>=.05 and wins>=2
    return {'choices': choices if accepted else {k:'previous' for k in TYPES},'proposed_choices':choices,
        'accepted':bool(accepted),'development_previous':metrics(ref),'development_proposed':metrics(proposed),
        'overall_window_wins':int(wins),'candidate_scores':details,
        'rules':{'metric':'MdAPE only','min_gain_pp':.10,'min_windows':2,'rare_min_gain_pp':.25,'rare_min_windows':3,
                 'rare_min_n':100,'ordinary_min_n':200,'overall_min_gain_pp':.05,'outer_check_labels_used':False}}
