"""Past-only support/age, price-free lot context, and conservative lexical state."""
import re

import numpy as np
import pandas as pd

from experiments.plan_a.features import group_keys, property_features

VARIANTS = ('baseline', 'history_only', 'land_only', 'text_only', 'all_three')
CONFIG = {'windows_days': [90, 365], 'reliable_min_n': 5,
          'lot_lookback_days': 365, 'lot_min_n': 20,
          'lot_query': 'completed month before min(date_listed, cutoff)',
          'text_version': 'clause_scoped_v1'}
LEVELS = [('community', ['community', 'house_type_name']),
          ('municipality', ['municipality', 'house_type_name']),
          ('type', ['house_type_name'])]


class ProcessingFeatures:
    def __init__(self, reference, cutoff):
        self.cutoff = pd.Timestamp(cutoff)
        self.ref = reference.loc[pd.to_datetime(reference.date_sold) < self.cutoff].reset_index(drop=True)
        if (pd.to_datetime(self.ref.date_sold) < pd.to_datetime(self.ref.date_listed)).any():
            raise ValueError('Sale before listing permits self leakage')
        self.base = property_features(self.ref)
        self.sold = pd.to_datetime(self.ref.date_sold).to_numpy(dtype='datetime64[ns]')
        self.tables = {}
        for name, columns in LEVELS:
            table = {}
            keys = group_keys(self.base, columns)
            for key, positions in keys.groupby(keys, sort=False).groups.items():
                dates = np.sort(self.sold[np.asarray(positions)])
                months = dates.astype('datetime64[M]')
                runs = np.cumsum(np.r_[True, months[1:] != months[:-1]])
                table[key] = (dates, runs)
            self.tables[name] = table
        self.lot_cache = {}

    def queries(self, homes):
        listed = pd.to_datetime(homes.date_listed).to_numpy(dtype='datetime64[ns]')
        if np.isnat(listed).any():
            raise ValueError('Listed dates required')
        return listed, np.minimum(listed, self.cutoff.to_datetime64())

    def history(self, homes):
        x = property_features(homes)
        listed, query = self.queries(homes)
        result = pd.DataFrame(index=x.index)
        result['info_days_after_cutoff'] = np.maximum((listed-self.cutoff.to_datetime64()) / np.timedelta64(1, 'D'), 0)
        stats = {}
        for name, columns in LEVELS:
            keys = group_keys(x, columns)
            for days in CONFIG['windows_days']:
                n = np.zeros(len(x))
                age = np.full(len(x), np.nan)
                covered = np.zeros(len(x))
                for key, positions in keys.groupby(keys, sort=False).groups.items():
                    table = self.tables[name].get(key)
                    if table is None:
                        continue
                    dates, runs = table
                    idx = np.asarray(positions)
                    right = np.searchsorted(dates, query[idx], side='left')
                    left = np.searchsorted(dates, query[idx]-np.timedelta64(days, 'D'), side='left')
                    n[idx] = right-left
                    good = right > left
                    dest, r, l = idx[good], right[good], left[good]
                    age[dest] = (query[dest]-dates[r-1])/np.timedelta64(1, 'D')
                    covered[dest] = runs[r-1]-runs[l]+1
                prefix = f'info_{name}_{days}'
                result[prefix+'_latest_age_days'] = age
                result[prefix+'_covered_months'] = covered
                if name == 'municipality':
                    result[prefix+'_n'] = n
                stats[name, days] = (n, age)
        for days in CONFIG['windows_days']:
            level = np.full(len(x), 3.)
            age = np.full(len(x), np.nan)
            for i, (name, _) in enumerate(LEVELS):
                n, a = stats[name, days]
                mask = (level == 3) & (n >= CONFIG['reliable_min_n'])
                level[mask], age[mask] = i, a[mask]
            result[f'info_reliable_{days}_level'] = level
            result[f'info_reliable_{days}_age_days'] = age
        return result

    def lot_tables(self, month):
        if month in self.lot_cache:
            return self.lot_cache[month]
        b = self.base
        valid = ((self.sold < month) & (self.sold >= month-np.timedelta64(CONFIG['lot_lookback_days'], 'D'))
                 & np.isfinite(b.lot_area) & (b.lot_area > 0)
                 & np.isfinite(b.lot_front) & (b.lot_front > 0)
                 & np.isfinite(b.lot_depth) & (b.lot_depth > 0))
        subset = b.loc[valid].copy()
        tables = {}
        for name, columns in LEVELS:
            table = {}
            keys = group_keys(subset, columns) if len(subset) else pd.Series(dtype=str)
            for key, positions in keys.groupby(keys, sort=False).groups.items():
                z = b.loc[positions]
                table[key] = (np.sort(np.log(z.lot_area.to_numpy())), np.sort(np.log(z.lot_front.to_numpy())))
            tables[name] = table
        self.lot_cache[month] = tables
        return tables

    def land(self, homes):
        x = property_features(homes)
        _, q = self.queries(homes)
        months = q.astype('datetime64[M]').astype('datetime64[ns]')
        names = ['land_context_log_area', 'land_relative_log_area', 'land_area_percentile',
                 'land_context_log_front', 'land_relative_log_front', 'land_front_percentile',
                 'land_context_n', 'land_context_level', 'land_context_available']
        result = pd.DataFrame(np.nan, index=x.index, columns=names)
        result[['land_context_n', 'land_context_available']] = 0.
        result['land_context_level'] = 3.
        all_keys = {name: group_keys(x, columns) for name, columns in LEVELS}
        for month in np.unique(months):
            tables = self.lot_tables(month)
            for idx in np.flatnonzero(months == month):
                if not (np.isfinite(x.lot_front.iloc[idx]) and x.lot_front.iloc[idx] > 0
                        and np.isfinite(x.lot_depth.iloc[idx]) and x.lot_depth.iloc[idx] > 0):
                    continue
                for level, (name, _) in enumerate(LEVELS):
                    context = tables[name].get(all_keys[name].iloc[idx])
                    if context is None or len(context[0]) < CONFIG['lot_min_n']:
                        continue
                    values = []
                    for arr, value in zip(context, [x.lot_area.iloc[idx], x.lot_front.iloc[idx]]):
                        logv, median = np.log(value), np.median(arr)
                        pct = (np.searchsorted(arr, logv, 'left')+np.searchsorted(arr, logv, 'right'))/(2*len(arr))
                        values.extend([median, logv-median, pct])
                    result.loc[idx] = values+[len(context[0]), level, 1.]
                    break
        return result


PATTERNS = {
    'original': r'\boriginal (?:finishes|kitchen|bath\w*|condition|windows|fixtures|light fixtures)\b|\bfinishes (?:are |remain |mostly |still )*original\b|\bdated\b|\buntouched\b',
    'maintained': r'\bwell[ -](?:maintained|kept|tended|cared[ -]for)\b|\b(?:good|great|excellent|pristine) (?:condition|shape)\b|\bclean and (?:cared for|liveable)\b|\bcared for\b|\bregular mechanical servicing\b',
    'cosmetic': r'\bfresh(?:ly)? paint(?:ed)?\b|\bcosmetic updates?\b|\bnew countertops\b|\brefinished floors?\b|\bnewer laminate\b',
    'kitchen_updated': r'\b(?:new|newer|updated|renovated|refreshed|redone|remodelled|remodeled) kitchen\b|\bkitchen (?:has been |was |is )?(?:updated|renovated|refreshed|redone|remodelled|remodeled)\b',
    'bath_updated': r'\b(?:new|newer|updated|renovated|refreshed|redone|remodelled|remodeled) bathrooms?\b|\bbathrooms? (?:has been |was |is |have been |were |are )?(?:updated|renovated|refreshed|redone|remodelled|remodeled)\b',
    'full_renovation': r'\bfully (?:renovated|redone|remodelled|remodeled)\b|\btop[ -]to[ -]bottom\b|\bgut renovat\w*\b|\bcomplete(?:ly)? renovat\w*\b',
    'work_required': r'\bneeds? (?:\w+ ){0,2}(?:work|updat\w*|replac\w*|renovat\w*|attention)\b|\bfixer\b|\bhandyman\b|\bas[ -]is\b|\bcosmetic overhaul\b|\brequires? (?:\w+ ){0,2}(?:work|waterproof\w*|renovat\w*)\b|\bupdate over time\b|\bmodernize over time\b|\brenovation opportunity\b|\bready for replacement\b',
    'basement_unfinished': r'\bunfinished basement\b|\bbasement (?:is |remains? )?unfinished\b|\bbasement (?:\w+ ){0,3}(?:requires?|needs?) (?:\w+ ){0,2}(?:waterproofing|finishing)\b',
    'basement_finished': r'\bfinished basement\b|\bbasement (?:apartment|suite)\b|\bbasement (?:is |has been )?finished\b',
    'separate_entrance': r'\bseparate (?:side )?(?:entrance|entry)\b|\bwalk[ -]out basement\b',
    'mechanical_older': r'\b(?:older|original) (?:but working )?(?:furnace|windows|roof|appliances)\b|\b(?:furnace|windows|roof|appliances) (?:are |is )?(?:older|original)\b|\b(?:roof|windows) (?:and \w+ )?near the end\b',
    'mechanical_newer': r'\b(?:new|newer|replaced|updated) (?:furnace|windows|roof|electrical|plumbing|appliances)\b',
    'ready': r'\bturnkey\b|\bmove[ -]in ready\b|\bnothing to do\b|\bnothing to .{0,12}unpack\b',
    'update_general': r'\b(?:renovated|updated|refreshed|remodelled|remodeled|redone)\b',
}
NEGATION = re.compile(r'\b(?:not|never|no|without)\s+(?:\w+[ -]+){0,3}$')
PROSPECTIVE = re.compile(r'\b(?:needs?|requires?|could|can|potential(?:ly)?|plans? to|ready to|looking to|opportunity to)\s+(?:\w+[ -]+){0,3}$')
REPLACED = ['condition_full_renovation', 'condition_update', 'condition_needs_work', 'condition_ready', 'feature_basement']


def text_features(homes):
    output = []
    for raw in homes.remarks.fillna('').astype(str):
        text = raw.lower().replace('’', "'").replace("isn't", 'is not').replace("wasn't", 'was not').replace("hasn't", 'has not')
        clauses = re.split(r'[.!?;,\n]|\b(?:but|however|although|though)\b', text)
        flags = {name: 0. for name in PATTERNS}
        negated = 0.
        for clause in clauses:
            for name, pattern in PATTERNS.items():
                for match in re.finditer(pattern, clause):
                    before = clause[max(0, match.start()-65):match.start()]
                    if NEGATION.search(before):
                        if name in ('full_renovation', 'update_general', 'kitchen_updated', 'bath_updated'):
                            negated = 1.
                        continue
                    if name in ('full_renovation', 'update_general', 'kitchen_updated', 'bath_updated', 'basement_finished', 'mechanical_newer') and PROSPECTIVE.search(before):
                        continue
                    flags[name] = 1.
        flags['negated_update'] = negated
        flags['mixed_original_updated'] = flags['original']*max(flags[k] for k in ('cosmetic', 'kitchen_updated', 'bath_updated', 'full_renovation', 'update_general'))
        flags['original_maintained'] = flags['original']*flags['maintained']
        flags['full_and_original'] = flags['original']*flags['full_renovation']
        flags['basement_mixed'] = flags['basement_finished']*flags['basement_unfinished']
        flags['unknown'] = float(not any(flags[k] for k in ('original', 'maintained', 'cosmetic', 'kitchen_updated', 'bath_updated', 'full_renovation', 'work_required', 'ready', 'update_general', 'negated_update')))
        output.append(flags)
    return pd.DataFrame(output).add_prefix('state_')


def assemble(base, history, land, text, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown predeclared variant')
    x = base.reset_index(drop=True).copy()
    if variant in ('history_only', 'all_three'):
        x = pd.concat([x, history.reset_index(drop=True)], axis=1)
    if variant in ('land_only', 'all_three'):
        x = pd.concat([x, land.reset_index(drop=True)], axis=1)
    if variant in ('text_only', 'all_three'):
        x = x.drop(columns=REPLACED)
        states = text.reset_index(drop=True)
        x = pd.concat([x, states], axis=1)
        x['condition_full_renovation'] = states.state_full_renovation
        x['condition_update'] = states[['state_update_general', 'state_cosmetic', 'state_kitchen_updated', 'state_bath_updated', 'state_full_renovation']].max(axis=1)
        x['condition_needs_work'] = states[['state_original', 'state_work_required']].max(axis=1)
        x['condition_ready'] = states.state_ready
        x['feature_basement'] = states[['state_basement_finished', 'state_separate_entrance']].max(axis=1)
    if x.columns.duplicated().any():
        raise ValueError('Duplicate features')
    return x.replace([np.inf, -np.inf], np.nan)
