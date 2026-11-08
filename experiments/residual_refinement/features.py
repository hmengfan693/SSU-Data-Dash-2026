"""Correct same-location historical priors; lexical condition features only."""
import re
import numpy as np
import pandas as pd
from experiments.plan_a.features import HistoryFeatures, property_features, group_keys

TEXT_PATTERNS = {
    'text_original': r'original (?:finishes|kitchen|bath|condition|windows|fixtures)|finishes.{0,20}(?:original|tired)|dated|untouched',
    'text_maintained': r'well[ -](?:maintained|kept|tended)|(?:clean|good|great|excellent|pristine) (?:condition|shape)|clean and liveable',
    'text_cosmetic': r'fresh(?:ly)? paint|freshly painted|cosmetic|new countertops|refinish|newer laminate',
    'text_kitchen_bath': r'(?:kitchen|bath\w*).{0,35}(?:renovat|updat|redone)|(?:new|updated|renovated) (?:kitchen|bath)',
    'text_major_renovation': r'fully (?:renovated|redone|remodelled|remodeled)|top to bottom|gut renovat|complete(?:ly)? renovat|updated electrical and plumbing',
    'text_work_required': r'needs? (?:work|updat|replac|renovat)|fixer|handyman|as[ -]is|cosmetic overhaul|requires? (?:work|waterproof|renovat)|update over time|renovation opportunity',
    'text_basement_unfinished': r'unfinished basement|basement.{0,45}(?:requires|needs|unfinished|waterproof)',
    'text_basement_finished': r'\bfinished basement\b|\bbasement (?:apartment|suite)\b',
    'text_separate_entrance': r'separate (?:side )?entrance|separate entry|walk[ -]out basement',
    'text_rebuild': r'rebuild|tear[ -]down|building lot|redevelop|addition|extension|expanded',
    'text_mechanical_older': r'older (?:but working )?(?:furnace|windows|roof)|original (?:furnace|windows|roof)|(?:furnace|windows|roof).{0,20}(?:older|original|need)',
    'text_mechanical_newer': r'(?:new|newer|replaced|updated) (?:furnace|windows|roof|electrical|plumbing)',
}


def condition_features(homes):
    text = homes.remarks.fillna('').astype(str).str.lower().reset_index(drop=True)
    text = text.str.replace(r'\b(?:not|never|no)\s+(?:(?:\w+)\s+){0,3}(?:renovated|updated|remodelled|remodeled|redone)\b', '', regex=True)
    x = pd.DataFrame({name: text.str.contains(p, regex=True, flags=re.I).astype(float)
                      for name, p in TEXT_PATTERNS.items()})
    x['text_mixed_original_updated'] = x.text_original * x[['text_cosmetic', 'text_kitchen_bath', 'text_major_renovation']].max(axis=1)
    x['text_condition_unknown'] = x[['text_original','text_maintained','text_cosmetic','text_kitchen_bath','text_major_renovation','text_work_required']].sum(axis=1).eq(0).astype(float)
    return x


class LocationHistory(HistoryFeatures):
    """Recompute location sums with community-type rather than market-type prior.

    Actual valid ppsf counts are used, including for incomplete area inputs.
    All windows remain [min(listed, cutoff)-365d, min(listed, cutoff)).
    """
    def corrected(self, homes, base=None):
        x = (self.transform(homes) if base is None else base).reset_index(drop=True).copy()
        q = np.minimum(pd.to_datetime(homes.date_listed).to_numpy(dtype='datetime64[ns]'), self.cutoff.to_datetime64())
        keys = group_keys(x, ['location_group','house_type_name'])
        sums, counts = np.zeros((len(x),2)), np.zeros((len(x),2))
        for key, pos in keys.groupby(keys, sort=False).groups.items():
            t = self.tables['location_type'].get(key)
            if t is None:
                continue
            idx = np.asarray(pos)
            dates, price, ppsf, valid = t
            r = np.searchsorted(dates, q[idx], side='left')
            l = np.searchsorted(dates, q[idx]-np.timedelta64(365,'D'), side='left')
            sums[idx,0], sums[idx,1] = price[r]-price[l], ppsf[r]-ppsf[l]
            counts[idx,0], counts[idx,1] = r-l, valid[r]-valid[l]
        for j, suffix in enumerate(['log_price','log_ppsf']):
            parent = x[f'hist_community_type_365_{suffix}'].to_numpy()
            weight = np.where(np.isfinite(parent),20.,0.)
            denom = counts[:,j]+weight
            x[f'hist_location_type_365_{suffix}'] = np.divide(sums[:,j]+weight*np.nan_to_num(parent),denom,
                out=np.full(len(x),np.nan),where=denom>0)
        x['location_history_no_sales'] = counts[:,0].eq(0) if isinstance(counts[:,0],pd.Series) else (counts[:,0]==0).astype(float)
        return x


def enrich(x, homes):
    return pd.concat([x.reset_index(drop=True), condition_features(homes)],axis=1)
