"""Fixed-alpha Ridge learns past out-of-time log-price errors, never fit errors."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from experiments.plan_a.features import property_features

RIDGE_ALPHA = 100.
MAX_LOG_ADJUSTMENT = float(np.log(1.15))
CAT = ['kind','community_kind','municipality_kind']
NUM = ['base_log','log_area','log_lot_area','washrooms','time','month_sin','month_cos']


def residual_design(homes, prediction, cutoff):
    h = homes.reset_index(drop=True)
    x = property_features(h)
    d = pd.DataFrame({'kind':h.house_type_name.astype(str),
        'community_kind':h.community.astype(str)+'|'+h.house_type_name.astype(str),
        'municipality_kind':h.municipality.astype(str)+'|'+h.house_type_name.astype(str),
        'base_log':np.log(prediction), 'log_area':x.log_area, 'log_lot_area':x.log_lot_area,
        'washrooms':x.washrooms,
        'time':(pd.to_datetime(h.date_listed)-pd.Timestamp(cutoff)).dt.days/365.25,
        'month_sin':x.month_sin, 'month_cos':x.month_cos})
    return d


class ResidualCalibrator:
    def __init__(self, cutoff):
        self.cutoff = str(pd.Timestamp(cutoff).date())
        self.pipeline = Pipeline([('design', ColumnTransformer([
            ('category',OneHotEncoder(handle_unknown='ignore'),CAT),
            ('numeric',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True)),
                                 ('scale',StandardScaler())]),NUM)])),
            ('ridge',Ridge(alpha=RIDGE_ALPHA,solver='lsqr'))])

    def fit(self, homes, predicted, sold, source_cutoffs):
        h = homes.reset_index(drop=True)
        cutoffs = pd.to_datetime(pd.Series(source_cutoffs).reset_index(drop=True))
        dates = pd.to_datetime(h.date_sold)
        outer = pd.Timestamp(self.cutoff)
        if not (dates < outer).all() or not (cutoffs < outer).all() or not (dates >= cutoffs).all():
            raise ValueError('Residual calibration requires strictly earlier out-of-time predictions')
        if h.ml_num.duplicated().any() or len(h)!=len(predicted):
            raise ValueError('Residual calibration IDs must be unique and aligned')
        if not (np.asarray(predicted)>0).all() or not (np.asarray(sold)>0).all():
            raise ValueError('Residual prices must be positive')
        self.pipeline.fit(residual_design(h,predicted,self.cutoff),np.log(np.asarray(sold)/np.asarray(predicted)))
        self.audit = {'cutoff':self.cutoff,'oof_n':len(h),'oof_type_counts':{str(k):int(v) for k,v in h.house_type_name.value_counts().items()},
            'max_calibration_sale':str(dates.max()),'inner_cutoffs':sorted(cutoffs.dt.strftime('%Y-%m-%d').unique()),
            'ridge_alpha':RIDGE_ALPHA,'max_abs_log_adjustment':MAX_LOG_ADJUSTMENT,
            'calibration_uses_in_sample_residuals':False,'outer_evaluation_labels_used':False}
        return self

    def adjustment(self, homes, predicted):
        return np.clip(self.pipeline.predict(residual_design(homes,predicted,self.cutoff)),-MAX_LOG_ADJUSTMENT,MAX_LOG_ADJUSTMENT)

    def predict(self, homes, predicted):
        return np.round(np.asarray(predicted)*np.exp(self.adjustment(homes,predicted)))

    def save(self, folder):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
        joblib.dump(self,folder/'calibrator.joblib')
        (folder/'audit.json').write_text(json.dumps(self.audit,ensure_ascii=False,indent=2)+'\n')

    @classmethod
    def load(cls, folder):
        return joblib.load(Path(folder)/'calibrator.joblib')
