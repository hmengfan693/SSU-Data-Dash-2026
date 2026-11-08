"""Fit fixed ablations, select on 2025, evaluate frozen routes, refit and replay."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import data
import check_submission
from experiments.plan_a.features import FORBIDDEN
from experiments.plan_a.models import PARAMS
from experiments.plan_a.run_experiment import metrics
from experiments.property_groups.models import GroupModel, group_labels
from .features import LocationHistory, enrich
from .models import ResidualCalibrator, RIDGE_ALPHA, MAX_LOG_ADJUSTMENT
from .selection import DEV, TYPES, LAND_CANDIDATES, COOP_CANDIDATES, choose, route

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OLD=REPO/'experiments/property_groups/artifacts/run_2026-10-17_v1'
LATEST=REPO/'experiments/land_refinement/artifacts/run_2026-10-18_v2'
LAND=['Detached','Semi-Detached','Freehold Townhouse']
APARTMENT=['Condo Apt','Co-Op Apt']
LOG=logging.getLogger(__name__)
SPECS={
    'shared_original':('lightgbm','land','base','global'),
    'location':('lightgbm','land','location','global'),
    'location_catboost':('catboost','land','location','global'),
    'text_catboost':('catboost','land','text','global'),
    'apartment_base':('catboost','apartment','base','apartments'),
    'apartment_text':('catboost','apartment','apartment_text','apartments'),
    'legacy_detached':('lightgbm','Detached','base','detached'),
    'legacy_attached':('lightgbm','attached','base','attached_freehold'),
    'legacy_condo_townhouse':('lightgbm','Condo Townhouse','base','condo_townhouse')}
OUTER_MODELS=['shared_original','location','location_catboost','text_catboost','apartment_base','apartment_text']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(ZoneInfo('America/Toronto')).isoformat()
def write_json(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def family_mask(homes,family):
    if family=='land':return homes.house_type_name.isin(LAND).to_numpy()
    if family=='apartment':return homes.house_type_name.isin(APARTMENT).to_numpy()
    if family=='attached':return homes.house_type_name.isin(LAND[1:]).to_numpy()
    return homes.house_type_name.eq(family).to_numpy()


def rows(homes,predicted,cutoff,candidate,protocol):
    z=homes[['ml_num','house_type_name','community']].reset_index(drop=True).copy()
    if 'sold_price' in homes:z['sold_price']=homes.sold_price.to_numpy()
    z['month']=pd.to_datetime(homes.date_sold if 'date_sold' in homes else homes.date_listed).dt.strftime('%Y-%m').to_numpy()
    z['predicted_price']=predicted
    z['cutoff_exclusive']=str(pd.Timestamp(cutoff).date())
    z['candidate']=candidate;z['protocol']=protocol
    return z


def variant(base,location,homes,name):
    if name=='base':return base
    if name=='location':return location
    return enrich(location if name=='text' else base,homes)


class Experiment:
    def __init__(self,raw,test,out):
        self.raw,self.test,self.out=raw,test,Path(out)
        self.feature_audits={};self.model_audits={}
        self.old_dev=pd.read_csv(OLD/'development_routed_predictions.csv')
        self.old_check=pd.read_csv(OLD/'check_routed_predictions.csv')
        for archived in [self.old_dev,self.old_check]:
            truth=archived[['ml_num','sold_price']].merge(raw[['ml_num','sold_price']],on='ml_num',suffixes=('_old','_raw'),validate='one_to_one')
            if len(truth)!=len(archived) or not np.array_equal(truth.sold_price_old,truth.sold_price_raw):
                raise ValueError('Archived baseline target/ID mismatch')

    def split(self,cutoff,end=None,final=False):
        train=self.raw.loc[self.raw.date_sold<pd.Timestamp(cutoff)].sort_values(['date_sold','ml_num']).reset_index(drop=True)
        homes=(self.test.copy() if final else self.raw.loc[(self.raw.date_sold>=pd.Timestamp(cutoff))&(self.raw.date_sold<pd.Timestamp(end))].copy()).reset_index(drop=True)
        if train.empty or homes.empty or set(train.ml_num)&set(homes.ml_num):raise ValueError('Invalid split')
        return train,homes

    def prepare(self,cutoff,end=None,final=False):
        train,homes=self.split(cutoff,end,final)
        folder=self.out/'bases'/str(pd.Timestamp(cutoff).date())
        if (folder/'feature_audit.json').exists():
            arrays=[pd.read_parquet(folder/(s+'.parquet')) for s in ['train_base','train_location','valid_base','valid_location']]
            return train,homes,folder,arrays
        folder.mkdir(parents=True)
        LOG.info('[features] %s train=%s predict=%s',cutoff,len(train),len(homes))
        history=LocationHistory(train,cutoff)
        hidden=data.hide_answers(homes,'off_market')
        btrain,bvalid=history.transform(train),history.transform(hidden)
        ltrain,lvalid=history.corrected(train,btrain),history.corrected(hidden,bvalid)
        arrays=[btrain,ltrain,bvalid,lvalid]
        if FORBIDDEN.intersection(set().union(*(set(x.columns) for x in arrays))):raise ValueError('Forbidden model input')
        for label,x in zip(['train_base','train_location','valid_base','valid_location'],arrays):x.to_parquet(folder/(label+'.parquet'),index=False)
        train[['ml_num','date_listed','date_sold']].to_csv(folder/'train_ids.csv',index=False)
        homes[['ml_num','date_listed']].to_csv(folder/'predict_ids.csv',index=False)
        audit={'cutoff':cutoff,'end':end,'final':final,'train_n':len(train),'predict_n':len(homes),
            'max_train_sale':str(train.date_sold.max()),'history_query':'[min(date_listed,cutoff)-window, min(date_listed,cutoff))',
            'no_outer_targets_in_features':True,'base_features':list(btrain),'location_features':list(ltrain)}
        write_json(folder/'feature_audit.json',audit)
        self.feature_audits[cutoff]=audit
        return train,homes,folder,arrays

    def get_window(self,cutoff,end=None,final=False,needed=('location','apartment_base')):
        train,homes,folder,arrays=self.prepare(cutoff,end,final)
        bt,lt,bv,lv=arrays
        predictions={}
        for name in needed:
            algorithm,family,v,scope=SPECS[name]
            model_folder=folder/name
            output=model_folder/'predictions.csv'
            if output.exists():
                predictions[name]=pd.read_csv(output)
                continue
            xtrain=variant(bt,lt,train,v);xvalid=variant(bv,lv,homes,v)
            tm,vm=family_mask(train,family),family_mask(homes,family)
            if not tm.any() or not vm.any():raise ValueError('Empty family')
            start=time.monotonic()
            LOG.info('[fit] %s %-24s n=%s features=%s',cutoff,name,tm.sum(),len(xtrain.columns))
            model=GroupModel(scope,algorithm).fit(xtrain.loc[tm],train.sold_price.loc[tm])
            price=model.predict(xvalid.loc[vm])
            model.save_bundle(model_folder,cutoff,int(tm.sum()))
            loaded=GroupModel.load_bundle(model_folder)
            if not np.array_equal(price,loaded.predict(xvalid.loc[vm])):raise ValueError('Model reload mismatch')
            p=rows(homes.loc[vm],price,cutoff,name,'final' if final else 'base')
            p.to_csv(output,index=False)
            importance=model.model.get_feature_importance() if algorithm=='catboost' else model.model.booster_.feature_importance(importance_type='gain')
            pd.DataFrame({'feature':model.feature_names,'importance':importance}).sort_values('importance',ascending=False).to_csv(model_folder/'feature_importance.csv',index=False)
            audit={'cutoff':cutoff,'model':name,'training_n':int(tm.sum()),'evaluation_n':int(vm.sum()),'max_train_sale':str(train.date_sold.loc[tm].max()),
                   'reload_exact':True,'fit_predict_seconds':time.monotonic()-start}
            write_json(model_folder/'audit.json',audit)
            self.model_audits[f'{cutoff}/{name}']=audit
            predictions[name]=p
            LOG.info('[done] %s %-24s seconds=%.1f',cutoff,name,audit['fit_predict_seconds'])
        return homes,predictions

    def calibrators(self,cutoff):
        out=self.out/'calibrators'/cutoff
        if (out/'land'/'audit.json').exists() and (out/'apartment'/'audit.json').exists():
            return {family:ResidualCalibrator.load(out/family) for family in ['land','apartment']}
        oof=[]
        for months in [8,4]:
            inner=pd.Timestamp(cutoff)-pd.DateOffset(months=months)
            end=inner+pd.DateOffset(months=4)
            _,p=self.get_window(str(inner.date()),str(end.date()))
            oof.append(pd.concat([p['location'].assign(family='land'),p['apartment_base'].assign(family='apartment')],ignore_index=True))
        oof=pd.concat(oof,ignore_index=True)
        if not (pd.to_datetime(oof.month+'-01')<pd.Timestamp(cutoff)).all():raise ValueError('OOF future month')
        out.mkdir(parents=True,exist_ok=True)
        oof.to_csv(out/'oof_predictions.csv',index=False)
        result={}
        for family,g in oof.groupby('family',sort=True):
            h=g[['ml_num']].merge(self.raw,on='ml_num',validate='one_to_one')
            if not np.array_equal(h.ml_num,g.ml_num):raise ValueError('OOF alignment')
            model=ResidualCalibrator(cutoff).fit(h,g.predicted_price.to_numpy(),h.sold_price.to_numpy(),g.cutoff_exclusive)
            model.save(out/family)
            if not np.array_equal(model.predict(h,g.predicted_price.to_numpy()),ResidualCalibrator.load(out/family).predict(h,g.predicted_price.to_numpy())):
                raise ValueError('Calibrator reload mismatch')
            result[family]=model
        LOG.info('[calibration] %s land=%s apartment=%s earlier OOF rows',cutoff,result['land'].audit['oof_n'],result['apartment'].audit['oof_n'])
        return result

    def baseline(self,homes,cutoff,protocol,predictions=None):
        if protocol!='final':
            archived=self.old_dev if protocol=='development' else self.old_check
            p=archived.loc[archived.cutoff_exclusive.eq(cutoff)].copy()
            if set(p.ml_num)!=set(homes.ml_num) or p.ml_num.duplicated().any():raise ValueError('Baseline cohort mismatch')
            p=p[['ml_num','house_type_name','community','sold_price','month','predicted_price','cutoff_exclusive']]
            p['protocol']=protocol;p['candidate']='previous'
            return p
        p=pd.concat([predictions[n] for n in ['apartment_base','legacy_detached','legacy_attached','legacy_condo_townhouse']],ignore_index=True)
        p['candidate']='previous';p['protocol']='final'
        if len(p)!=len(homes) or p.ml_num.duplicated().any():raise ValueError('Final baseline coverage')
        return p

    def evaluate(self,cutoff,end,protocol):
        final=protocol=='final'
        needed=OUTER_MODELS+(['legacy_detached','legacy_attached','legacy_condo_townhouse'] if final else [])
        homes,p=self.get_window(cutoff,end,final,needed)
        baseline=self.baseline(homes,cutoff,protocol,p)
        if not final:
            b=baseline.loc[baseline.house_type_name.isin(APARTMENT)].set_index('ml_num').predicted_price
            a=p['apartment_base'].set_index('ml_num').predicted_price.reindex(b.index)
            if not np.array_equal(a,b):raise ValueError('Retrained apartment control differs from archived control')
        result=[baseline]
        for name in ['shared_original','location','location_catboost','text_catboost']:
            result.append(p[name].assign(protocol=protocol,candidate=name))
        result.append(p['apartment_text'].loc[p['apartment_text'].house_type_name.eq('Co-Op Apt')].assign(protocol=protocol,candidate='apartment_text'))
        cal=self.calibrators(cutoff)
        adjustments=[]
        for family,name,candidate in [('land','location','location_ridge'),('apartment','apartment_base','apartment_ridge')]:
            z=p[name].copy()
            h=z[['ml_num']].merge(homes,on='ml_num',validate='one_to_one')
            pred=z.predicted_price.to_numpy()
            adj=cal[family].adjustment(data.hide_answers(h,'off_market'),pred)
            adjustments.append(pd.DataFrame({'ml_num':z.ml_num,'family':family,'base_price':pred,'log_adjustment':adj,'adjustment_pct':100*np.expm1(adj)}))
            z['predicted_price']=cal[family].predict(data.hide_answers(h,'off_market'),pred)
            z['candidate']=candidate;z['protocol']=protocol
            result.append(z if family=='land' else z.loc[z.house_type_name.eq('Co-Op Apt')])
        folder=self.out/protocol/cutoff;folder.mkdir(parents=True,exist_ok=False)
        pd.concat(adjustments,ignore_index=True).to_csv(folder/'residual_adjustments.csv',index=False)
        all_rows=pd.concat(result,ignore_index=True)
        all_rows.to_csv(folder/'all_candidates.csv',index=False)
        if not final:
            for kind in TYPES:
                g=all_rows.loc[all_rows.house_type_name.eq(kind)]
                text=', '.join(f'{c}={metrics(z)["mdape"]:.2f}%' for c,z in g.groupby('candidate'))
                LOG.info('[score] %s %s %s',cutoff,kind,text)
        return all_rows


def breakdown(frame,keys):
    output=[]
    for key,g in frame.groupby(keys,observed=True,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        output.append({**dict(zip(keys,key)),**metrics(g)})
    return output


def score_bundle(frame):
    return {'overall':metrics(frame),'by_type':breakdown(frame,['house_type_name']),
            'by_month':breakdown(frame,['month']),'by_type_month':breakdown(frame,['house_type_name','month']),
            'by_window':breakdown(frame,['cutoff_exclusive'])}


def infer_saved(raw,test,run):
    """Fresh feature reconstruction and serialized models only, no fit calls."""
    run=Path(run);cutoff='2026-06-01'
    train=raw.loc[raw.date_sold<pd.Timestamp(cutoff)].sort_values(['date_sold','ml_num']).reset_index(drop=True)
    history=LocationHistory(train,cutoff)
    hidden=data.hide_answers(test,'off_market').reset_index(drop=True)
    b=history.transform(hidden);l=history.corrected(hidden,b)
    preds={}
    for name in OUTER_MODELS+['legacy_detached','legacy_attached','legacy_condo_townhouse']:
        _,family,v,_=SPECS[name]
        mask=family_mask(hidden,family)
        x=variant(b,l,hidden,v)
        price=GroupModel.load_bundle(run/'bases'/cutoff/name).predict(x.loc[mask])
        preds[name]=rows(hidden.loc[mask],price,cutoff,name,'final')
    base=pd.concat([preds[n] for n in ['apartment_base','legacy_detached','legacy_attached','legacy_condo_townhouse']],ignore_index=True).assign(candidate='previous')
    parts=[base,*[preds[n] for n in ['shared_original','location','location_catboost','text_catboost']],
           preds['apartment_text'].loc[preds['apartment_text'].house_type_name.eq('Co-Op Apt')]]
    for family,name,candidate in [('land','location','location_ridge'),('apartment','apartment_base','apartment_ridge')]:
        z=preds[name].copy();h=z[['ml_num']].merge(hidden,on='ml_num',validate='one_to_one')
        z['predicted_price']=ResidualCalibrator.load(run/'calibrators'/cutoff/family).predict(h,z.predicted_price.to_numpy())
        z['candidate']=candidate
        parts.append(z if family=='land' else z.loc[z.house_type_name.eq('Co-Op Apt')])
    candidates=pd.concat(parts,ignore_index=True)
    choices=json.loads((run/'selection.json').read_text())['choices']
    selected=route(candidates,choices).set_index('ml_num').reindex(hidden.ml_num).reset_index()
    return selected[['ml_num','predicted_price']],candidates


def report(out,scores,selection,candidate_scores,verification):
    lines=['# 房型残差校正与文本增强：重新训练结果','','仅 Off-market。标签为 log(sold_price)，挂牌价不作为输入。所有模型固定 600 轮，无外层验证早停。',
           '','## 已完成的训练与选择','','位置候选只把精确位置历史的父级从全市场同房型改成同社区同房型，并保留无成交标记。共享原特征 LightGBM 单独作为人口范围改变的对照。',
           '','残差层采用固定 alpha=100 的岭回归，输入包含房型/社区/区域、基准估价、面积、土地、卫生间、挂牌日期趋势及季节项。每个外层窗口只使用其前八个月的两个四个月滚动预测误差；没有使用主模型训练内残差。调整上限为 ±log(1.15)，上下价格比例分别为 1.15 和 1/1.15。',
           '','文本增强使用 14 项预先固定的房况词句特征，区分原装、维护、局部/全面装修、维修需求、地下室、独立入口和设备新旧。没有用大模型猜测缺失标签，没有从文本提取价格或任意数字。CatBoost 的位置版与文本版用相同训练人群和参数作对照。',
           '','2025 年三个窗口只按 MdAPE 选择，普通房型须合并改善至少 0.10 个百分点且至少两个窗口改善。合作公寓只用共享公寓模型，须至少 100 条开发样本、三个窗口都改善且合并改善至少 0.25 个百分点。整体还须改善至少 0.05 个百分点且至少两个窗口改善，否则退回原方案。',
           '',f'最终按开发规则接受新路由：{selection["accepted"]}。选择冻结时间：{selection["frozen_at"]}。',
           '','## 相同 2026 年 2—5 月房屋上的结果','','| 房型 | 选用候选 | 套数 | 原 MdAPE | 新 MdAPE | 新中位偏差 |','|---|---|---:|---:|---:|---:|']
    before={s['house_type_name']:s for s in scores['previous']['by_type']}
    for s in scores['selected']['by_type']:
        k=s['house_type_name'];b=before[k]
        lines.append(f'| {k} | {selection["choices"].get(k,"previous")} | {s["n"]:,} | {b["mdape"]:.2f}% | {s["mdape"]:.2f}% | {s["median_signed_error"]:+.2f}% |')
    lines+=['','| 整体方案 | MdAPE | MAPE |','|---|---:|---:|']
    for name,s in scores.items():
        z=s['overall'];lines.append(f'| {name} | {z["mdape"]:.3f}% | {z["mape"]:.3f}% |')
    lines+=['','latest_land_refinement 是之前几何/相似成交版本的存档对照，只作如实比较，不据此重选本轮路由。整体 MdAPE 由逐套预测合并计算，不能据此声称每个房型都改善。',
            '','## 所有预定候选','','| 房型 | 候选 | 开发 MdAPE | 后续 MdAPE | 后续偏差 | 选用 |','|---|---|---:|---:|---:|---|']
    dev={(r['house_type_name'],r['candidate']):r for r in selection['candidate_scores']}
    for s in candidate_scores:
        k,c=s['house_type_name'],s['candidate'];d=dev.get((k,c))
        if d:lines.append(f'| {k} | {c} | {d["mdape"]:.2f}% | {s["mdape"]:.2f}% | {s["median_signed_error"]:+.2f}% | {"是" if selection["choices"].get(k,"previous")==c else ""} |')
    lines+=['','previous=原分组模型；shared_original=有土地房型共享 LightGBM、原特征；location=共享 LightGBM、修正位置历史；location_ridge=位置版加岭回归校正；location_catboost=位置版 CatBoost；text_catboost=位置版 CatBoost 加房况特征；apartment_ridge=共享公寓 CatBoost 加校正；apartment_text=共享公寓 CatBoost 加房况特征。',
            '','## 核验与交付','',f'截至 2026 年 5 月重训后已生成 {verification["final_n"]:,} 行预测；所有最终主模型、岭回归校正器均保存重载，独立从原始数据重建特征重放一致，官方格式校验通过。原始输入、已有模型/预测/提交哈希保持不变。',
            '','最终文件：final_predictions/submission_off_market.csv。各房型路由：final_predictions/prediction_routes.csv。复核详情见 verification.json、metrics.json、source_manifest.json、selection.json 与各模型 audit.json。',
            '','## 解释边界','','2025 年开发期和 2026 年 2—5 月均已被项目多次查看，属于累计开发回测，不是新的独立测试。冻结后即使回测退步也不事后换路由或调校正系数。6—9 月没有公开标签，无法报告隐藏测试误差。',
            '','模型和历史池在每个四个月窗口内冻结；模型训练按成交日期截止，历史查询早于 min(挂牌日,截止日)，不声称在每套房挂牌当天重训。模型学到的文本和社区关联不代表因果关系。所有历史统计、标准化、类别词表和残差训练均仅依赖对应窗口的过去数据。',
            '','本轮最终比较的是整体方案；位置变动、共享训练、算法与文本的差异可通过各对照分别评估。没有证明一种方法对未来所有市场阶段有效。','']
    (out/'训练结果与分析.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(out/'run.log')])
    started=now();os.environ['DATA_DIR']=str(REPO.parent/'data')
    inputs=[data.find_file('listings'),data.find_file('off_market')]
    protected_files=[*inputs,*OLD.rglob('*'),*LATEST.rglob('*'),*list((REPO/'submissions').glob('*.csv'))]
    protected={str(p.resolve()):sha(p) for p in protected_files if p.is_file()}
    source=[*HERE.glob('*.py'),REPO/'experiments/plan_a/features.py',REPO/'experiments/plan_a/models.py',REPO/'experiments/property_groups/models.py',REPO/'tests/test_residual_refinement.py']
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in source:shutil.copy2(p,snapshot/(str(p.relative_to(REPO)).replace('/','__')))
    config={'started_at':started,'outer_windows':DEV,'check_cutoff':'2026-02-01','check_end':'2026-06-01',
        'final_cutoff':'2026-06-01','inner_lookback_months':[8,4],'inner_horizon_months':4,
        'target':'log(sold_price)','params':PARAMS,'ridge_alpha':RIDGE_ALPHA,'max_abs_log_adjustment':MAX_LOG_ADJUSTMENT,
        'land_candidates':LAND_CANDIDATES,'co_op_candidates':COOP_CANDIDATES,
        'baseline':'property_groups frozen routed models','selection_metric':'MdAPE only',
        'evaluation_is_previously_examined_development':True,
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','catboost','scikit-learn','scipy','joblib']},
        'source_sha256':{str(p):sha(p) for p in source},'protected_sha256':protected}
    write_json(out/'source_manifest.json',config)
    raw,test=data.load('listings'),data.load('off_market')
    if raw.ml_num.duplicated().any() or test.ml_num.duplicated().any() or not raw.sold_price.gt(0).all():raise ValueError('Invalid raw data')
    exp=Experiment(raw,test,out)
    dev=[]
    for cutoff in DEV:
        end=str((pd.Timestamp(cutoff)+pd.DateOffset(months=4)).date())
        dev.append(exp.evaluate(cutoff,end,'development'))
    dev=pd.concat(dev,ignore_index=True);dev.to_csv(out/'development_candidates.csv',index=False)
    selection=choose(dev);selection['frozen_at']=now()
    write_json(out/'selection.json',selection)
    frozen_hash=sha(out/'selection.json')
    LOG.info('[frozen selection] %s accepted=%s',selection['choices'],selection['accepted'])
    check=exp.evaluate('2026-02-01','2026-06-01','check')
    selected=route(check,selection['choices']);previous=check.loc[check.candidate.eq('previous')]
    selected.to_csv(out/'check_selected_predictions.csv',index=False)
    check.to_csv(out/'check_candidates.csv',index=False)
    latest=pd.read_csv(LATEST/'check_selected_predictions.csv')
    if set(latest.ml_num)!=set(previous.ml_num):raise ValueError('Latest comparison cohort mismatch')
    scores={'previous':score_bundle(previous),'selected':score_bundle(selected),'latest_land_refinement':score_bundle(latest)}
    write_json(out/'metrics.json',scores)
    candidate_scores=breakdown(check.loc[check.house_type_name.isin(TYPES)],['house_type_name','candidate'])
    pd.DataFrame(candidate_scores).to_csv(out/'candidate_scores.csv',index=False)
    pd.DataFrame(scores['selected']['by_type_month']).to_csv(out/'property_type_month_errors.csv',index=False)
    final=exp.evaluate('2026-06-01',None,'final')
    final_route=route(final,selection['choices']).set_index('ml_num').reindex(test.ml_num).reset_index()
    folder=out/'final_predictions';folder.mkdir()
    submission=final_route[['ml_num','predicted_price']]
    submission.to_csv(folder/'submission_off_market.csv',index=False)
    final_route.to_csv(folder/'prediction_routes.csv',index=False)
    errors=check_submission.check_file(folder/'submission_off_market.csv',test.ml_num)
    if errors:raise ValueError(errors)
    replay,replay_candidates=infer_saved(raw,test,out)
    replay.to_csv(folder/'replay.csv',index=False)
    if not submission.equals(replay):raise ValueError('Final raw-feature replay mismatch')
    a=final[['ml_num','candidate','predicted_price']].sort_values(['candidate','ml_num']).reset_index(drop=True)
    b=replay_candidates[['ml_num','candidate','predicted_price']].sort_values(['candidate','ml_num']).reset_index(drop=True)
    if not a.equals(b):raise ValueError('Final unselected candidate replay mismatch')
    if sha(out/'selection.json')!=frozen_hash:raise ValueError('Selection changed after freeze')
    changed=[p for p,h in protected.items() if sha(p)!=h]
    if changed:raise ValueError(f'Protected files changed: {changed}')
    verification={'finished_at':now(),'final_n':len(submission),'final_submission_sha256':sha(folder/'submission_off_market.csv'),
        'official_format_errors':errors,'fresh_feature_replay_exact':True,'all_final_candidates_replay_exact':True,
        'selection_sha256':frozen_hash,'selection_unchanged_after_check':True,'protected_file_n':len(protected),'protected_files_changed':[],
        'model_audit_n':len(list((out/'bases').glob('*/*/audit.json'))),'calibrator_n':len(list((out/'calibrators').glob('*/*/audit.json')))}
    write_json(out/'verification.json',verification)
    report(out,scores,selection,candidate_scores,verification)
    LOG.info('[complete] previous MdAPE=%.3f%% selected=%.3f%% final=%s',scores['previous']['overall']['mdape'],scores['selected']['overall']['mdape'],folder/'submission_off_market.csv')


if __name__=='__main__':main()
