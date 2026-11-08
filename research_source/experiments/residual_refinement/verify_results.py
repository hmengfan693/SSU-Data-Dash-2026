"""Independent numerical, temporal and artifact audit; no estimator fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(p):
    e=(p.predicted_price.to_numpy()-p.sold_price.to_numpy())/p.sold_price.to_numpy()*100
    return {'n':len(p),'mdape':float(np.median(np.abs(e))),'mape':float(np.mean(np.abs(e))),
            'median_signed_error':float(np.median(e)),
            'over_30_pct':float(np.mean(np.abs(100*(p.predicted_price.to_numpy()/p.sold_price.to_numpy()-1))>30)*100)}


def compare(actual,expected):
    for name,value in expected.items():
        if name in ['n','mdape','mape','median_signed_error','over_30_pct']:
            assert np.isclose(actual[name],value,atol=1e-9,rtol=1e-11),(name,actual[name],value)


def independent_selection(rows):
    choices={};count=0
    cuts=sorted(rows.cutoff_exclusive.unique())
    for kind in ['Detached','Semi-Detached','Freehold Townhouse','Co-Op Apt']:
        g=rows.loc[rows.house_type_name.eq(kind)];ref=g[g.candidate.eq('previous')]
        eligible=[]
        for candidate,z in g.groupby('candidate'):
            count+=1
            assert set(zip(z.cutoff_exclusive,z.ml_num))==set(zip(ref.cutoff_exclusive,ref.ml_num))
            wins=sum(score(z[z.cutoff_exclusive.eq(c)])['mdape']<score(ref[ref.cutoff_exclusive.eq(c)])['mdape'] for c in cuts)
            n,w,gain=(100,3,.25) if kind=='Co-Op Apt' else (200,2,.10)
            if candidate!='previous' and len(z)>=n and wins>=w and score(ref)['mdape']-score(z)['mdape']>=gain:
                eligible.append((score(z)['mdape'],candidate))
        choices[kind]=min(eligible)[1] if eligible else 'previous'
    proposed=pd.concat([g[g.candidate.eq(choices.get(k,'previous'))] for k,g in rows.groupby('house_type_name')],ignore_index=True)
    ref=rows[rows.candidate.eq('previous')]
    wins=sum(score(proposed[proposed.cutoff_exclusive.eq(c)])['mdape']<score(ref[ref.cutoff_exclusive.eq(c)])['mdape'] for c in cuts)
    accepted=score(ref)['mdape']-score(proposed)['mdape']>=.05 and wins>=2
    return choices if accepted else {k:'previous' for k in choices},bool(accepted),count


def direct_location(raw,home,cutoff):
    q=min(pd.Timestamp(home.date_listed),pd.Timestamp(cutoff))
    h=raw.loc[(raw.date_sold<q)&(raw.date_sold>=q-pd.Timedelta(days=365))]
    typ=h.loc[h.house_type_name.eq(home.house_type_name)]
    community=typ.loc[typ.community.eq(home.community)]
    location=typ.loc[typ.location_group.eq(home.location_group)]
    result={}
    for field,suffix in [('log_price','log_price'),('log_ppsf','log_ppsf')]:
        def mean(g):return g[field].mean()
        market=mean(h)
        def smooth(g,parent,strength):
            values=g[field].dropna();weight=strength if np.isfinite(parent) else 0
            return (values.sum()+weight*np.nan_to_num(parent))/(len(values)+weight) if len(values)+weight else np.nan
        t=smooth(typ,market,10);c=smooth(community,t,20);local=smooth(location,c,20)
        result['hist_location_type_365_'+suffix]=local
    result['location_history_no_sales']=float(len(location)==0)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    out=args.run.resolve();repo=Path(__file__).resolve().parents[2]
    manifest=json.loads((out/'source_manifest.json').read_text())
    for p,h in {**manifest['protected_sha256'],**manifest['source_sha256']}.items():assert sha(p)==h,p
    raw=pd.read_parquet(repo.parent/'data/listings.parquet')
    test=pd.read_parquet(repo.parent/'data/test_off_market.parquet')
    for c in ['date_listed','date_sold']:raw[c]=pd.to_datetime(raw[c])
    test['date_listed']=pd.to_datetime(test.date_listed)
    dev=pd.read_csv(out/'development_candidates.csv');check=pd.read_csv(out/'check_candidates.csv')
    selected=pd.read_csv(out/'check_selected_predictions.csv')
    checks=0
    for frame in [dev,check]:
        assert not frame.duplicated(['ml_num','candidate','cutoff_exclusive']).any()
        assert (frame.predicted_price>0).all() and np.isfinite(frame.predicted_price).all()
        labels=frame[['ml_num','sold_price']].merge(raw[['ml_num','sold_price']],on='ml_num',suffixes=('_pred','_raw'),validate='many_to_one')
        assert len(labels)==len(frame) and np.array_equal(labels.sold_price_pred,labels.sold_price_raw)
    selection=json.loads((out/'selection.json').read_text())
    choices,accepted,candidate_n=independent_selection(dev)
    assert choices==selection['choices'] and accepted==selection['accepted']
    for d in selection['candidate_scores']:
        z=dev.loc[dev.house_type_name.eq(d['house_type_name'])&dev.candidate.eq(d['candidate'])]
        compare(score(z),d);checks+=1
    expected=pd.concat([g[g.candidate.eq(choices.get(k,'previous'))] for k,g in check.groupby('house_type_name')],ignore_index=True)
    pd.testing.assert_frame_equal(expected[['ml_num','predicted_price']].sort_values('ml_num').reset_index(drop=True),
        selected[['ml_num','predicted_price']].sort_values('ml_num').reset_index(drop=True))
    bundles=json.loads((out/'metrics.json').read_text())
    latest=pd.read_csv(repo/'experiments/land_refinement/artifacts/run_2026-10-18_v2/check_selected_predictions.csv')
    for name,frame in [('previous',check[check.candidate.eq('previous')]),('selected',selected),('latest_land_refinement',latest)]:
        stored=bundles[name];compare(score(frame),stored['overall']);checks+=1
        for label,cols in [('by_type',['house_type_name']),('by_month',['month']),('by_type_month',['house_type_name','month']),('by_window',['cutoff_exclusive'])]:
            for s in stored[label]:
                mask=np.ones(len(frame),dtype=bool)
                for c in cols:mask &= frame[c].astype(str).eq(str(s[c])).to_numpy()
                compare(score(frame.loc[mask]),s);checks+=1
    for s in pd.read_csv(out/'candidate_scores.csv').to_dict('records'):
        z=check[check.house_type_name.eq(s['house_type_name'])&check.candidate.eq(s['candidate'])]
        compare(score(z),s);checks+=1
    # Direct pool scans, independent of HistoryFeatures prefix sums.
    area=raw.house_area.astype(str).str.extract(r'^\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*$').astype(float)
    mid=(area[0]+area[1].fillna(area[0]))/2
    raw['log_price']=np.log(raw.sold_price);raw['log_ppsf']=np.log(raw.sold_price/mid)
    raw['location_group']=raw.latitude.round(5).astype(str)+'|'+raw.longitude.round(5).astype(str)
    test['location_group']=test.latitude.round(5).astype(str)+'|'+test.longitude.round(5).astype(str)
    location_n=0
    for folder in sorted((out/'bases').iterdir()):
        a=json.loads((folder/'feature_audit.json').read_text());cutoff=a['cutoff']
        assert pd.Timestamp(a['max_train_sale'])<pd.Timestamp(cutoff)
        for split,id_file in [('train','train_ids.csv'),('valid','predict_ids.csv')]:
            ids=pd.read_csv(folder/id_file).ml_num
            source=test if a['final'] and split=='valid' else raw
            h=pd.DataFrame({'ml_num':ids}).merge(source,on='ml_num',validate='one_to_one')
            cached=pd.read_parquet(folder/(split+'_location.parquet'))
            positions=np.unique(np.linspace(0,len(h)-1,min(8,len(h)),dtype=int))
            for i in positions:
                result=direct_location(raw,h.iloc[i],cutoff)
                for c,value in result.items():assert np.isclose(cached.loc[i,c],value,equal_nan=True,atol=1e-9),(cutoff,split,i,c)
                location_n+=1
    calibration_n=0
    for folder in sorted((out/'calibrators').iterdir()):
        cutoff=pd.Timestamp(folder.name);oof=pd.read_csv(folder/'oof_predictions.csv')
        m=oof.merge(raw[['ml_num','date_sold','sold_price']],on='ml_num',suffixes=('_saved','_raw'),validate='one_to_one')
        assert (m.date_sold<cutoff).all() and (m.date_sold>=pd.to_datetime(m.cutoff_exclusive)).all()
        assert np.array_equal(m.sold_price_saved,m.sold_price_raw)
        for family,g in oof.groupby('family'):
            a=json.loads((folder/family/'audit.json').read_text())
            assert a['oof_n']==len(g) and not a['calibration_uses_in_sample_residuals']
            assert pd.Timestamp(a['max_calibration_sale'])<cutoff
            for c in g.cutoff_exclusive.unique():
                model='location' if family=='land' else 'apartment_base'
                src=pd.read_csv(out/'bases'/c/model/'predictions.csv').set_index('ml_num')
                z=g[g.cutoff_exclusive.eq(c)].set_index('ml_num')
                np.testing.assert_array_equal(z.predicted_price,src.reindex(z.index).predicted_price)
                audit=json.loads((out/'bases'/c/model/'audit.json').read_text())
                assert pd.Timestamp(audit['max_train_sale'])<pd.Timestamp(c)
            calibration_n+=1
    final=pd.read_csv(out/'final_predictions/submission_off_market.csv')
    replay=pd.read_csv(out/'final_predictions/replay.csv')
    pd.testing.assert_frame_equal(final,replay)
    assert list(final.columns)==['ml_num','predicted_price'] and final.ml_num.tolist()==test.ml_num.tolist()
    assert len(final)==3974 and np.isfinite(final.predicted_price).all() and final.predicted_price.gt(0).all()
    # Independently retrained original branches must reproduce the old final file.
    base=pd.read_csv(out/'final/2026-06-01/all_candidates.csv')
    base=base[base.candidate.eq('previous')].set_index('ml_num').reindex(test.ml_num).predicted_price
    archive=pd.read_csv(repo/'experiments/property_groups/artifacts/run_2026-10-17_v1/final_predictions/submission_off_market.csv').set_index('ml_num').reindex(test.ml_num).predicted_price
    np.testing.assert_array_equal(base,archive)
    result={'verifier_source_sha256':sha(Path(__file__)),'all_protected_and_training_source_hashes_match':True,
        'metric_bundles_recomputed':checks,'selection_candidate_comparisons':candidate_n,
        'selection_recomputed_without_check_labels':True,'selected_routes_match':True,
        'prediction_rows_checked':len(dev)+len(check),'direct_location_rows_checked':location_n,
        'temporal_calibrators_checked':calibration_n,'original_final_baseline_retrained_exactly':True,
        'final_replay_exact':True,'final_n':len(final),'hidden_test_error_available':False}
    (out/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
