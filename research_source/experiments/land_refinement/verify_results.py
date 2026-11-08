"""Independent NumPy scoring, stability-gate reconstruction, and saved replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_experiment import replay, validate_archive


def score(frame):
    error=100*(frame.predicted_price.to_numpy()-frame.sold_price.to_numpy())/frame.sold_price.to_numpy()
    # Reproduce the original diagnostic's floating-point convention exactly
    # at a 30% boundary; MdAPE itself remains independently difference-based.
    ratio_error=100*(frame.predicted_price.to_numpy()/frame.sold_price.to_numpy()-1)
    return {"n":len(frame),"mdape":float(np.median(np.abs(error))),"mape":float(np.mean(np.abs(error))),
        "median_signed_error":float(np.median(error)),"over_30_pct":float(100*np.mean(np.abs(ratio_error)>30))}


def match(frame,expected):
    for k,v in score(frame).items():
        assert np.isclose(v,expected[k],rtol=0,atol=1e-10),(k,v,expected[k])


def verify(run,data_dir):
    run,data_dir=Path(run),Path(data_dir)
    manifest=json.loads((run/"manifest.json").read_text())
    assert manifest['status']=='complete'
    previous=Path(manifest['previous_run'])
    validate_archive(previous,data_dir)
    assert hashlib.sha256((previous/'manifest.json').read_bytes()).hexdigest()==manifest['previous_manifest_sha256']
    for path,digest in manifest['source_and_input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    selection=json.loads((run/'selection.json').read_text())
    choices=selection['choices']
    rows=pd.read_csv(run/'all_candidate_predictions.csv')
    assert not rows.duplicated(['ml_num','candidate']).any()
    assert np.isfinite(rows.predicted_price).all() and (rows.predicted_price>0).all()
    listings=pd.read_parquet(data_dir/'listings.parquet')
    listings['date_sold']=pd.to_datetime(listings.date_sold)
    raw=listings.set_index('ml_num')
    np.testing.assert_array_equal(rows.sold_price,raw.loc[rows.ml_num].sold_price)
    np.testing.assert_array_equal(rows.house_type_name,raw.loc[rows.ml_num].house_type_name)
    np.testing.assert_array_equal(rows.month,raw.loc[rows.ml_num].date_sold.dt.strftime('%Y-%m'))
    dev=rows.loc[rows.protocol=='development']
    for kind,frame in dev.groupby('house_type_name'):
        candidates={name:f for name,f in frame.groupby('candidate')}
        values={name:score(f) for name,f in candidates.items()}
        anchor=min(('previous','global'),key=lambda n:(values[n]['mdape'],n))
        assert anchor==selection['details'][kind]['anchor']
        eligible=[anchor]
        for name,f in candidates.items():
            assert set(f.ml_num)==set(frame.ml_num)
            match(f,selection['details'][kind]['candidates'][name]['overall'])
            wins=sum(score(f.loc[f.cutoff_exclusive==c])['mdape']<score(candidates[anchor].loc[
                candidates[anchor].cutoff_exclusive==c])['mdape'] for c in sorted(frame.cutoff_exclusive.unique()))
            if values[name]['mdape']<values[anchor]['mdape'] and wins>=2:
                eligible.append(name)
        best=min(eligible,key=lambda n:(values[n]['mdape'],n))
        assert best==choices[kind]
    metric_bundles=0
    summaries=json.loads((run/'metrics.json').read_text())
    for protocol in ('development','check'):
        old=pd.read_csv(run/f'{protocol}_previous_predictions.csv')
        selected=pd.read_csv(run/f'{protocol}_selected_predictions.csv')
        expected_old=pd.read_csv(previous/f'{protocol}_routed_predictions.csv').set_index('ml_num')
        assert set(old.ml_num)==set(expected_old.index)
        np.testing.assert_array_equal(old.predicted_price,old.ml_num.map(expected_old.predicted_price))
        candidates=rows.loc[rows.protocol==protocol]
        chosen=candidates.loc[candidates.candidate.eq(candidates.house_type_name.map(choices))].set_index('ml_num')
        assert not chosen.index.duplicated().any()
        mask=selected.ml_num.isin(chosen.index)
        np.testing.assert_array_equal(selected.loc[mask,'predicted_price'],selected.loc[mask,'ml_num'].map(chosen.predicted_price))
        np.testing.assert_array_equal(selected.loc[~mask,'predicted_price'],old.loc[~mask,'predicted_price'])
        assert set(selected.ml_num)==set(old.ml_num) and not selected.ml_num.duplicated().any()
        for label,data in (('previous',old),('selected',selected)):
            bundle=summaries[protocol][label]
            match(data,bundle['overall']);metric_bundles+=1
            for dimension,keys in (('by_group',['group']),('by_type',['house_type_name']),('by_month',['month']),
                ('by_group_month',['group','month']),('by_window',['cutoff_exclusive'])):
                for item in bundle[dimension]:
                    m=np.ones(len(data),dtype=bool)
                    for k in keys:m&=data[k].eq(item[k]).to_numpy()
                    match(data.loc[m],item);metric_bundles+=1
    comparison=pd.read_csv(run/'candidate_scores.csv')
    for item in comparison.to_dict('records'):
        match(rows.loc[(rows.protocol==item['protocol'])&(rows.house_type_name==item['house_type_name'])&
              (rows.candidate==item['candidate'])],item)
    # Every arithmetic blend is independently aligned and recomputed.
    for (protocol,cutoff,kind),f in rows.groupby(['protocol','cutoff_exclusive','house_type_name']):
        global_price=f.loc[f.candidate=='global'].set_index('ml_num').predicted_price
        for name,b in f.loc[f.candidate.str.startswith('half_')].groupby('candidate'):
            source=f.loc[f.candidate==name.removeprefix('half_')].set_index('ml_num').predicted_price
            expected=np.round(.5*b.ml_num.map(global_price).to_numpy()+.5*b.ml_num.map(source).to_numpy())
            np.testing.assert_array_equal(expected,b.predicted_price)
    for protocol,cutoff,end in [("development",*w) for w in manifest['development_windows']]+[("check",*manifest['check_window'])]:
        audit=json.loads((run/protocol/cutoff[:7]/'audit.json').read_text())
        train=listings.loc[listings.date_sold<cutoff]
        valid=listings.loc[(listings.date_sold>=cutoff)&(listings.date_sold<end)]
        assert audit['train_n']==len(train) and audit['evaluation_n']==len(valid)
        assert audit['evaluation_labels_used_for_fit_or_early_stopping'] is False
        for identifier,detail in audit['models'].items():
            kind,variant=identifier.split('/')
            assert detail['train_n']==int(train.house_type_name.eq(kind).sum())
            frame=rows.loc[(rows.protocol==protocol)&(rows.cutoff_exclusive==cutoff)&
                (rows.house_type_name==kind)&(rows.candidate==variant)]
            assert set(frame.ml_num)==set(valid.loc[valid.house_type_name==kind,'ml_num'])
            assert detail['reload_exact'] is True
            match(frame,detail['scores'])
    actual,_=replay(run,data_dir,run/'verification_replay.csv')
    expected=pd.read_csv(run/'final_predictions/submission_off_market.csv')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
    homes=pd.read_parquet(data_dir/'test_off_market.parquet')
    old_final=pd.read_csv(previous/'final_predictions/submission_off_market.csv').set_index('ml_num')
    fixed=~homes.house_type_name.isin(choices)
    np.testing.assert_array_equal(actual.loc[fixed,'predicted_price'],homes.loc[fixed,'ml_num'].map(old_final.predicted_price))
    checks=json.loads((run/'checks.json').read_text())
    assert hashlib.sha256((run/'selection.json').read_bytes()).hexdigest()==checks['selection_sha256']
    report={'all_checks_passed':True,'candidate_rows_recomputed':len(rows),'metric_bundles_recomputed':metric_bundles,
        'candidate_score_groups_recomputed':len(comparison),'frozen_choices_independently_recomputed':True,
        'all_blends_recomputed_by_id':True,'original_archive_intact':True,'apartment_predictions_unchanged':True,
        'final_predictions_reproduced':len(actual)}
    (run/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[3]/'data')
    args=parser.parse_args()
    print(json.dumps(verify(args.run,args.data_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
