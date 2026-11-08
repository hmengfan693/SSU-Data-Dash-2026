"""Independently verify raw labels, coverage, scores, training boundaries and replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.property_groups.run_experiment import sha, write_json
from experiments.property_groups.selection import DEV_WINDOWS, CHECK_WINDOW
from .models import SSGAM, eligible

REPO = Path(__file__).resolve().parents[2]


def verify(out):
    listings, test = data.load('listings'), data.load('off_market')
    raw = listings.set_index('ml_num')
    rows = pd.read_csv(out/'all_predictions.csv')
    scores = pd.read_csv(out/'scores.csv')
    if rows.duplicated(['ml_num','candidate']).any():
        raise AssertionError('Duplicate predictions')
    assert eligible(rows).all()
    for field in ('sold_price','house_type_name','community'):
        np.testing.assert_array_equal(rows[field], raw.loc[rows.ml_num,field])
    for record in scores.to_dict('records'):
        selected = rows.loc[(rows.protocol == record['protocol']) & (rows.candidate == record['candidate'])]
        if record['dimension'] != 'overall':
            selected = selected.loc[selected[record['dimension']].eq(record['slice'])]
        assert len(selected) == record['n']
        signed = 100 * (selected.predicted_price.to_numpy()/selected.sold_price.to_numpy()-1)
        computed = {'mdape': np.median(abs(signed)), 'mape': np.mean(abs(signed)),
                    'median_signed_error': np.median(signed), 'over_30_pct':100*np.mean(abs(signed)>30)}
        for field,value in computed.items():
            np.testing.assert_allclose(record[field], value, atol=1e-10, rtol=1e-10)
    replayed = 0
    for cutoff,end in (*DEV_WINDOWS,CHECK_WINDOW):
        protocol = 'check' if cutoff == CHECK_WINDOW[0] else 'development'
        homes = listings.loc[(listings.date_sold >= cutoff)&(listings.date_sold < end)&eligible(listings)].reset_index(drop=True)
        expected_ids = set(homes.ml_num)
        current = rows.loc[rows.cutoff_exclusive.eq(cutoff)]
        assert set(current.candidate) == {'core_ss_gam','core_gam','extended_ss_gam','extended_gam','existing_group_lightgbm'}
        for _, candidate in current.groupby('candidate'):
            assert set(candidate.ml_num) == expected_ids
        hidden = data.hide_answers(homes,'off_market')
        for config in ('core','extended'):
            folder = out/protocol/cutoff[:7]/config
            model = SSGAM.load(folder/'model.pkl')
            expected_train = set(listings.loc[eligible(listings)&(listings.date_sold < cutoff),'ml_num'])
            assert set(model.train_ids) == expected_train
            assert set(pd.read_csv(folder/'train_ids.csv').ml_num) == expected_train
            assert not expected_ids.intersection(model.train_ids)
            assert pd.Timestamp(model.max_training_sale) < pd.Timestamp(cutoff)
            assert model.optimization['success']
            for kind in ('ss_gam','gam'):
                reference = current.loc[current.candidate.eq(config+'_'+kind)].set_index('ml_num').loc[homes.ml_num]
                actual = model.predict(hidden,kind)
                np.testing.assert_allclose(actual, reference.predicted_price, rtol=1e-12, atol=1e-7)
                poison = homes.assign(sold_price=1e30,list_price=-1,date_sold=pd.Timestamp('2100-01-01'),
                                      days_on_market=-100,external_avm_estimate=-1e30)
                np.testing.assert_array_equal(actual, model.predict(poison,kind))
                replayed += len(actual)
    nonapt = test.loc[eligible(test)].reset_index(drop=True)
    final = pd.read_csv(out/'non_apartment_predictions.csv').set_index('ml_num').loc[nonapt.ml_num]
    assert set(final.index) == set(nonapt.ml_num) and not final.index.duplicated().any()
    for config in ('core','extended'):
        model = SSGAM.load(out/'final_models'/config/'model.pkl')
        assert set(model.train_ids) == set(listings.loc[eligible(listings),'ml_num'])
        assert pd.Timestamp(model.max_training_sale) < pd.Timestamp('2026-06-01')
        prediction = model.predict(data.hide_answers(nonapt,'off_market'))
        if config == 'extended':
            np.testing.assert_allclose(prediction,final.predicted_price,rtol=1e-12,atol=1e-7)
        replayed += len(prediction)
    submission_file = out/'experimental_submission_off_market.csv'
    assert not check_submission.check_file(submission_file,test.ml_num.tolist())
    submission = pd.read_csv(submission_file).set_index('ml_num').loc[test.ml_num]
    baseline = pd.read_csv(REPO/'experiments/fixed_four_groups/artifacts/run_2026-10-18_v1/final_predictions/submission_off_market.csv').set_index('ml_num')
    apartment_ids = test.loc[~eligible(test),'ml_num']
    np.testing.assert_array_equal(submission.loc[apartment_ids,'predicted_price'],baseline.loc[apartment_ids,'predicted_price'])
    np.testing.assert_allclose(submission.loc[nonapt.ml_num,'predicted_price'],final.predicted_price,rtol=1e-12)
    manifest = json.loads((out/'manifest.json').read_text())
    for key in ('sources','data','baseline_files'):
        for name,expected_hash in manifest[key].items():
            path = Path(name) if Path(name).is_absolute() else REPO/name
            assert sha(path) == expected_hash, path
    paired = pd.read_csv(out/'paired_comparison.csv')
    np.testing.assert_allclose(paired.delta_mdape_pp, paired.mdape-paired.baseline_mdape, atol=1e-10)
    result = {'status':'passed','score_slices_recomputed':len(scores),'prediction_rows_replayed':replayed,
              'raw_labels_exact':True,'coverage_exact':True,'train_boundaries_verified':True,
              'forbidden_field_poisoning_passed':True,'apartment_predictions_preserved':True,
              'source_data_and_baseline_hashes_verified':True,'submission_n':len(submission),
              'non_apartment_n':len(nonapt),'final_model_replay_passed':True}
    write_json(out/'verification.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(parser.parse_args().run.resolve()),indent=2))
