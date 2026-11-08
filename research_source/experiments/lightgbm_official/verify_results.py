"""Recompute metrics from raw labels and replay every saved model independently."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
from experiments.data_processing_round1.run_experiment import load_data
from .models import PARAMS, ROUNDS, OfficialLightGBM
from .run_experiment import (REPO, WINDOWS, FINAL_CUTOFF, CANDIDATE, BASELINE, PROCESSING, RESIDUAL,
    load_designs, archived_baseline, fresh_final_predict, sha, write_json, now)


def independent_metrics(rows):
    error = 100*(rows.predicted_price.to_numpy()/rows.sold_price.to_numpy()-1)
    return dict(mdape=np.median(np.abs(error)), mape=np.mean(np.abs(error)),
                median_signed_error=np.median(error), over_30_pct=100*np.mean(np.abs(error)>30))


def verify(run, data_dir):
    run = Path(run).resolve()
    protocol = json.loads((run/'protocol.json').read_text())
    freeze = json.loads((run/'prediction_freeze.json').read_text())
    assert sha(run/'protocol.json') == freeze['protocol_sha256']
    assert sha(run/'all_predictions_unscored.csv') == freeze['unscored_sha256']
    for mapping in ('source_sha256', 'input_sha256', 'runtime_sha256'):
        for path, digest in protocol[mapping].items():
            assert sha(path) == digest, path
    for item in protocol['upstream']['files'].values():
        assert sha(run/item['local_file']) == item['sha256']
    raw, test = load_data(data_dir)
    stored = pd.read_csv(run/'all_predictions_scored.csv')
    unscored = pd.read_csv(run/'all_predictions_unscored.csv')
    pd.testing.assert_frame_equal(stored[list(unscored)], unscored)
    assert not stored.duplicated(['cutoff_exclusive', 'ml_num', 'candidate']).any()
    truth = raw.set_index('ml_num')
    np.testing.assert_array_equal(stored.sold_price, truth.sold_price.reindex(stored.ml_num))
    np.testing.assert_array_equal(pd.to_datetime(stored.date_sold), truth.date_sold.reindex(stored.ml_num))
    assert (pd.to_datetime(stored.date_sold) >= pd.to_datetime(stored.cutoff_exclusive)).all()
    end_map = dict(WINDOWS)
    assert (pd.to_datetime(stored.date_sold) < pd.to_datetime(stored.cutoff_exclusive.map(end_map))).all()
    dimensions = dict(overall=['protocol'], type=['protocol', 'house_type_name'],
                      window=['protocol', 'cutoff_exclusive'],
                      type_window=['protocol', 'cutoff_exclusive', 'house_type_name'],
                      month=['protocol', 'sold_month'])
    scores = pd.read_csv(run/'scores.csv')
    for _, row in scores.iterrows():
        mask = stored.candidate.eq(row.candidate)
        for column in dimensions[row.level]:
            mask &= stored[column].eq(row[column])
        part = stored.loc[mask]
        assert len(part) == row.n
        for key, value in independent_metrics(part).items():
            assert np.isclose(row[key], value, atol=1e-10, rtol=0)
    comparison = pd.read_csv(run/'comparison_to_current_best.csv')
    for _, row in comparison.iterrows():
        mask = pd.Series(True, index=stored.index)
        for column in dimensions[row.level]:
            mask &= stored[column].eq(row[column])
        part = stored.loc[mask]
        a = part.loc[part.candidate.eq(BASELINE)].set_index('ml_num').sort_index()
        b = part.loc[part.candidate.eq(CANDIDATE)].set_index('ml_num').sort_index()
        assert a.index.equals(b.index) and len(a) == row.n
        for key, value in independent_metrics(a).items():
            other = independent_metrics(b)[key]
            assert np.isclose(row['current_best_'+key], value, atol=1e-10, rtol=0)
            assert np.isclose(row['official_lightgbm_'+key], other, atol=1e-10, rtol=0)
            assert np.isclose(row['delta_'+key+'_pp'], other-value, atol=1e-10, rtol=0)
    count = controls = 0
    for cutoff, end in [*WINDOWS, (FINAL_CUTOFF, None)]:
        homes, designs = load_designs(raw, test, cutoff, end)
        rows = pd.read_csv(run/'windows'/cutoff/'predictions_unscored.csv')
        baseline = archived_baseline(homes, cutoff, final=end is None).set_index('ml_num')
        for _, candidate in rows.groupby('candidate'):
            assert not candidate.ml_num.duplicated().any() and set(candidate.ml_num) == set(homes.ml_num)
            apartment = candidate.house_type_name.isin(['Condo Apt', 'Co-Op Apt'])
            np.testing.assert_array_equal(candidate.loc[apartment, 'predicted_price'],
                baseline.predicted_price.reindex(candidate.loc[apartment, 'ml_num']))
        for kind, (train, valid, xt, xv) in designs.items():
            assert (train.date_sold < pd.Timestamp(cutoff)).all()
            assert not set(train.ml_num) & set(valid.ml_num)
            folder = run/'windows'/cutoff/kind/CANDIDATE
            schema = json.loads((folder/'schema.json').read_text())
            assert schema['params'] == PARAMS
            assert schema['cutoff_exclusive'] == cutoff and schema['training_n'] == len(train)
            assert pd.Timestamp(schema['max_train_sale']) == train.date_sold.max()
            assert schema['num_boost_round'] == ROUNDS
            model = OfficialLightGBM.load(folder)
            assert model.booster.current_iteration() == ROUNDS
            assert model.booster.dump_model()['objective'] == 'regression_l1'
            expected = rows.loc[rows.candidate.eq(CANDIDATE)].set_index('ml_num').predicted_price.reindex(valid.ml_num)
            np.testing.assert_array_equal(model.predict(xv), expected)
            branch = pd.read_csv(folder/'predictions_unscored.csv')
            np.testing.assert_array_equal(branch.ml_num, valid.ml_num)
            np.testing.assert_array_equal(branch.predicted_price, expected)
            if end is not None:
                subset = stored.loc[stored.cutoff_exclusive.eq(cutoff) & stored.candidate.eq(CANDIDATE)]
                np.testing.assert_array_equal(subset.set_index('ml_num').predicted_price.reindex(valid.ml_num), expected)
                np.testing.assert_array_equal(model.predict(xv), baseline.predicted_price.reindex(valid.ml_num))
                controls += 1
            np.testing.assert_array_equal(expected, baseline.predicted_price.reindex(valid.ml_num))
            count += 1
    assert count == 20 and controls == 16
    path = run/'final_predictions/submission_off_market_lightgbm_official.csv'
    final = pd.read_csv(path)
    assert not check_submission.check_file(path, test.ml_num)
    pd.testing.assert_frame_equal(final, fresh_final_predict(raw, test, run))
    before = json.loads((run/'current_best_before.json').read_text())
    overall = comparison.loc[comparison.level.eq('overall') & comparison.protocol.eq('check')].iloc[0]
    assert np.isclose(overall.current_best_mdape,
        before['off_market_fixed_four_months']['metrics_percent']['mdape'], rtol=0, atol=1e-10)
    assert int(overall.n) == 7773
    result = dict(verified_at=now(), metric_rows=len(scores), comparison_rows=len(comparison),
        official_lightgbm_models_replayed=count, backtest_archived_branches_reproduced=controls,
        raw_labels_time_boundaries_ids_checked=True, raw_final_feature_replay_exact=True,
        current_best_baseline_reproduced=True, final_n=len(final), protected_hashes_unchanged=True)
    np.testing.assert_array_equal(final.predicted_price, archived_baseline(test, FINAL_CUTOFF, final=True).predicted_price)
    assert (comparison.delta_mdape_pp == 0).all()
    assert json.loads((REPO/'CURRENT_BEST_MODEL.json').read_text()) == before
    result.update(all_complete_predictions_identical=True, best_index_unchanged=True, official_submissions_replaced=False)
    write_json(run/'independent_verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    args = parser.parse_args()
    verify(args.run, args.data_dir)


if __name__ == '__main__':
    main()
