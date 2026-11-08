"""Independent raw-label metrics, selection audit, and both versions' replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
from experiments.xgboost.models import XGBoostPriceModel, PARAMS
from experiments.xgboost.verify_results import independent_metrics
from experiments.xgboost.run_experiment import fresh_final_predict as native_final_predict
from .models import OneHotPriceModel, PREFIX
from .selection import choose, NATIVE, ONEHOT
from .run_experiment import (REPO, WINDOWS, FINAL_CUTOFF, NATIVE_RUN, OLD_CANDIDATE,
    BASELINE, load_data, load_designs, archived_baseline, fresh_final_predict, sha, write_json, now)


def verify(run, data_dir):
    run = Path(run).resolve()
    protocol = json.loads((run/'protocol.json').read_text())
    freeze = json.loads((run/'prediction_freeze.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    check_start = json.loads((run/'check_start.json').read_text())
    assert sha(run/'protocol.json') == freeze['protocol_sha256']
    assert sha(run/'all_predictions_unscored.csv') == freeze['unscored_sha256']
    assert sha(run/'selection.json') == freeze['selection_sha256'] == check_start['selection_sha256']
    assert pd.Timestamp(selection['frozen_at']) <= pd.Timestamp(check_start['started_at'])
    assert sha(run/'development_predictions_unscored.csv') == selection['development_predictions_sha256']
    for mapping in ('source_sha256', 'input_sha256'):
        for path, digest in protocol[mapping].items():
            assert sha(path) == digest, path
    raw, test = load_data(data_dir)
    stored = pd.read_csv(run/'all_predictions_scored.csv')
    unscored = pd.read_csv(run/'all_predictions_unscored.csv')
    pd.testing.assert_frame_equal(stored[list(unscored)], unscored)
    assert not stored.duplicated(['candidate', 'cutoff_exclusive', 'ml_num']).any()
    truth = raw.set_index('ml_num')
    np.testing.assert_array_equal(stored.sold_price, truth.sold_price.reindex(stored.ml_num))
    np.testing.assert_array_equal(pd.to_datetime(stored.date_sold), truth.date_sold.reindex(stored.ml_num))
    assert (pd.to_datetime(stored.date_sold) >= pd.to_datetime(stored.cutoff_exclusive)).all()
    assert (pd.to_datetime(stored.date_sold) < pd.to_datetime(stored.cutoff_exclusive.map(dict(WINDOWS)))).all()
    dimensions = dict(overall=['protocol'], type=['protocol', 'house_type_name'],
        window=['protocol', 'cutoff_exclusive'], month=['protocol', 'sold_month'],
        type_window=['protocol', 'cutoff_exclusive', 'house_type_name'])
    scores = pd.read_csv(run/'scores.csv')
    for _, row in scores.iterrows():
        mask = stored.candidate.eq(row.candidate)
        for key in dimensions[row.level]:
            mask &= stored[key].eq(row[key])
        part = stored.loc[mask]
        assert len(part) == row.n
        for key, value in independent_metrics(part).items():
            assert np.isclose(row[key], value, rtol=0, atol=1e-10)
    comparison = pd.read_csv(run/'comparison_to_current_best.csv')
    for _, row in comparison.iterrows():
        mask = pd.Series(True, index=stored.index)
        for key in dimensions[row.level]:
            mask &= stored[key].eq(row[key])
        part = stored.loc[mask]
        a = part.loc[part.candidate.eq(BASELINE)].set_index('ml_num').sort_index()
        b = part.loc[part.candidate.eq(row.candidate)].set_index('ml_num').sort_index()
        assert a.index.equals(b.index) and len(a) == row.n
        for key, left in independent_metrics(a).items():
            right = independent_metrics(b)[key]
            for label, value in [(f'current_best_{key}', left),
                (f'candidate_{key}', right), (f'delta_{key}_pp', right-left)]:
                assert np.isclose(row[label], value, rtol=0, atol=1e-10)
    dev = stored.loc[stored.protocol.eq('development') & stored.candidate.isin([NATIVE, ONEHOT])]
    expected_selection = choose(dev)
    for key, value in expected_selection.items():
        assert selection[key] == value, key
    dev_scores = {name: independent_metrics(dev.loc[dev.candidate.eq(name)]) for name in (NATIVE, ONEHOT)}
    independent_choice = min((NATIVE, ONEHOT), key=lambda name:
        (dev_scores[name]['mdape'], dev_scores[name]['mape'], name != NATIVE))
    assert independent_choice == selection['selected_candidate']
    onehot_count = native_count = 0
    for cutoff, end in [*WINDOWS, (FINAL_CUTOFF, None)]:
        homes, designs = load_designs(raw, test, cutoff, end)
        rows = pd.read_csv(run/'windows'/cutoff/'predictions_unscored.csv')
        if end is not None:
            left = stored.loc[stored.cutoff_exclusive.eq(cutoff), list(rows)].reset_index(drop=True)
            pd.testing.assert_frame_equal(left, rows)
        baseline = archived_baseline(homes, cutoff, final=end is None).set_index('ml_num')
        for name, group in rows.groupby('candidate'):
            assert not group.ml_num.duplicated().any() and set(group.ml_num) == set(homes.ml_num)
            subset = group if name == BASELINE else group.loc[group.house_type_name.isin(['Condo Apt', 'Co-Op Apt'])]
            np.testing.assert_array_equal(subset.predicted_price, baseline.predicted_price.reindex(subset.ml_num))
        for kind, (train, valid, xt, xv) in designs.items():
            assert (train.date_sold < pd.Timestamp(cutoff)).all()
            assert not set(train.ml_num) & set(valid.ml_num)
            folder = run/'windows'/cutoff/kind/ONEHOT
            model = OneHotPriceModel.load(folder)
            schema = json.loads((folder/'schema.json').read_text())
            assert schema['params'] == PARAMS
            assert schema['training_n'] == len(train) and schema['cutoff_exclusive'] == cutoff
            assert pd.Timestamp(schema['max_train_sale']) == train.date_sold.max()
            assert schema['init_log_price'] == float(np.median(np.log(train.sold_price)))
            assert model.categories['house_type_name'] == sorted(train.house_type_name.unique())
            encoded = model.encode(xv)
            native = XGBoostPriceModel.load(NATIVE_RUN/'windows'/cutoff/kind/OLD_CANDIDATE)
            reference_encoded = native.encode(xv)
            pd.testing.assert_frame_equal(encoded.drop(columns=[c for c in encoded if c.startswith(PREFIX)]),
                                          reference_encoded.drop(columns='house_type_name'))
            columns = [c for c in encoded if c.startswith(PREFIX)]
            expected_dummies = np.column_stack([xv.house_type_name.eq(v).astype('float32')
                for v in model.categories['house_type_name']])
            np.testing.assert_array_equal(encoded[columns].to_numpy(), expected_dummies)
            assert set(encoded[columns].to_numpy().ravel()).issubset({0., 1.})
            assert 'house_type_name' not in model.model.get_booster().feature_names
            for name, fitted in ((NATIVE, native), (ONEHOT, model)):
                expected = rows.loc[rows.candidate.eq(name)].set_index('ml_num').predicted_price.reindex(valid.ml_num)
                np.testing.assert_array_equal(fitted.predict(xv), expected)
                assert fitted.model.get_booster().num_boosted_rounds() == 600
            onehot_count += 1
            native_count += 1
    assert onehot_count == native_count == 20
    target = run/'final_predictions'
    for name, reference in [('onehot', fresh_final_predict(raw, test, run)),
                            ('native', native_final_predict(raw, test, NATIVE_RUN))]:
        path = target/f'submission_off_market_{name}.csv'
        assert not check_submission.check_file(path, test.ml_num)
        pd.testing.assert_frame_equal(pd.read_csv(path), reference)
    selected_path = target/'submission_off_market_selected.csv'
    suffix = 'onehot' if independent_choice == ONEHOT else 'native'
    pd.testing.assert_frame_equal(pd.read_csv(selected_path), pd.read_csv(target/f'submission_off_market_{suffix}.csv'))
    assert not check_submission.check_file(selected_path, test.ml_num)
    best = json.loads((run/'current_best_before.json').read_text())['off_market_fixed_four_months']
    overall = comparison.loc[comparison.level.eq('overall') & comparison.protocol.eq('check')]
    assert np.allclose(overall.current_best_mdape, best['metrics_percent']['mdape'], rtol=0, atol=1e-10)
    result = dict(verified_at=now(), onehot_models_replayed=onehot_count, native_models_replayed=native_count,
        metric_rows=len(scores), paired_comparison_rows=len(comparison), native_features_unchanged_except_house_type=True,
        onehot_dense_zero_one_checked=True, selection_development_only=True,
        selected_candidate=independent_choice, raw_final_both_versions_replay_exact=True,
        raw_labels_time_boundaries_ids_checked=True, protected_hashes_unchanged=True, final_n=len(test))
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
