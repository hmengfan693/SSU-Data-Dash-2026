"""Independent metric arithmetic, complete saved-model replay and raw-feature replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
from experiments.data_processing_round1.run_experiment import load_data
from .models import DropoutModel, DROP_SEEDS, parameters
from .run_experiment import (REPO, WINDOWS, FINAL_CUTOFF, PRIMARY_SEED, sha, write_json,
                             load_designs, archived_baseline, fresh_final_predict, now)


def verify(run, data_dir):
    run = Path(run)
    protocol = json.loads((run/'protocol.json').read_text())
    freeze = json.loads((run/'prediction_freeze.json').read_text())
    assert sha(run/'protocol.json') == freeze['protocol_sha256']
    assert sha(run/'all_predictions_unscored.csv') == freeze['unscored_sha256']
    for mapping in ('source_sha256', 'input_sha256'):
        for path, digest in protocol[mapping].items():
            assert sha(path) == digest, path
    raw, test = load_data(data_dir)
    stored = pd.read_csv(run/'all_predictions_scored.csv')
    assert not stored.duplicated(['cutoff_exclusive', 'ml_num', 'candidate']).any()
    truth = raw.set_index('ml_num')
    np.testing.assert_array_equal(stored.sold_price, truth.sold_price.reindex(stored.ml_num))
    np.testing.assert_array_equal(pd.to_datetime(stored.date_sold), truth.date_sold.reindex(stored.ml_num))
    assert (pd.to_datetime(stored.date_sold) >= pd.to_datetime(stored.cutoff_exclusive)).all()
    scores = pd.read_csv(run/'scores.csv')
    columns = {'overall': ['protocol', 'candidate'], 'type': ['protocol', 'candidate', 'house_type_name'],
               'window': ['protocol', 'candidate', 'cutoff_exclusive'],
               'type_window': ['protocol', 'candidate', 'cutoff_exclusive', 'house_type_name'],
               'month': ['protocol', 'candidate', 'sold_month']}
    for _, row in scores.iterrows():
        mask = np.ones(len(stored), dtype=bool)
        for column in columns[row.level]:
            mask &= stored[column].eq(row[column]).to_numpy()
        part = stored.loc[mask]
        error = 100*(part.predicted_price.to_numpy()/part.sold_price.to_numpy()-1)
        assert len(part) == row.n
        for name, value in [('mdape', np.median(np.abs(error))), ('mape', np.mean(np.abs(error))),
                             ('median_signed_error', np.median(error)), ('over_30_pct', 100*np.mean(np.abs(error) > 30))]:
            assert np.isclose(row[name], value, rtol=0, atol=1e-10), (row.level, name)
    count = 0
    for cutoff, end in [*WINDOWS, (FINAL_CUTOFF, None)]:
        homes, designs = load_designs(raw, test, cutoff, end)
        control = archived_baseline(homes, cutoff, final=end is None).set_index('ml_num')
        rows = pd.read_csv(run/'windows'/cutoff/'predictions_unscored.csv')
        assert rows.groupby('candidate').size().eq(len(homes)).all()
        for kind, (train, valid, xt, xv) in designs.items():
            assert (train.date_sold < pd.Timestamp(cutoff)).all()
            assert not set(train.ml_num) & set(valid.ml_num)
            seeds = (PRIMARY_SEED,) if end is None else (None, *DROP_SEEDS)
            for seed in seeds:
                name = 'gbdt' if seed is None else f'dart_seed_{seed}'
                folder = run/'windows'/cutoff/kind/name
                schema = json.loads((folder/'schema.json').read_text())
                assert schema['params'] == parameters(seed)
                assert schema['cutoff_exclusive'] == cutoff and schema['training_n'] == len(train)
                model = DropoutModel.load(folder)
                actual = model.predict(xv)
                expected = rows.loc[rows.candidate.eq(name)].set_index('ml_num').predicted_price.reindex(valid.ml_num)
                np.testing.assert_array_equal(actual, expected)
                assert model.booster.current_iteration() == 600
                text = (folder/'model.txt').read_text()
                assert f'[boosting: {"gbdt" if seed is None else "dart"}]' in text
                if seed is None:
                    np.testing.assert_array_equal(actual, control.predicted_price.reindex(valid.ml_num))
                else:
                    assert '[drop_rate: 0.01]' in text and '[uniform_drop: 1]' in text
                    assert f'[drop_seed: {seed}]' in text
                count += 1
        for _, candidate in rows.groupby('candidate'):
            apartment = candidate.house_type_name.isin(['Condo Apt', 'Co-Op Apt'])
            np.testing.assert_array_equal(candidate.loc[apartment, 'predicted_price'],
                control.predicted_price.reindex(candidate.loc[apartment, 'ml_num']))
    final_path = run/'final_predictions/submission_off_market_dart.csv'
    final = pd.read_csv(final_path)
    assert not check_submission.check_file(final_path, test.ml_num)
    replay = fresh_final_predict(raw, test, run)
    pd.testing.assert_frame_equal(final, replay)
    assert count == 68
    result = dict(verified_at=now(), metric_rows=len(scores), backtest_prediction_rows=len(stored),
                  saved_models_replayed=count, all_training_cutoffs_checked=True,
                  native_dart_params_checked=True, raw_final_feature_replay_exact=True,
                  final_prediction_n=len(final), catboost_branches_unchanged=True,
                  input_source_protocol_hashes_unchanged=True)
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
