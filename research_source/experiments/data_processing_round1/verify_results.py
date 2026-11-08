"""Independently verify IDs/truth, all scores, boundaries, frozen rules and models."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import json

from experiments.property_groups.models import GroupModel
from .features import assemble
from .run_experiment import REPO, TARGETS, LAND, CHECK, FINAL_CUTOFF, sha, write_json, now, infer_saved, load_data


def calculate(frame):
    e = 100*(frame.predicted_price.to_numpy()/frame.sold_price.to_numpy()-1)
    return {'n': len(e), 'mdape': np.median(np.abs(e)), 'mape': np.mean(np.abs(e)),
            'median_signed_error': np.median(e), 'over_30_pct': 100*np.mean(np.abs(e)>30)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    args = parser.parse_args()
    out = args.run.resolve()
    protocol = json.loads((out/'protocol.json').read_text())
    freeze = json.loads((out/'prediction_freeze.json').read_text())
    assert sha(out/'protocol.json') == freeze['protocol_sha256']
    assert sha(out/'all_predictions_unscored.csv') == freeze['unscored_predictions_sha256']
    assert protocol['frozen_at_Toronto'] < freeze['frozen_at_Toronto']
    assert protocol['designated_candidate'] == 'all_three' and protocol['selection_by_scores'] is False
    for name, digest in protocol['source_hashes'].items():
        assert sha(REPO/name) == digest
        assert sha(out/'source_snapshot'/name.replace('/', '__')) == digest
    for name, digest in protocol['protected_sha256'].items():
        assert sha(name) == digest
    raw, test = load_data(args.data_dir)
    p = pd.read_csv(out/'all_predictions_scored.csv')
    assert not p.duplicated(['cutoff_exclusive', 'ml_num', 'variant']).any()
    truth = p[['ml_num', 'sold_price', 'date_sold']].merge(raw[['ml_num', 'sold_price', 'date_sold']],
        on='ml_num', validate='many_to_one', suffixes=('_saved', '_raw'))
    assert len(truth) == len(p)
    np.testing.assert_array_equal(truth.sold_price_saved, truth.sold_price_raw)
    np.testing.assert_array_equal(pd.to_datetime(truth.date_sold_saved), pd.to_datetime(truth.date_sold_raw))
    assert (pd.to_datetime(p.date_sold) >= pd.to_datetime(p.cutoff_exclusive)).all()
    scored_count = 0
    scores = pd.read_csv(out/'scores.csv')
    for _, row in scores.iterrows():
        mask = p.protocol.eq(row.protocol) & p.house_type_name.eq(row.house_type_name) & p.variant.eq(row.variant)
        if row.level in ('window', 'month'):
            mask &= p.cutoff_exclusive.eq(row.cutoff_exclusive)
        if row.level == 'month':
            mask &= p.sold_month.eq(row.sold_month)
        for metric, value in calculate(p.loc[mask]).items():
            assert np.isclose(value, row[metric], rtol=0, atol=1e-10), (row.to_dict(), metric, value)
        scored_count += 1
    # Re-run every saved model on cached feature blocks, with independently verified row IDs.
    replayed_models = 0
    for folder in sorted((out/'windows').iterdir()):
        cutoff = folder.name
        f = folder/'features'
        train_ids = pd.read_csv(f/'train_ids.csv')
        valid_ids = pd.read_csv(f/'predict_ids.csv')
        expected_train = raw.loc[(raw.date_sold < pd.Timestamp(cutoff)) & raw.house_type_name.isin(LAND)].sort_values(['date_sold', 'ml_num'])
        np.testing.assert_array_equal(train_ids.ml_num, expected_train.ml_num)
        if cutoff == FINAL_CUTOFF:
            expected_ids = test.loc[test.house_type_name.isin(TARGETS), 'ml_num']
        else:
            end = CHECK[1] if cutoff == CHECK[0] else str((pd.Timestamp(cutoff)+pd.DateOffset(months=4)).date())
            expected_ids = raw.loc[(raw.date_sold >= pd.Timestamp(cutoff)) & (raw.date_sold < pd.Timestamp(end)) & raw.house_type_name.isin(TARGETS), 'ml_num']
        np.testing.assert_array_equal(valid_ids.ml_num, expected_ids)
        blocks = [pd.read_parquet(f/f'predict_{name}.parquet') for name in ('base', 'history', 'land', 'text')]
        variants = ['all_three'] if cutoff == FINAL_CUTOFF else protocol['variants']
        for variant in variants:
            x = assemble(*blocks, variant)
            for target in TARGETS:
                model_folder = folder/target/variant
                model = GroupModel.load_bundle(model_folder)
                schema = json.loads((model_folder/'schema.json').read_text())
                assert schema['params'] == protocol['params']
                mask = valid_ids.house_type_name.eq(target).to_numpy()
                saved = pd.read_csv(model_folder/'predictions_unscored.csv')
                np.testing.assert_array_equal(saved.ml_num, valid_ids.loc[mask, 'ml_num'])
                np.testing.assert_array_equal(saved.predicted_price, model.predict(x.loc[mask]))
                replayed_models += 1
    # Fresh hidden-test reconstruction, including features, in this separate process.
    final = pd.read_csv(out/'final_predictions/submission_off_market.csv')
    fresh = infer_saved(raw, test, out, Path(protocol['archive']))
    np.testing.assert_array_equal(final.ml_num, fresh.ml_num)
    np.testing.assert_array_equal(final.predicted_price, fresh.predicted_price)
    archive = pd.read_csv(Path(protocol['archive'])/'final_predictions/submission_off_market.csv').set_index('ml_num')
    target_mask = test.house_type_name.isin(TARGETS).to_numpy()
    np.testing.assert_array_equal(final.loc[~target_mask, 'predicted_price'], archive.predicted_price.reindex(test.loc[~target_mask, 'ml_num']))
    result = {'verified_at_Toronto': now(), 'protocol_and_sources_frozen': True,
        'truth_rows_verified': len(p), 'score_rows_recomputed': scored_count,
        'models_replayed': replayed_models, 'fresh_final_replay_exact': True,
        'non_target_final_unchanged': True, 'final_n': len(final),
        'protected_files_unchanged': len(protocol['protected_sha256']), 'score_based_selection': False}
    write_json(out/'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
