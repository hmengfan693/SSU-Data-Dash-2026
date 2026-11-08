"""Independently audit completed monthly folds and recompute pooled errors."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import data

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    verification = json.loads((out / 'verification.json').read_text())
    for name, digest in verification['files_sha256'].items():
        if sha(out / name) != digest:
            raise ValueError('Changed result artifact: ' + name)
    for name, digest in {**protocol['protected_sha256'], **protocol['source_sha256']}.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Changed protected input or source: ' + name)
    if sha(data.find_file('listings')) != protocol['input_sha256']:
        raise ValueError('Raw data hash mismatch')
    raw = data.load('listings')
    all_folds = []
    for month in pd.period_range('2025-12', '2026-05', freq='M'):
        cutoff, end = month.start_time, (month + 1).start_time
        folder = out / 'folds' / str(month)
        ids = pd.read_csv(folder / 'train_ids.csv')
        train = raw.loc[raw.date_sold < cutoff].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
        homes = raw.loc[(raw.date_sold >= cutoff) & (raw.date_sold < end)]
        np.testing.assert_array_equal(ids.ml_num, train.ml_num)
        np.testing.assert_array_equal(pd.to_datetime(ids.date_sold), train.date_sold)
        audit = json.loads((folder / 'audit.json').read_text())
        assert audit['train_n'] == len(train) and audit['predict_n'] == len(homes)
        assert pd.Timestamp(audit['max_train_sale']) < cutoff
        for path in (folder / 'off_market').glob('*/schema.json'):
            s = json.loads(path.read_text())
            template = json.loads((ROOT / 'saved_models' / 'off_market' / path.parent.name / 'schema.json').read_text())
            population = template.get('training_types', ['Condo Apt', 'Co-Op Apt'])
            branch_ids = pd.read_csv(path.parent / 'train_ids.csv')
            expected_ids = train.loc[train.house_type_name.isin(population)]
            np.testing.assert_array_equal(branch_ids.ml_num, expected_ids.ml_num)
            np.testing.assert_array_equal(pd.to_datetime(branch_ids.date_sold), expected_ids.date_sold)
            assert s['training_n'] == len(expected_ids)
            assert s['cutoff_exclusive'] == str(cutoff.date())
            assert s['params'] == template['params'] and s['features'] == template['features']
            if 'num_boost_round' in template:
                assert s['num_boost_round'] == template['num_boost_round']
        ls = json.loads((folder / 'list_aware' / 'schema.json').read_text())
        lt = json.loads((ROOT / 'saved_models' / 'list_aware' / 'schema.json').read_text())
        assert ls['training_n'] == len(train) and ls['cutoff_exclusive'] == str(cutoff.date())
        for key in ['params', 'features', 'num_boost_round', 'target']:
            assert ls[key] == lt[key]
        rows = pd.read_csv(folder / 'predictions.csv')
        assert len(list(folder.rglob('schema.json'))) == 7
        assert sha(folder / 'predictions.csv') == audit['prediction_sha256']
        assert sha(folder / 'train_ids.csv') == audit['train_ids_sha256']
        for track in ['off_market', 'list_aware']:
            p = rows.loc[rows.track.eq(track)]
            assert not p.ml_num.duplicated().any() and set(p.ml_num) == set(homes.ml_num)
            assert set(p.month) == {str(month)} and set(p.cutoff_exclusive) == {str(cutoff.date())}
            assert not set(p.ml_num) & set(train.ml_num)
            np.testing.assert_array_equal(p.sold_price, homes.set_index('ml_num').reindex(p.ml_num).sold_price)
        all_folds.append(rows)
    predictions = pd.read_csv(out / 'validation_predictions.csv')
    pd.testing.assert_frame_equal(predictions, pd.concat(all_folds, ignore_index=True))
    scores = json.loads((out / 'metrics_precise.json').read_text())
    official = json.loads((out / 'official_metrics.json').read_text())
    for track in ['off_market', 'list_aware']:
        p = predictions.loc[predictions.track.eq(track)]
        ape = np.abs(p.predicted_price.to_numpy() / p.sold_price.to_numpy() - 1) * 100
        assert not p.ml_num.duplicated().any() and scores[track]['n'] == len(p)
        np.testing.assert_allclose([np.median(ape), np.mean(ape)],
            [scores[track]['mdape'], scores[track]['mape']], rtol=0, atol=1e-11)
        assert official[track]['mdape'] == round(scores[track]['mdape'], 2)
        assert official[track]['mape'] == round(scores[track]['mape'], 2)
        # February has the identical training cutoff/pool to the current-best
        # fixed Feb-May archive: the new trainer must reproduce it exactly.
        archived = pd.read_csv(ROOT / 'evidence' / track / 'predictions_unscored.csv')
        february = p.loc[p.month.eq('2026-02')]
        original = archived.set_index('ml_num').predicted_price.reindex(february.ml_num)
        np.testing.assert_array_equal(february.predicted_price, original)
    print('Verified: 6 monthly cutoffs; 42 refitted model schemas; exact training/population IDs; '
          'all prediction IDs/truth; frozen parameters; artifact hashes; independently pooled metrics; '
          'February predictions exactly reproduce indexed current-best models.')
    print(json.dumps(official, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    if args.data_dir:
        os.environ['DATA_DIR'] = str(args.data_dir.resolve())
    verify(args.output.resolve())
