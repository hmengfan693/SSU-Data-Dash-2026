"""Check and score archived fixed-horizon development predictions; no model fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    if args.data_dir is not None:
        os.environ['DATA_DIR'] = str(args.data_dir.resolve())
    root = Path(__file__).resolve().parent
    index = json.loads((root / 'CURRENT_BEST_MODEL.json').read_text())
    raw = data.load('listings')
    if raw.ml_num.duplicated().any():
        raise ValueError('Duplicate observed listing IDs')
    for track in ('off_market', 'list_aware'):
        entry = index[track + '_fixed_four_months']
        homes = raw.loc[(raw.date_sold >= entry['evaluation_start_inclusive']) &
                        (raw.date_sold < entry['evaluation_end_exclusive'])]
        path = root / 'evidence' / track / 'predictions_unscored.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['prediction_evidence_sha256']:
            raise ValueError('Archived prediction hash differs: ' + track)
        predicted = pd.read_csv(path)
        if (predicted.ml_num.duplicated().any() or
                set(predicted.ml_num) != set(homes.ml_num)):
            raise ValueError('Archived prediction IDs differ: ' + track)
        values = predicted.set_index('ml_num').predicted_price.reindex(homes.ml_num).to_numpy(float)
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError('Invalid archived prices: ' + track)
        errors = np.abs(values / homes.sold_price.to_numpy(float) - 1) * 100
        mdape, mape = np.median(errors), np.mean(errors)
        expected = entry['metrics_percent']
        if len(homes) != expected['n']:
            raise ValueError('Development row count differs')
        np.testing.assert_allclose([mdape, mape], [expected['mdape'], expected['mape']],
                                   rtol=0, atol=1e-10)
        print(f'{track}: n={len(homes)}; MdAPE={mdape:.9f}%; MAPE={mape:.9f}%; '
              'matches indexed current best (delta 0 pp).')
    print('Reused fixed-horizon development backtests; not hidden-test or monthly-refit scores.')


if __name__ == '__main__':
    main()
