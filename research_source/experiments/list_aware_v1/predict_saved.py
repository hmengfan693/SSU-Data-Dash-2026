"""Rebuild features and replay the frozen final model without retraining."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_experiment import FINAL, load_data
from .verify_results import replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    raw, test = load_data(args.data_dir or Path(protocol['data_dir']))
    candidate = json.loads((args.run/'selection.json').read_text())['candidate']
    prices = replay(raw, test, FINAL, args.run/'final', [candidate])[candidate]
    archived = pd.read_csv(args.run/'final_predictions/submission_list_aware.csv')
    np.testing.assert_array_equal(test.ml_num, archived.ml_num)
    np.testing.assert_array_equal(prices, archived.predicted_price)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ml_num': test.ml_num, 'predicted_price': prices}).to_csv(args.output, index=False)
    print(f'重放一致：{len(test)} 行 → {args.output}')


if __name__ == '__main__':
    main()
