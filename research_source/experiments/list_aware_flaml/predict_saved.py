"""Rebuild final features and replay the saved tuned model, without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.list_aware_v1.run_experiment import load_data, FINAL
from .verify_results import replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    raw, test = load_data(args.data_dir or Path(protocol['data_dir']))
    result = replay(args.run, raw, test, FINAL, 'final')
    old = pd.read_csv(args.run/'final_predictions/submission_list_aware_flaml.csv')
    np.testing.assert_array_equal(old.ml_num, test.ml_num)
    np.testing.assert_array_equal(old.predicted_price, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(ml_num=test.ml_num, predicted_price=result)).to_csv(args.output, index=False)
    print(f'独立进程原始数据重放一致：{len(result)} 行 → {args.output}')


if __name__ == '__main__':
    main()
