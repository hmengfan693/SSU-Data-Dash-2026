"""Replay the improved final predictions without training."""
import argparse
from pathlib import Path

from .run_experiment import replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[3]/'data')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result,_=replay(args.run,args.data_dir,args.output)
    print(f'Replayed {len(result):,} predictions: {args.output}')


if __name__=='__main__':main()
