"""Rebuild hidden-test features from raw inputs and load saved models only."""
import argparse
from pathlib import Path
from .run_experiment import infer_saved, load_data, REPO, ARCHIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw, test = load_data(args.data_dir)
    infer_saved(raw, test, args.run, args.archive).to_csv(args.output, index=False)
    print(f'Saved replay: {args.output}', flush=True)


if __name__ == '__main__':
    main()
