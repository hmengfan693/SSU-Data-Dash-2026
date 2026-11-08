"""Rebuild features from raw data and replay the frozen final route."""
import argparse
from pathlib import Path
import os
import data
import check_submission
from .run_experiment import REPO,infer_saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    os.environ['DATA_DIR']=str(REPO.parent/'data')
    raw,test=data.load('listings'),data.load('off_market')
    result,_=infer_saved(raw,test,a.run)
    result.to_csv(a.output,index=False)
    errors=check_submission.check_file(a.output,test.ml_num)
    if errors:raise ValueError(errors)
    print(f'Raw-data feature rebuild and saved-model replay: {len(result):,} rows; official format passed.')


if __name__=='__main__':main()
