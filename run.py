"""Predict with frozen final models, check CSVs, or refit official monthly folds.

python run.py validate --data-dir /path/to/data --output /new/validation/folder
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

# Keep final replay portable and bound the numerical-library thread pools.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.plan_a.features import HistoryFeatures
from experiments.property_groups.models import GroupModel, LOT_COLUMNS
from experiments.residual_refinement.features import enrich
from experiments.data_processing_round1.features import ProcessingFeatures, assemble, text_features
from experiments.lightgbm_official.models import OfficialLightGBM
from experiments.list_aware_v1.features import FeatureBuilder
from experiments.list_aware_v1.models import RatioModel

ROOT = Path(__file__).resolve().parent
CUTOFF = '2026-06-01'


def off_market(reference, homes):
    hidden = data.hide_answers(homes, 'off_market').reset_index(drop=True)
    provider = HistoryFeatures(reference, CUTOFF)
    base = provider.transform(hidden)
    land = hidden.loc[hidden.house_type_name.isin(['Detached', 'Semi-Detached'])].reset_index(drop=True)
    processing = ProcessingFeatures(reference, CUTOFF)
    enhanced = assemble(provider.transform(land), processing.history(land),
                        processing.land(land), text_features(land), 'all_three')
    predictions = pd.Series(np.nan, index=hidden.ml_num, dtype=float)
    for kind in ['Detached', 'Semi-Detached', 'Freehold Townhouse', 'Condo Townhouse']:
        query, features = (land, enhanced) if kind in ['Detached', 'Semi-Detached'] else (hidden, base)
        mask = query.house_type_name.eq(kind)
        x = features.loc[mask]
        if kind == 'Condo Townhouse':
            x = x.drop(columns=list(LOT_COLUMNS))
        model = OfficialLightGBM.load(ROOT/'saved_models/off_market'/kind)
        predictions.loc[query.loc[mask, 'ml_num']] = model.predict(x)
    # The historical apartment_text branch enriches BASE features, not corrected history.
    for kind in ['Condo Apt', 'Co-Op Apt']:
        mask = hidden.house_type_name.eq(kind)
        x = base.loc[mask].reset_index(drop=True)
        if kind == 'Co-Op Apt':
            x = enrich(x, hidden.loc[mask].reset_index(drop=True))
        model = GroupModel.load_bundle(ROOT/'saved_models/off_market'/kind)
        predictions.loc[hidden.loc[mask, 'ml_num']] = model.predict(x)
    values = predictions.reindex(hidden.ml_num).to_numpy()
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError('Unknown property type or invalid Off-market prediction')
    return pd.DataFrame({'ml_num': hidden.ml_num, 'predicted_price': values})


def predict(output):
    raw = data.load('listings')
    if raw.ml_num.duplicated().any() or not (raw.date_sold < pd.Timestamp(CUTOFF)).all():
        raise ValueError('Training IDs or final training cutoff differ')
    reference = raw.sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    outputs = {}
    expected = json.loads((ROOT/'CURRENT_BEST_MODEL.json').read_text())
    for track in ['off_market', 'list_aware']:
        homes = data.load(track)
        if set(homes.ml_num) & set(raw.ml_num):
            raise ValueError('Train/test IDs overlap')
        print(f'Rebuilding {track} features and predicting {len(homes)} homes...', flush=True)
        if track == 'off_market':
            result = off_market(reference, homes)
        else:
            hidden = data.hide_answers(homes, track).reset_index(drop=True)
            x = FeatureBuilder(reference, CUTOFF).transform(hidden)
            values = RatioModel.load(ROOT/'saved_models/list_aware').predict(x, hidden)
            result = pd.DataFrame({'ml_num': hidden.ml_num, 'predicted_price': values})
        payload = result.to_csv(index=False).encode('utf-8')
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected[track+'_fixed_four_months']['final_predictions_sha256']:
            raise ValueError(f'{track} replay differs from selected archive: {digest}')
        # Official checker validates columns, IDs, row coverage, finite positive prices.
        problems = check_submission.check_file(io.BytesIO(payload), homes.ml_num)
        if problems:
            raise ValueError(problems)
        outputs[track] = payload
    # Write only after both complete forecasts match their archived hashes.
    output.mkdir(parents=True, exist_ok=True)
    for track, payload in outputs.items():
        target = output/f'submission_{track}.csv'
        target.write_bytes(payload)
        print(f'{target.name}: exact archived bytes; official format OK', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['predict', 'check', 'validate'])
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.data_dir is not None:
        os.environ['DATA_DIR'] = str(args.data_dir.resolve())
    if args.command == 'validate':
        import sys
        from validate_official import main as validate_main
        arguments = [sys.argv[0]]
        if args.data_dir is not None:
            arguments += ['--data-dir', str(args.data_dir)]
        if args.output is not None:
            arguments += ['--output', str(args.output)]
        sys.argv = arguments
        validate_main()
    elif args.command == 'predict':
        predict(args.output or ROOT/'submissions')
    else:
        if not check_submission.check_all(args.output or ROOT/'submissions'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
