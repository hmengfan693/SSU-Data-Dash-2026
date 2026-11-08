"""Recompute paired errors and rebuild/replay check and final inputs from raw data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from .features import FeatureBuilder, RatioHistory, historical_prediction, list_prices
from .models import GROUPS, RatioModel, grouped_predict
from .run_experiment import (CANDIDATES, CHECK, DEV, FINAL, metric, load_data, select_candidate,
                             score_table, sha, write_json, log)


def replay(raw, homes, cutoff, folder, candidates):
    hidden = data.hide_answers(homes, 'list_aware').reset_index(drop=True)
    reference = raw.loc[raw.date_sold < cutoff].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    result = {}
    if 'list_price' in candidates:
        result['list_price'] = list_prices(hidden)
    if 'historical_ratio' in candidates:
        result['historical_ratio'] = historical_prediction(hidden, RatioHistory(reference, cutoff).transform(hidden))
    if set(candidates) & {'global_lgbm', 'grouped_lgbm'}:
        x = FeatureBuilder(reference, cutoff).transform(hidden)
        pd.testing.assert_frame_equal(x, pd.read_parquet(folder/'prediction_features.parquet'))
        if 'global_lgbm' in candidates:
            result['global_lgbm'] = RatioModel.load(folder/'global_lgbm').predict(x, hidden)
        if 'grouped_lgbm' in candidates:
            result['grouped_lgbm'] = grouped_predict(
                {g: RatioModel.load(folder/'grouped_lgbm'/g) for g in GROUPS}, x, hidden)
    return result


def verify(run, data_dir):
    run = Path(run)
    protocol = json.loads((run/'protocol.json').read_text())
    for path, expected in {**protocol['data_hashes'], **protocol['source_hashes'], **protocol['protected_hashes']}.items():
        if sha(path) != expected:
            raise ValueError(f'Hash changed: {path}')
    raw, test = load_data(data_dir)
    development = pd.read_csv(run/'development_predictions.csv')
    selected, dev_scores = select_candidate(development, raw)
    selection = json.loads((run/'selection.json').read_text())
    if selected != selection['candidate']:
        raise ValueError('Frozen candidate does not match development selection')
    pd.testing.assert_frame_equal(dev_scores, pd.read_csv(run/'development_comparison.csv'), check_dtype=False)
    for cutoff, end in [*DEV, CHECK, (FINAL, None)]:
        folder = run/('final' if cutoff == FINAL else 'check' if (cutoff, end) == CHECK else f'development/{cutoff}')
        train = raw.loc[raw.date_sold < cutoff].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
        ids = pd.read_csv(folder/'train_ids.csv')
        np.testing.assert_array_equal(ids.ml_num, train.ml_num)
        if not (pd.to_datetime(ids.date_sold) < pd.Timestamp(cutoff)).all():
            raise ValueError('Training cutoff violated')
        homes = test if end is None else raw.loc[(raw.date_sold >= cutoff) & (raw.date_sold < end)]
        actual = pd.read_csv(folder/'predictions_unscored.csv')
        candidates = [selected] if end is None else CANDIDATES
        for candidate in candidates:
            p = actual.loc[actual.candidate == candidate]
            np.testing.assert_array_equal(p.ml_num, homes.ml_num)
            if not (p.cutoff_exclusive == cutoff).all():
                raise ValueError('Prediction cutoff changed')
        for schema_file in folder.rglob('schema.json'):
            schema = json.loads(schema_file.read_text())
            expected_n = len(train)
            if schema_file.parent.parent.name == 'grouped_lgbm':
                group = schema_file.parent.name
                expected_n = int(train.house_type_name.isin(GROUPS[group]).sum())
            if schema['cutoff_exclusive'] != cutoff or schema['training_n'] != expected_n:
                raise ValueError('Model training audit mismatch')
    homes = raw.loc[(raw.date_sold >= CHECK[0]) & (raw.date_sold < CHECK[1])]
    predictions = pd.read_csv(run/'check/predictions_unscored.csv')
    scores = score_table(predictions, homes)
    pd.testing.assert_frame_equal(scores, pd.read_csv(run/'check_comparison.csv'), check_dtype=False)
    # Independent array calculation for the selected model, rather than relying
    # only on the same table writer used by the training process.
    paired = homes[['ml_num', 'sold_price']].merge(predictions.loc[predictions.candidate == selected],
                                                on='ml_num', validate='one_to_one')
    ape = np.abs(paired.predicted_price.to_numpy()-paired.sold_price.to_numpy())/paired.sold_price.to_numpy()*100
    recorded = scores.loc[(scores.level == 'overall') & (scores.candidate == selected)].iloc[0]
    np.testing.assert_allclose([np.median(ape), np.mean(ape)], [recorded.mdape, recorded.mape], rtol=0, atol=1e-12)
    log('独立核验：从原始数据重建检查窗口特征并重放全部候选')
    for candidate, values in replay(raw, homes, CHECK[0], run/'check', CANDIDATES).items():
        np.testing.assert_array_equal(values, predictions.loc[predictions.candidate == candidate].predicted_price)
    log('独立核验：从原始数据重建最终特征并重放 4064 套预测')
    final_values = replay(raw, test, FINAL, run/'final', [selected])[selected]
    submission = pd.read_csv(run/'final_predictions/submission_list_aware.csv')
    np.testing.assert_array_equal(test.ml_num, submission.ml_num)
    np.testing.assert_array_equal(final_values, submission.predicted_price)
    issues = check_submission.check_file(run/'final_predictions/submission_list_aware.csv', test.ml_num)
    if issues:
        raise ValueError(issues)
    return dict(input_source_protected_hashes=True, train_ids_and_cutoffs=True,
                selection_2025_only=True, paired_metrics=True, independent_numpy_metrics=True,
                raw_check_feature_and_model_replay=True, raw_final_replay=True,
                official_submission_check=True, final_rows=len(test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    checks = verify(args.run, args.data_dir or Path(protocol['data_dir']))
    write_json(args.run/'verification.json', checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
