"""Independent array metrics plus raw-feature rebuild and saved-model replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.list_aware_v1.features import FeatureBuilder
from experiments.list_aware_v1.models import RatioModel
from experiments.list_aware_v1.run_experiment import load_data, metric, log, sha, write_json
from .models import TunedRatioModel
from .run_experiment import INNER, CHECK, FINAL, select_trial, paired_comparison


def replay(run, raw, homes, cutoff, stage):
    train = raw.loc[raw.date_sold < cutoff].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    hidden = data.hide_answers(homes, 'list_aware').reset_index(drop=True)
    x = FeatureBuilder(train, cutoff).transform(hidden)
    pd.testing.assert_frame_equal(x, pd.read_parquet(Path(run)/stage/'prediction_features.parquet'))
    return TunedRatioModel.load(Path(run)/stage/'model').predict(x, hidden)


def verify(run, data_dir):
    run = Path(run)
    protocol = json.loads((run/'protocol.json').read_text())
    for path, digest in {**protocol['data_hashes'], **protocol['source_hashes'], **protocol['protected_hashes']}.items():
        if sha(path) != digest:
            raise ValueError(f'Protected input/source changed: {path}')
    if protocol['inner_window'] != list(INNER) or protocol['check_window'] != list(CHECK):
        raise ValueError('Unexpected windows')
    raw, test = load_data(data_dir)
    selection = json.loads((run/'selected_config.json').read_text())
    trials = json.loads((run/'tuning/trials.json').read_text())
    best_trial = select_trial(trials)
    if best_trial['config'] != selection['config'] or selection['selection_window'] != list(INNER):
        raise ValueError('Frozen config differs from best inner trial')
    archive = Path(protocol['archive'])
    for stage, cutoff, end, archive_folder in [
            ('inner', *INNER, archive/'development'/INNER[0]),
            ('check', *CHECK, archive/'check'), ('final', FINAL, None, archive/'final')]:
        folder = run/stage
        train = raw.loc[raw.date_sold < cutoff].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
        ids = pd.read_csv(folder/'train_ids.csv')
        np.testing.assert_array_equal(ids.ml_num, train.ml_num)
        np.testing.assert_array_equal(pd.to_datetime(ids.date_sold), train.date_sold)
        if not (train.date_sold < cutoff).all():
            raise ValueError('Training cutoff violation')
        homes = test if end is None else raw.loc[(raw.date_sold >= cutoff) & (raw.date_sold < end)]
        np.testing.assert_array_equal(pd.read_csv(folder/'predict_ids.csv').ml_num, homes.ml_num)
        audit = json.loads((folder/'audit.json').read_text())
        if sha(folder/'train_features.parquet') != audit['train_features_sha256']:
            raise ValueError('Training cache hash changed')
        if sha(folder/'prediction_features.parquet') != audit['predict_features_sha256']:
            raise ValueError('Prediction cache hash changed')
        if stage != 'inner':
            model = TunedRatioModel.load(folder/'model')
            if model.config != selection['config']:
                raise ValueError('Saved model uses different frozen parameters')
            schema = json.loads((folder/'model/schema.json').read_text())
            if schema['cutoff_exclusive'] != cutoff or schema['training_n'] != len(train):
                raise ValueError('Saved model training metadata mismatch')
            np.testing.assert_array_equal(model.predict(pd.read_parquet(folder/'prediction_features.parquet'),
                                                       data.hide_answers(homes, 'list_aware').reset_index(drop=True)),
                pd.read_csv(folder/'predictions_unscored.csv').predicted_price)
        else:
            baseline = pd.read_csv(folder/'baseline_predictions.csv')
            score = metric(homes.sold_price, baseline.predicted_price)
            np.testing.assert_allclose([score['mdape'], score['mape']],
                                       [trials[0]['mdape'], trials[0]['mape']], rtol=0, atol=1e-12)
        # Replay the incumbent using rebuilt query features, independently of
        # the feature caches generated during training.
        log(f'核验 {stage}：从原始数据重建预测特征')
        hidden = data.hide_answers(homes, 'list_aware').reset_index(drop=True)
        x = FeatureBuilder(train, cutoff).transform(hidden)
        pd.testing.assert_frame_equal(x, pd.read_parquet(folder/'prediction_features.parquet'))
        old_values = RatioModel.load(archive_folder/'global_lgbm').predict(x, hidden)
        np.testing.assert_array_equal(old_values, pd.read_csv(folder/'baseline_predictions.csv').predicted_price)
        if stage != 'inner':
            np.testing.assert_array_equal(TunedRatioModel.load(folder/'model').predict(x, hidden),
                pd.read_csv(folder/'predictions_unscored.csv').predicted_price)
    homes = raw.loc[(raw.date_sold >= CHECK[0]) & (raw.date_sold < CHECK[1])]
    baseline = pd.read_csv(run/'check/baseline_predictions.csv')
    prediction = pd.read_csv(run/'check/predictions_unscored.csv')
    pair, table = paired_comparison(homes, baseline, prediction)
    pd.testing.assert_frame_equal(table, pd.read_csv(run/'comparison.csv'), check_dtype=False)
    pd.testing.assert_frame_equal(pair.assign(date_sold=pair.date_sold.astype(str)),
        pd.read_csv(run/'paired_predictions.csv'), check_dtype=False)
    # Independent formula, applied to paired rows rather than the report table.
    a = np.abs(pair.predicted_price.to_numpy()/pair.sold_price.to_numpy()-1)*100
    b = np.abs(pair.baseline_price.to_numpy()/pair.sold_price.to_numpy()-1)*100
    np.testing.assert_allclose([np.median(a), np.mean(a), np.median(b), np.median(a)-np.median(b)],
        table.iloc[0][['candidate_mdape','candidate_mape','baseline_mdape','delta_mdape_pp']].to_numpy(float),
        rtol=0, atol=1e-12)
    issues = check_submission.check_file(run/'final_predictions/submission_list_aware_flaml.csv', test.ml_num)
    if issues:
        raise ValueError(issues)
    final = pd.read_csv(run/'final_predictions/submission_list_aware_flaml.csv')
    np.testing.assert_array_equal(final.ml_num, test.ml_num)
    np.testing.assert_array_equal(final.predicted_price, pd.read_csv(run/'final/predictions_unscored.csv').predicted_price)
    return dict(hashes_unchanged=True, inner_only_selection=True, original_parameters_reproduced=True,
        train_ids_cutoffs=True, frozen_config_matches_models=True, raw_query_features_rebuilt=True,
        saved_incumbent_and_candidate_replay_exact=True, paired_metrics=True, independent_numpy_metrics=True,
        official_submission_check=True, final_rows=len(final))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    result = verify(args.run, args.data_dir or Path(protocol['data_dir']))
    write_json(args.run/'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
