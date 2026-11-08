"""Recompute metrics from raw labels and replay every saved model independently."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
from experiments.dart_dropout.models import DropoutModel
from experiments.data_processing_round1.run_experiment import load_data
from .models import PARAMS, XGBoostPriceModel
from .run_experiment import (REPO, WINDOWS, FINAL_CUTOFF, CANDIDATE, BASELINE, PROCESSING, RESIDUAL,
    load_designs, archived_baseline, fresh_final_predict, sha, write_json, now)


def independent_metrics(rows):
    error = 100*(rows.predicted_price.to_numpy()/rows.sold_price.to_numpy()-1)
    return dict(mdape=np.median(np.abs(error)), mape=np.mean(np.abs(error)),
                median_signed_error=np.median(error), over_30_pct=100*np.mean(np.abs(error)>30))


def verify(run, data_dir):
    run = Path(run).resolve()
    protocol = json.loads((run/'protocol.json').read_text())
    freeze = json.loads((run/'prediction_freeze.json').read_text())
    assert sha(run/'protocol.json') == freeze['protocol_sha256']
    assert sha(run/'all_predictions_unscored.csv') == freeze['unscored_sha256']
    for mapping in ('source_sha256', 'input_sha256'):
        for path, digest in protocol[mapping].items():
            assert sha(path) == digest, path
    raw, test = load_data(data_dir)
    stored = pd.read_csv(run/'all_predictions_scored.csv')
    unscored = pd.read_csv(run/'all_predictions_unscored.csv')
    pd.testing.assert_frame_equal(stored[list(unscored)], unscored)
    assert not stored.duplicated(['cutoff_exclusive', 'ml_num', 'candidate']).any()
    truth = raw.set_index('ml_num')
    np.testing.assert_array_equal(stored.sold_price, truth.sold_price.reindex(stored.ml_num))
    np.testing.assert_array_equal(pd.to_datetime(stored.date_sold), truth.date_sold.reindex(stored.ml_num))
    assert (pd.to_datetime(stored.date_sold) >= pd.to_datetime(stored.cutoff_exclusive)).all()
    end_map = dict(WINDOWS)
    assert (pd.to_datetime(stored.date_sold) < pd.to_datetime(stored.cutoff_exclusive.map(end_map))).all()
    dimensions = dict(overall=['protocol'], type=['protocol', 'house_type_name'],
                      window=['protocol', 'cutoff_exclusive'],
                      type_window=['protocol', 'cutoff_exclusive', 'house_type_name'],
                      month=['protocol', 'sold_month'])
    scores = pd.read_csv(run/'scores.csv')
    for _, row in scores.iterrows():
        mask = stored.candidate.eq(row.candidate)
        for column in dimensions[row.level]:
            mask &= stored[column].eq(row[column])
        part = stored.loc[mask]
        assert len(part) == row.n
        for key, value in independent_metrics(part).items():
            assert np.isclose(row[key], value, atol=1e-10, rtol=0)
    comparison = pd.read_csv(run/'comparison_to_current_best.csv')
    for _, row in comparison.iterrows():
        mask = pd.Series(True, index=stored.index)
        for column in dimensions[row.level]:
            mask &= stored[column].eq(row[column])
        part = stored.loc[mask]
        a = part.loc[part.candidate.eq(BASELINE)].set_index('ml_num').sort_index()
        b = part.loc[part.candidate.eq(CANDIDATE)].set_index('ml_num').sort_index()
        assert a.index.equals(b.index) and len(a) == row.n
        for key, value in independent_metrics(a).items():
            other = independent_metrics(b)[key]
            assert np.isclose(row['current_best_'+key], value, atol=1e-10, rtol=0)
            assert np.isclose(row['xgboost_'+key], other, atol=1e-10, rtol=0)
            assert np.isclose(row['delta_'+key+'_pp'], other-value, atol=1e-10, rtol=0)
    count = controls = 0
    for cutoff, end in [*WINDOWS, (FINAL_CUTOFF, None)]:
        homes, designs = load_designs(raw, test, cutoff, end)
        rows = pd.read_csv(run/'windows'/cutoff/'predictions_unscored.csv')
        baseline = archived_baseline(homes, cutoff, final=end is None).set_index('ml_num')
        for _, candidate in rows.groupby('candidate'):
            assert not candidate.ml_num.duplicated().any() and set(candidate.ml_num) == set(homes.ml_num)
            apartment = candidate.house_type_name.isin(['Condo Apt', 'Co-Op Apt'])
            np.testing.assert_array_equal(candidate.loc[apartment, 'predicted_price'],
                baseline.predicted_price.reindex(candidate.loc[apartment, 'ml_num']))
        for kind, (train, valid, xt, xv) in designs.items():
            assert (train.date_sold < pd.Timestamp(cutoff)).all()
            assert not set(train.ml_num) & set(valid.ml_num)
            folder = run/'windows'/cutoff/kind/CANDIDATE
            schema = json.loads((folder/'schema.json').read_text())
            assert schema['params'] == PARAMS
            assert schema['cutoff_exclusive'] == cutoff and schema['training_n'] == len(train)
            assert pd.Timestamp(schema['max_train_sale']) == train.date_sold.max()
            assert schema['init_log_price'] == float(np.median(np.log(train.sold_price)))
            model = XGBoostPriceModel.load(folder)
            assert model.model.get_booster().num_boosted_rounds() == 600
            native = json.loads(model.model.get_booster().save_config())['learner']
            assert native['objective']['name'] == 'reg:absoluteerror'
            assert native['gradient_booster']['name'] == 'gbtree'
            expected = rows.loc[rows.candidate.eq(CANDIDATE)].set_index('ml_num').predicted_price.reindex(valid.ml_num)
            np.testing.assert_array_equal(model.predict(xv), expected)
            branch = pd.read_csv(folder/'predictions_unscored.csv')
            np.testing.assert_array_equal(branch.ml_num, valid.ml_num)
            np.testing.assert_array_equal(branch.predicted_price, expected)
            if end is not None:
                subset = stored.loc[stored.cutoff_exclusive.eq(cutoff) & stored.candidate.eq(CANDIDATE)]
                np.testing.assert_array_equal(subset.set_index('ml_num').predicted_price.reindex(valid.ml_num), expected)
                control = DropoutModel.load(run/'windows'/cutoff/kind/'lightgbm_control')
                np.testing.assert_array_equal(control.predict(xv), baseline.predicted_price.reindex(valid.ml_num))
                controls += 1
            count += 1
    assert count == 20 and controls == 16
    path = run/'final_predictions/submission_off_market_xgboost.csv'
    final = pd.read_csv(path)
    assert not check_submission.check_file(path, test.ml_num)
    pd.testing.assert_frame_equal(final, fresh_final_predict(raw, test, run))
    before = json.loads((run/'current_best_before.json').read_text())
    overall = comparison.loc[comparison.level.eq('overall') & comparison.protocol.eq('check')].iloc[0]
    assert np.isclose(overall.current_best_mdape,
        before['off_market_fixed_four_months']['metrics_percent']['mdape'], rtol=0, atol=1e-10)
    assert int(overall.n) == 7773
    result = dict(verified_at=now(), metric_rows=len(scores), comparison_rows=len(comparison),
        xgboost_models_replayed=count, lightgbm_controls_replayed=controls,
        raw_labels_time_boundaries_ids_checked=True, raw_final_feature_replay_exact=True,
        current_best_baseline_reproduced=True, final_n=len(final), protected_hashes_unchanged=True)
    write_json(run/'independent_verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def update_best(run):
    """Archive prior record; update only after a complete independent pass."""
    run = Path(run).resolve()
    if not (run/'independent_verification.json').exists():
        raise ValueError('Independent verification required')
    current = json.loads((REPO/'CURRENT_BEST_MODEL.json').read_text())
    before = json.loads((run/'current_best_before.json').read_text())
    frame = pd.read_csv(run/'comparison_to_current_best.csv')
    row = frame.loc[frame.level.eq('overall') & frame.protocol.eq('check')].iloc[0]
    model_id = f'xgboost/{run.name}/{CANDIDATE}'
    improved = row.xgboost_mdape < row.current_best_mdape
    previous = current['off_market_fixed_four_months']
    if improved and previous['model_id'] != model_id:
        if current != before:
            raise ValueError('Current best changed since experiment; compare before updating')
        prefix = str(run.relative_to(REPO))
        evidence = run/'all_predictions_scored.csv'
        final = run/'final_predictions/submission_off_market_xgboost.csv'
        current['recorded_date'] = now()[:10]
        current['off_market_fixed_four_months'] = dict(
            model_id=model_id, display_name='固定XGBoost Log-L1四分支＋既有公寓路由',
            training_cutoff_exclusive='2026-02-01', evaluation_start_inclusive='2026-02-01',
            evaluation_end_exclusive='2026-06-01', reference_pool_updates_within_evaluation=False,
            selection_metric='MdAPE', metrics_percent=dict(n=int(row.n), mdape=row.xgboost_mdape, mape=row.xgboost_mape),
            prediction_evidence=f'{prefix}/all_predictions_scored.csv',
            prediction_evidence_filter=dict(protocol='check', candidate=CANDIDATE),
            prediction_evidence_sha256=sha(evidence),
            final_predictions=f'{prefix}/final_predictions/submission_off_market_xgboost.csv',
            final_predictions_sha256=sha(final),
            previous_best_record=f'{prefix}/current_best_before.json',
            comparison_evidence=f'{prefix}/comparison_to_current_best.csv',
            independent_verification=f'{prefix}/independent_verification.json',
            lineage=[prefix, str(PROCESSING.relative_to(REPO)), str(RESIDUAL.relative_to(REPO))],
            comparison_note='完整固定方案同ID累计开发回测；未搜索房型赢家，未覆盖正式提交，隐藏测试未知。')
        write_json(REPO/'CURRENT_BEST_MODEL.json', current)
    write_json(run/'acceptance.json', dict(recorded_at=now(),
        previous_model=before['off_market_fixed_four_months']['model_id'],
        previous_mdape=row.current_best_mdape, candidate_mdape=row.xgboost_mdape,
        delta_mdape_pp=row.delta_mdape_pp, improved=bool(improved),
        best_index_updated=bool(improved), official_submissions_replaced=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--update-best', action='store_true')
    args = parser.parse_args()
    verify(args.run, args.data_dir)
    if args.update_best:
        update_best(args.run)


if __name__ == '__main__':
    main()
