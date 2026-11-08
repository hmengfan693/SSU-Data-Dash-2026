"""Check chronological tuning, saved large trees and raw-input final replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
from experiments.dart_dropout.run_experiment import REPO, TARGETS, sha, now, write_json, load_designs
from experiments.data_processing_round1.run_experiment import load_data
from experiments.flaml_tuning.models import price_metrics, validate_window, replace_branch
from experiments.flaml_tuning.run_experiment import comparison
from experiments.flaml_tuning.verify_results import predict_saved
from experiments.lightgbm_official.models import OfficialLightGBM
from .models import complexity, make_model
from .run_experiment import INNER_WINDOWS, OUTER, FINAL, BEST


def verify_run(run, data_dir):
    run = Path(run).resolve()
    protocol = json.loads((run/'protocol.json').read_text())
    for key in ('source_sha256', 'input_sha256', 'protected_submissions'):
        for name, digest in protocol[key].items():
            if sha(name) != digest:
                raise ValueError(f'Changed {key}: {name}')
    completion = json.loads((run/'completion.json').read_text())
    if completion['selected_sha256'] != sha(run/'selected_configs.json'):
        raise ValueError('Frozen parameter hash changed')
    selection = json.loads((run/'selected_configs.json').read_text())
    if selection['outer_labels_used'] or selection['frozen_at'] >= completion['completed_at']:
        raise ValueError('Invalid freeze audit')
    raw, test = load_data(data_dir)
    raw_by_id = raw.set_index('ml_num')
    for target in TARGETS:
        folder = run/'tuning'/target
        trials = json.loads((folder/'trials.json').read_text())
        if len(trials) != protocol['trials_per_branch']:
            raise ValueError('Incomplete search')
        for cutoff, end in INNER_WINDOWS:
            train = pd.read_csv(folder/'splits'/cutoff/'train_ids.csv')
            valid = pd.read_csv(folder/'splits'/cutoff/'valid_ids.csv')
            validate_window(train, valid, cutoff, end)
            for frame in (train, valid):
                expected = raw_by_id.date_sold.reindex(frame.ml_num).to_numpy()
                if not np.array_equal(expected, pd.to_datetime(frame.date_sold).to_numpy()):
                    raise ValueError('Inner split differs from raw dates')
        for trial in trials:
            truths, predictions = [], []
            for cutoff, end in INNER_WINDOWS:
                rows = pd.read_csv(folder/f'trial_{trial["trial"]:03d}'/cutoff/'predictions.csv')
                expected = raw_by_id.reindex(rows.ml_num)
                ids = pd.read_csv(folder/'splits'/cutoff/'valid_ids.csv').ml_num
                if not np.array_equal(rows.ml_num.to_numpy(), ids.to_numpy()):
                    raise ValueError('Inner trial cohort mismatch')
                if not (expected.house_type_name.eq(target).all()
                        and np.array_equal(rows.sold_price.to_numpy(), expected.sold_price.to_numpy())):
                    raise ValueError('Inner trial labels do not match raw data')
                truths.extend(rows.sold_price)
                predictions.extend(rows.predicted_price)
            actual = price_metrics(truths, predictions)
            if not np.isclose(actual['mdape'], trial['mdape'], atol=1e-10, rtol=0):
                raise ValueError('Pooled inner MdAPE mismatch')
        winner = min(trials, key=lambda r: (r['mdape'], r['mape'], r['trial']))
        rounds = int(np.floor(np.median([f['selected_rounds'] for f in winner['folds']])+.5))
        if selection['configs'][target] != dict(winner['config'], n_estimators=rounds):
            raise ValueError('Selected config not determined by earlier windows')
        if selection['configs'][target]['num_leaves'] not in [127, 255, 511]:
            raise ValueError('Complex candidate did not use larger leaf capacity')

    for cutoff, end, model_dir in [(*OUTER, 'models'), (FINAL, None, 'final_models')]:
        homes, designs = load_designs(raw, test, cutoff, end)
        baseline = pd.read_csv(run/model_dir/'baseline_predictions.csv')
        current_info = protocol['current_best']
        evidence = REPO/current_info['final_predictions' if end is None else 'prediction_evidence']
        old = pd.read_csv(evidence)
        if (set(old.ml_num) != set(homes.ml_num) or
            not np.array_equal(baseline.ml_num.to_numpy(), homes.ml_num.to_numpy()) or
            not np.array_equal(baseline.predicted_price.to_numpy(),
                               old.set_index('ml_num').predicted_price.reindex(homes.ml_num).to_numpy())):
            raise ValueError('Current-best alignment mismatch')
        combined = baseline.copy()
        audit = pd.read_csv(run/model_dir/'capacity_audit.csv')
        for target in TARGETS:
            train, valid, xt, xv = designs[target]
            expected_params = make_model(selection['configs'][target], protocol['threads']).params
            fitted = OfficialLightGBM.load(run/model_dir/target)
            if fitted.params != expected_params or fitted.num_boost_round != selection['configs'][target]['n_estimators']:
                raise ValueError('Trained parameters differ from frozen configuration')
            ids = pd.read_csv(run/model_dir/target/'train_ids.csv')
            if not np.array_equal(ids.ml_num.to_numpy(), train.ml_num.to_numpy()):
                raise ValueError('Full-data training IDs mismatch')
            expected = pd.read_csv(run/model_dir/target/'predictions.csv')
            values = fitted.predict(xv)
            if not (np.array_equal(expected.ml_num.to_numpy(), valid.ml_num.to_numpy())
                    and np.array_equal(expected.predicted_price.to_numpy(), values)):
                raise ValueError('Saved large-model prediction mismatch')
            actual = complexity(fitted)
            recorded = audit.loc[audit.target.eq(target) & audit.candidate.eq('complex')].iloc[0]
            for key, value in actual.items():
                if not np.isclose(recorded[key], value, rtol=1e-12):
                    raise ValueError('Actual capacity audit mismatch')
            combined = replace_branch(combined, valid, values)
        published = pd.read_csv(run/model_dir/'predictions_unscored.csv')
        if not (np.array_equal(published.ml_num.to_numpy(), combined.ml_num.to_numpy())
                and np.array_equal(published.predicted_price.to_numpy(), combined.predicted_price.to_numpy())):
            raise ValueError('Full candidate routing mismatch')
    homes, _ = load_designs(raw, test, *OUTER)
    baseline = pd.read_csv(run/'models/baseline_predictions.csv')
    candidate = pd.read_csv(run/'models/predictions_unscored.csv')
    paired, table = comparison(homes, baseline, candidate)
    pd.testing.assert_frame_equal(pd.read_csv(run/'comparison_to_current_best.csv'), table,
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(pd.read_csv(run/'paired_predictions.csv').drop(columns='date_sold'),
                                  paired.drop(columns='date_sold'), check_exact=False, rtol=1e-12)
    final_path = run/'final_predictions/submission_off_market_complex.csv'
    problems = check_submission.check_file(final_path, test.ml_num)
    if problems:
        raise ValueError(problems)
    final = pd.read_csv(final_path)
    replay = predict_saved(run, raw, test)
    if not (np.array_equal(final.ml_num.to_numpy(), replay.ml_num.to_numpy())
            and np.array_equal(final.predicted_price.to_numpy(), replay.predicted_price.to_numpy())):
        raise ValueError('Raw data final-model replay mismatch')
    row = table.iloc[0]
    if not np.isclose(row.baseline_mdape, protocol['current_best']['metrics_percent']['mdape'], atol=1e-10, rtol=0):
        raise ValueError('Baseline metric differs from index')
    result = dict(verified_at=now(), n=int(row.n), final_n=len(final),
        baseline_mdape=float(row.baseline_mdape), candidate_mdape=float(row.candidate_mdape),
        delta_mdape_pp=float(row.delta_mdape_pp), candidate_mape=float(row.candidate_mape),
        selected_sha256=completion['selected_sha256'], input_and_source_hashes=True,
        inner_selection_recomputed=True, saved_model_replay=True, actual_capacity_verified=True,
        raw_final_replay=True, final_sha256=sha(final_path),
        prediction_sha256=sha(run/'models/predictions_unscored.csv'))
    write_json(run/'verification.json', result)
    return result


def promote(run, verified):
    run = Path(run).resolve()
    if verified['delta_mdape_pp'] >= 0:
        write_json(run/'promotion.json', dict(promoted=False, reason='Larger trees did not improve matched overall MdAPE'))
        return
    current = json.loads(BEST.read_text())
    before = json.loads((run/'current_best_before.json').read_text())
    model_id = str(run.relative_to(REPO))
    if current['off_market_fixed_four_months']['model_id'] == model_id:
        return
    if current != before:
        raise ValueError('Current-best index changed during experiment')
    info = dict(before['off_market_fixed_four_months'])
    info.update(model_id=model_id, display_name='大容量LightGBM完整方案（双时间窗口选择、既有房型路由）',
        metrics_percent=dict(n=verified['n'], mdape=verified['candidate_mdape'], mape=verified['candidate_mape']),
        prediction_evidence=str((run/'models/predictions_unscored.csv').relative_to(REPO)),
        prediction_evidence_sha256=verified['prediction_sha256'],
        final_predictions=str((run/'final_predictions/submission_off_market_complex.csv').relative_to(REPO)),
        final_predictions_sha256=verified['final_sha256'],
        lineage=[before['off_market_fixed_four_months']['model_id'], model_id],
        comparison_note='127—511叶大容量候选仅在2025年两个窗口调参及早停，冻结后同7773套2—5月开发回测；隐藏成绩未知，未覆盖正式提交。')
    updated = dict(before, recorded_date=now()[:10], off_market_fixed_four_months=info)
    write_json(BEST, updated)
    write_json(run/'promotion.json', dict(promoted=True, promoted_at=now(),
        previous_index='current_best_before.json', delta_mdape_pp=verified['delta_mdape_pp']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run, args.data_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
