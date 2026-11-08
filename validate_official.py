"""Refit the selected frozen specification on the official six monthly folds.

python validate_official.py --data-dir /path/to/data --output /new/output/folder

Saved final schemas supply ONLY feature definitions, training populations and
fixed parameters. Final weights and category vocabularies are never reused.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import lightgbm as lgb
import numpy as np
import pandas as pd

import data
from experiments.plan_a.features import HistoryFeatures, CATEGORICAL
from experiments.property_groups.models import GroupModel
from experiments.residual_refinement.features import enrich
from experiments.data_processing_round1.features import ProcessingFeatures, assemble, text_features
from experiments.lightgbm_official.models import OfficialLightGBM
from experiments.list_aware_v1.features import FeatureBuilder, list_prices
from experiments.list_aware_v1.models import RatioModel

ROOT = Path(__file__).resolve().parent
MONTHS = pd.period_range('2025-12', '2026-05', freq='M')
LAND = ['Detached', 'Semi-Detached', 'Freehold Townhouse']
APARTMENTS = ['Condo Apt', 'Co-Op Apt']
TRACKS = ['off_market', 'list_aware']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def split_month(raw, month):
    month = pd.Period(month, freq='M')
    cutoff, end = month.start_time, (month + 1).start_time
    train = raw.loc[raw.date_sold < cutoff].sort_values(['date_sold', 'ml_num']).reset_index(drop=True)
    homes = raw.loc[(raw.date_sold >= cutoff) & (raw.date_sold < end)].reset_index(drop=True)
    if (train.empty or homes.empty or raw.ml_num.duplicated().any()
            or set(train.ml_num) & set(homes.ml_num)
            or not (train.date_sold < cutoff).all()
            or not homes.date_sold.between(cutoff, end, inclusive='left').all()):
        raise ValueError('Invalid monthly split')
    return train, homes, str(cutoff.date()), str(end.date())


def score(frame):
    truth = frame.sold_price.to_numpy(float)
    prediction = frame.predicted_price.to_numpy(float)
    if (len(truth) == 0 or not np.isfinite(truth).all() or not (truth > 0).all()
            or not np.isfinite(prediction).all() or not (prediction > 0).all()):
        raise ValueError('Invalid scoring prices')
    signed = 100 * (prediction - truth) / truth
    return dict(n=len(truth), mdape=float(np.median(np.abs(signed))),
                mape=float(np.mean(np.abs(signed))), median_signed_error=float(np.median(signed)))


def official_score(frame):
    """The same rounding and home bootstrap as the official run.py metrics."""
    result = score(frame)
    absolute = (100 * (frame.predicted_price - frame.sold_price) / frame.sold_price).abs().to_numpy()
    for name, statistic in [('mdape', np.median), ('mape', np.mean)]:
        rng = np.random.default_rng(0)
        samples = [statistic(rng.choice(absolute, size=len(absolute))) for _ in range(1000)]
        result[name + '_95ci'] = [round(float(x), 2) for x in np.percentile(samples, [2.5, 97.5])]
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in result.items()}


def schema(track, kind=None):
    folder = ROOT / 'saved_models' / track
    if kind is not None:
        folder /= kind
    return json.loads((folder / 'schema.json').read_text())


def fit_off_market(train, homes, cutoff, folder):
    hidden_train = data.hide_answers(train, 'off_market').reset_index(drop=True)
    hidden = data.hide_answers(homes, 'off_market').reset_index(drop=True)
    assert not set(data.AFTER_SALE_COLUMNS + ['list_price']).intersection(hidden.columns)
    provider = HistoryFeatures(train, cutoff)
    bt, bv = provider.transform(hidden_train), provider.transform(hidden)
    tm, vm = train.house_type_name.isin(LAND), homes.house_type_name.isin(LAND[:2])
    ht, hv = hidden_train.loc[tm].reset_index(drop=True), hidden.loc[vm].reset_index(drop=True)
    processing = ProcessingFeatures(train, cutoff)
    log(f'{cutoff} Off-market: building enhanced features')
    et = assemble(bt.loc[tm], processing.history(ht), processing.land(ht), text_features(ht), 'all_three')
    ev = assemble(bv.loc[vm], processing.history(hv), processing.land(hv), text_features(hv), 'all_three')
    result = pd.Series(np.nan, index=homes.ml_num)
    audits = []
    for kind in [*LAND, 'Condo Townhouse', *APARTMENTS]:
        s = schema('off_market', kind)
        population = s.get('training_types', APARTMENTS)
        tr_mask = train.house_type_name.isin(population)
        va_mask = homes.house_type_name.eq(kind)
        training = train.loc[tr_mask].reset_index(drop=True)
        query = hidden.loc[va_mask].reset_index(drop=True)
        if kind in LAND[:2]:
            xt = et.loc[ht.house_type_name.isin(population)].reset_index(drop=True)
            xv = ev.loc[hv.house_type_name.eq(kind)].reset_index(drop=True)
        else:
            xt, xv = bt.loc[tr_mask].reset_index(drop=True), bv.loc[va_mask].reset_index(drop=True)
            if kind == 'Co-Op Apt':
                xt = enrich(xt, hidden_train.loc[tr_mask].reset_index(drop=True))
                xv = enrich(xv, query)
        xt, xv = xt[s['features']], xv[s['features']]
        destination = folder / kind
        log(f'{cutoff} Off-market: fitting {kind} on {len(training):,} sales')
        if kind in APARTMENTS:
            model = GroupModel(s['scope'], s['algorithm'])
            # The parent fit expects the lot columns to be present and removes
            # them itself. The exact saved feature list has already removed them.
            from experiments.plan_a.models import PriceModel
            if model.model.get_params() != s['params']:
                raise ValueError('Apartment fixed parameters differ from selected schema')
            PriceModel.fit(model, xt, training.sold_price)
            model.save_bundle(destination, cutoff, len(training))
            replay = GroupModel.load_bundle(destination)
        else:
            model = OfficialLightGBM(s['num_boost_round'], **s['params']).fit(xt, training.sold_price)
            model.save(destination, cutoff_exclusive=cutoff, training_n=len(training),
                       training_types=population, target_type=kind)
            replay = OfficialLightGBM.load(destination)
        values = model.predict(xv)
        np.testing.assert_array_equal(values, replay.predict(xv))
        result.loc[query.ml_num] = values
        training[['ml_num', 'date_sold']].to_csv(destination / 'train_ids.csv', index=False)
        audits.append(dict(branch=kind, training_types=population, training_n=len(training),
                           prediction_n=len(query), max_train_sale=str(training.date_sold.max()),
                           saved_model_replay_exact=True))
    values = result.reindex(homes.ml_num).to_numpy()
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError('Incomplete Off-market routing')
    return values, audits


def fit_list_aware(train, homes, cutoff, folder):
    hidden_train = data.hide_answers(train, 'list_aware').reset_index(drop=True)
    hidden = data.hide_answers(homes, 'list_aware').reset_index(drop=True)
    assert not set(data.AFTER_SALE_COLUMNS).intersection(hidden.columns)
    builder = FeatureBuilder(train, cutoff)
    log(f'{cutoff} List-aware: building features')
    xt, xv = builder.transform(hidden_train), builder.transform(hidden)
    s = schema('list_aware')
    xt, xv = xt[s['features']], xv[s['features']]
    if lgb.__version__ != s['lightgbm_version']:
        raise ValueError('List-aware library version differs from selected model')
    model = RatioModel()
    model.params = s['params']
    model.features = list(xt)
    model.categories = {c: sorted(xt[c].dropna().unique().tolist()) for c in CATEGORICAL if c in xt}
    dataset = lgb.Dataset(model.encode(xt), label=np.log(train.sold_price.to_numpy() / list_prices(hidden_train)),
                          categorical_feature=list(model.categories))
    log(f'{cutoff} List-aware: fitting on {len(train):,} sales')
    model.booster = lgb.train(model.params, dataset, num_boost_round=s['num_boost_round'])
    # RatioModel.save records its historical default round count; write the exact
    # selected round count here instead, then use its unchanged loader/predictor.
    folder.mkdir(parents=True)
    model.booster.save_model(str(folder / 'model.txt'))
    write_json(folder / 'schema.json', dict(features=model.features, categories=model.categories,
        params=model.params, num_boost_round=s['num_boost_round'], target=s['target'],
        lightgbm_version=lgb.__version__, cutoff_exclusive=cutoff, training_n=len(train)))
    values = model.predict(xv, hidden)
    np.testing.assert_array_equal(values, RatioModel.load(folder).predict(xv, hidden))
    return values, [dict(branch='global_ratio', training_n=len(train), prediction_n=len(homes),
                        max_train_sale=str(train.date_sold.max()), saved_model_replay_exact=True)]


def make_reports(out, raw, index):
    predictions = pd.read_csv(out / 'validation_predictions.csv')
    summaries, groups, official = {}, [], {}
    for track in TRACKS:
        p = predictions.loc[predictions.track.eq(track)].copy()
        expected = raw.loc[raw.date_sold.between('2025-12-01', '2026-06-01', inclusive='left')]
        if p.ml_num.duplicated().any() or set(p.ml_num) != set(expected.ml_num):
            raise ValueError('Official evaluation cohort mismatch')
        truth = expected.set_index('ml_num').reindex(p.ml_num)
        np.testing.assert_array_equal(p.sold_price, truth.sold_price)
        np.testing.assert_array_equal(p.month, truth.date_sold.dt.strftime('%Y-%m'))
        summaries[track], official[track] = score(p), official_score(p)
        for column in ['month', 'house_type_name', 'community']:
            for name, rows in p.groupby(column, sort=True):
                groups.append(dict(track=track, level=column, group=name, **score(rows)))
    pd.DataFrame(groups).to_csv(out / 'group_metrics.csv', index=False)
    write_json(out / 'metrics_precise.json', summaries)
    write_json(out / 'official_metrics.json', official)
    # Only compare updating rules on the SAME Feb-May homes; Dec-Jan are absent
    # from the indexed fixed-four-month experiment and cannot be compared to it.
    comparisons = []
    for track in TRACKS:
        current = index[track + '_fixed_four_months']
        evidence = ROOT / 'evidence' / ('off_market' if track == 'off_market' else 'list_aware') / 'predictions_unscored.csv'
        archived = pd.read_csv(evidence)
        p = predictions.loc[predictions.track.eq(track) & predictions.month.ge('2026-02')].copy()
        if archived.ml_num.duplicated().any() or set(p.ml_num) != set(archived.ml_num):
            raise ValueError('Matched comparison IDs differ')
        p['fixed_price'] = p.ml_num.map(archived.set_index('ml_num').predicted_price)
        baseline = score(p.assign(predicted_price=p.fixed_price))
        np.testing.assert_allclose([baseline['mdape'], baseline['mape']],
            [current['metrics_percent']['mdape'], current['metrics_percent']['mape']], rtol=0, atol=1e-10)
        for level in ['overall', 'month', 'house_type_name']:
            slices = [('all', p)] if level == 'overall' else p.groupby(level, sort=True)
            for group, rows in slices:
                a, b = score(rows), score(rows.assign(predicted_price=rows.fixed_price))
                comparisons.append(dict(track=track, level=level, group=group, n=len(rows),
                    fixed_mdape=b['mdape'], monthly_mdape=a['mdape'], delta_mdape_pp=a['mdape']-b['mdape'],
                    comparison='Same IDs; different updating/horizon protocols; not a model ranking'))
    pd.DataFrame(comparisons).to_csv(out / 'comparison_fixed_vs_monthly.csv', index=False)
    lines = ['# Official-schedule monthly walk-forward validation', '',
        'The selected frozen models were refitted separately for December 2025 through May 2026. '
        'For each month, training uses every observed sale strictly before the first day of that month; '
        'prediction covers that month only. Parameters, feature definitions and routing remain fixed. '
        'The reference pool is refreshed between months and frozen within each month.', '',
        '| Track | Homes | MdAPE | MAPE | MdAPE 95% bootstrap interval |',
        '| --- | ---: | ---: | ---: | --- |']
    for track in TRACKS:
        s = official[track]
        lines.append(f"| {track.replace('_', '-')} | {s['n']:,} | {s['mdape']:.2f}% | {s['mape']:.2f}% | {s['mdape_95ci'][0]:.2f}%–{s['mdape_95ci'][1]:.2f}% |")
    lines += ['', 'Overall MdAPE is the median of all individual-home percentage errors pooled across '
        'the six months, not an average of monthly or property-type medians. Dollar predictions are '
        'rounded before scoring, matching the official starter. Bootstrap settings also match the starter; '
        'homes are resampled independently, so these intervals do not capture common monthly market shocks.', '',
        '## Monthly results', '', '| Month | Homes per track | Off-market MdAPE | List-aware MdAPE |',
        '| --- | ---: | ---: | ---: |']
    for month in map(str, MONTHS):
        a, b = [score(predictions.loc[predictions.track.eq(t) & predictions.month.eq(month)]) for t in TRACKS]
        lines.append(f"| {month} | {a['n']:,} | {a['mdape']:.2f}% | {b['mdape']:.2f}% |")
    lines += ['', '## Relationship to the indexed current best', '',
        'The models are the indexed complete Off-market property-type boosted-tree specification '
        'and the indexed complete List-aware global LightGBM log sold/list ratio specification. '
        'No new model is selected in this run. Reusing the indexed fixed-four-month protocol would '
        'therefore give the same indexed scores (difference: 0 percentage points). The official-schedule '
        'run changes training cutoff, reference updating and forecast horizon. It cannot be ranked '
        'against the fixed-four-month archive as if the protocols were identical.', '',
        '| Track | Feb–May homes | Indexed fixed-four-month MdAPE | Monthly refit MdAPE on same homes | Difference (pp) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in comparisons:
        if row['level'] == 'overall':
            lines.append(f"| {row['track'].replace('_', '-')} | {row['n']:,} | {row['fixed_mdape']:.4f}% | {row['monthly_mdape']:.4f}% | {row['delta_mdape_pp']:+.4f} |")
    lines += ['', '## Training, lookbacks and leakage boundary', '',
        'The expanding model-training span begins with the observed 2021 sales and ends before each '
        'monthly cutoff. Historical feature lookbacks are separate: price and sold/list summaries use '
        '90, 180 and 365 days; support and land context use their existing frozen definitions. '
        'Historical price queries end strictly before min(listing date, monthly cutoff). Land context '
        'uses a completed prior month and a 365-day lookback. Each forecast horizon is one calendar month. '
        'A one-year historical feature does not mean the model is trained on only one year.', '',
        'Each tree splits homes using feature thresholds or category groups and assigns a small correction '
        'at a leaf. Boosting adds many such corrections. Off-market learns log sale price; List-aware '
        'learns log(sale price / list price), then multiplies the predicted ratio by the known list price. '
        'Both minimize absolute error in the log target. This limits the influence of expensive outliers '
        'and emphasizes multiplicative errors, but is not an exact optimization of MdAPE. Expanding '
        'training keeps scarce property-type examples, while short historical features reflect recent prices.', '',
        'No current-month or future sale labels enter that fold’s model fit, category vocabulary or '
        'reference pool. No validation early stopping is used. Sold price, sale date, days on market '
        'and third-party valuation columns are removed from query inputs; Off-market also removes list price. '
        'The monthly model can have seen sales after a particular home was listed but before the monthly '
        'cutoff, as in the official sale-month protocol. This is not a claim of retraining at each listing date.', '',
        '**Development-use limitation:** the existing configuration-selection window includes September–December '
        '2025 and overlaps the first official validation month. Earlier routing and repeated project decisions '
        'also used development results. These scores are a retrospective, cumulative development validation '
        'under the official split schedule, not a pristine nested or independent test. Fold-local label '
        'isolation does not remove model-selection reuse. Parameters were held fixed during this run; '
        'no claim is made that the December configuration was selected without December outcomes.', '',
        'The June–September 2026 hidden-test prices remain unavailable, so hidden-test MdAPE and competition '
        'rank/points cannot be calculated locally. Model feature relationships are predictive, not causal; '
        'synthetic data and shared monthly market conditions limit generalization.', '',
        '## Reproduction and evidence', '',
        'Run `python run.py validate --data-dir /path/to/data --output /new/output/folder`. '
        'The standalone equivalent is `python validate_official.py --data-dir /path/to/data --output /new/output/folder`. '
        'Every fold saves its newly trained models, training IDs, audit and predictions. '
        '`official_metrics.json` follows official rounding/bootstrap; `metrics_precise.json` retains full precision. '
        '`group_metrics.csv` and `comparison_fixed_vs_monthly.csv` support diagnostics. '
        'Final submission CSVs and original models are not replaced.', '',
        'Official sources: [handout](https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/handout.md), '
        '[validation/scoring implementation](https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/run.py).', '']
    (out / 'validation_report.md').write_text('\n'.join(lines))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'official_monthly_2026-11-08')
    args = parser.parse_args()
    if args.data_dir:
        os.environ['DATA_DIR'] = str(args.data_dir.resolve())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protected = [ROOT / 'CURRENT_BEST_MODEL.json', *sorted((ROOT / 'saved_models').rglob('*')),
                 *sorted((ROOT / 'submissions').glob('*.csv'))]
    protected_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected if p.is_file()}
    index = json.loads((ROOT / 'CURRENT_BEST_MODEL.json').read_text())
    source_files = [Path(__file__), ROOT / 'data.py', *sorted((ROOT / 'experiments').rglob('*.py'))]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    raw_path = data.find_file('listings')
    raw = data.load('listings')
    if (raw.ml_num.duplicated().any() or not (raw.sold_price > 0).all()
            or not np.isfinite(raw.sold_price).all()
            or raw.date_sold.isna().any() or raw.date_listed.isna().any()
            or not (raw.date_sold < pd.Timestamp('2026-06-01')).all()):
        raise ValueError('Invalid observed data')
    write_json(out / 'protocol.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        months=list(map(str, MONTHS)), monthly_model_refit=True, within_month_pool_updates=False,
        use_pretrained_weights=False, source_sha256=source_hashes, input_sha256=sha(raw_path),
        protected_sha256=protected_hashes, current_best_snapshot=index,
        independent_test=False, selection_overlap_months=['2025-12'],
        versions={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'lightgbm', 'catboost']}))
    all_predictions = []
    for month in MONTHS:
        train, homes, cutoff, end = split_month(raw, month)
        folder = out / 'folds' / str(month)
        folder.mkdir(parents=True)
        train[['ml_num', 'date_sold']].to_csv(folder / 'train_ids.csv', index=False)
        audits, fold_predictions = {}, []
        for track, fit in [('off_market', fit_off_market), ('list_aware', fit_list_aware)]:
            log(f'{month} {track}: {len(train):,} earlier sales -> {len(homes):,} homes')
            values, audits[track] = fit(train, homes, cutoff, folder / track)
            rows = homes[['ml_num', 'house_type_name', 'community', 'sold_price']].copy()
            rows['track'], rows['month'], rows['predicted_price'] = track, str(month), np.round(values)
            rows['cutoff_exclusive'] = cutoff
            fold_predictions.append(rows)
            log(f'{month} {track}: MdAPE={score(rows)["mdape"]:.6f}%')
        frame = pd.concat(fold_predictions, ignore_index=True)
        frame.to_csv(folder / 'predictions.csv', index=False)
        write_json(folder / 'audit.json', dict(cutoff_exclusive=cutoff, evaluation_end_exclusive=end,
            train_n=len(train), predict_n=len(homes), max_train_sale=str(train.date_sold.max()),
            no_train_predict_id_overlap=True, query_answers_hidden=True, branches=audits,
            prediction_sha256=sha(folder / 'predictions.csv'), train_ids_sha256=sha(folder / 'train_ids.csv')))
        all_predictions.append(frame)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / 'validation_predictions.csv', index=False)
    scores = make_reports(out, raw, index)
    for name, digest in {**protected_hashes, **source_hashes}.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Protected file/source changed: ' + name)
    if sha(raw_path) != json.loads((out / 'protocol.json').read_text())['input_sha256']:
        raise ValueError('Data changed during validation')
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out / 'verification.json', dict(complete=True, folds=6, models_refit=42,
        saved_model_replay_exact=True, protected_files_unchanged=True, source_unchanged=True,
        exact_official_cohort=True, pooled_scoring=True, files_sha256=files, metrics=scores))
    log(f'COMPLETE: {json.dumps(scores)}')


if __name__ == '__main__':
    main()
