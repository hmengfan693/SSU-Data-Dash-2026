# Suggested report wording

Using the official December 2025–May 2026 monthly walk-forward schedule, our selected Off-market property-type boosted trees and List-aware global LightGBM ratio model achieved MdAPE of 7.62% and 3.93%, respectively (MAPE: 10.19% and 5.04%), across 10,069 homes per track. Each month, model weights and category vocabularies were refitted using only sales before that month, with the feature definitions, parameters and routing held fixed. These are cumulative development-validation results: existing configuration selection overlapped December 2025, and hidden-test performance remains unknown.

# Official-schedule monthly walk-forward validation

The selected frozen models were refitted separately for December 2025 through May 2026. For each month, training uses every observed sale strictly before the first day of that month; prediction covers that month only. Parameters, feature definitions and routing remain fixed. The reference pool is refreshed between months and frozen within each month.

| Track | Homes | MdAPE | MAPE | MdAPE 95% bootstrap interval |
| --- | ---: | ---: | ---: | --- |
| off-market | 10,069 | 7.62% | 10.19% | 7.45%–7.81% |
| list-aware | 10,069 | 3.93% | 5.04% | 3.83%–4.04% |

Overall MdAPE is the median of all individual-home percentage errors pooled across the six months, not an average of monthly or property-type medians. Dollar predictions are rounded before scoring, matching the official starter. Bootstrap settings also match the starter; homes are resampled independently, so these intervals do not capture common monthly market shocks.

## Monthly results

| Month | Homes per track | Off-market MdAPE | List-aware MdAPE |
| --- | ---: | ---: | ---: |
| 2025-12 | 1,238 | 7.95% | 3.85% |
| 2026-01 | 1,058 | 7.74% | 3.75% |
| 2026-02 | 1,442 | 7.80% | 3.92% |
| 2026-03 | 1,816 | 7.07% | 3.97% |
| 2026-04 | 2,224 | 7.56% | 3.94% |
| 2026-05 | 2,291 | 7.80% | 4.07% |

## Relationship to the indexed current best

The models are the indexed complete Off-market property-type boosted-tree specification and the indexed complete List-aware global LightGBM log sold/list ratio specification. No new model is selected in this run. Reusing the indexed fixed-four-month protocol would therefore give the same indexed scores (difference: 0 percentage points). The official-schedule run changes training cutoff, reference updating and forecast horizon. It cannot be ranked against the fixed-four-month archive as if the protocols were identical.

| Track | Feb–May homes | Indexed fixed-four-month MdAPE | Monthly refit MdAPE on same homes | Difference (pp) |
| --- | ---: | ---: | ---: | ---: |
| off-market | 7,773 | 7.9496% | 7.5663% | -0.3833 |
| list-aware | 7,773 | 3.9341% | 3.9873% | +0.0532 |

## Training, lookbacks and leakage boundary

The expanding model-training span begins with the observed 2021 sales and ends before each monthly cutoff. Historical feature lookbacks are separate: price and sold/list summaries use 90, 180 and 365 days; support and land context use their existing frozen definitions. Historical price queries end strictly before min(listing date, monthly cutoff). Land context uses a completed prior month and a 365-day lookback. Each forecast horizon is one calendar month. A one-year historical feature does not mean the model is trained on only one year.

Each tree splits homes using feature thresholds or category groups and assigns a small correction at a leaf. Boosting adds many such corrections. Off-market learns log sale price; List-aware learns log(sale price / list price), then multiplies the predicted ratio by the known list price. Both minimize absolute error in the log target. This limits the influence of expensive outliers and emphasizes multiplicative errors, but is not an exact optimization of MdAPE. Expanding training keeps scarce property-type examples, while short historical features reflect recent prices.

No current-month or future sale labels enter that fold’s model fit, category vocabulary or reference pool. No validation early stopping is used. Sold price, sale date, days on market and third-party valuation columns are removed from query inputs; Off-market also removes list price. The monthly model can have seen sales after a particular home was listed but before the monthly cutoff, as in the official sale-month protocol. This is not a claim of retraining at each listing date.

**Development-use limitation:** the existing configuration-selection window includes September–December 2025 and overlaps the first official validation month. Earlier routing and repeated project decisions also used development results. These scores are a retrospective, cumulative development validation under the official split schedule, not a pristine nested or independent test. Fold-local label isolation does not remove model-selection reuse. Parameters were held fixed during this run; no claim is made that the December configuration was selected without December outcomes.

The June–September 2026 hidden-test prices remain unavailable, so hidden-test MdAPE and competition rank/points cannot be calculated locally. Model feature relationships are predictive, not causal; synthetic data and shared monthly market conditions limit generalization.

## Reproduction and evidence

Run `python run.py validate --data-dir /path/to/data --output /new/output/folder`. The standalone equivalent is `python validate_official.py --data-dir /path/to/data --output /new/output/folder`. Every fold saves its newly trained models, training IDs, audit and predictions. `official_metrics.json` follows official rounding/bootstrap; `metrics_precise.json` retains full precision. `group_metrics.csv` and `comparison_fixed_vs_monthly.csv` support diagnostics. Final submission CSVs and original models are not replaced.

Official sources: [handout](https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/handout.md), [validation/scoring implementation](https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/run.py).

## Frozen specification

| Branch | Algorithm | Trees | Structure | Learning rate | Training population |
| --- | --- | ---: | --- | ---: | --- |
| List-aware | LightGBM | 680 | 21 leaves | 0.042084 | All six property types |
| Co-Op Apt | CatBoost | 600 | depth 6 | 0.050000 | Condo Apt, Co-Op Apt |
| Condo Apt | CatBoost | 600 | depth 6 | 0.050000 | Condo Apt, Co-Op Apt |
| Condo Townhouse | LightGBM | 359 | 15 leaves | 0.055790 | Condo Townhouse |
| Detached | LightGBM | 830 | 11 leaves | 0.082064 | Detached, Freehold Townhouse, Semi-Detached |
| Freehold Townhouse | LightGBM | 386 | 18 leaves | 0.144908 | Detached, Freehold Townhouse, Semi-Detached |
| Semi-Detached | LightGBM | 867 | 59 leaves | 0.053691 | Freehold Townhouse, Semi-Detached |

Learning rates above are rounded for readability; exact parameters and feature lists are in `frozen_model_specification.json` and each fold’s schema. Both libraries use L1 loss on the stated log target. MdAPE is median(100 × |predicted − sold| / sold); MAPE is the mean of those errors.
