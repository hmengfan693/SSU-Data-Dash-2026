# Toronto Home-Price Prediction

English slides and speaker notes | Final selected models | November 8, 2026

## Presentation plan

Seven main slides, with a target delivery of **4 minutes 40 seconds**, leaving 20 seconds before the competition's five-minute limit. Backup slides are for questions and are excluded from the main presentation PDF. Timing is a rehearsal target, not a measured recording.

Content follows the latest report, `output/reports/ssu_report_v17/SSU_Data_Dash_2026_Report_v17.md`, and the current model index, `SSU-Data-Dash/CURRENT_BEST_MODEL.json`, checked for this presentation. The November 4 compact submission package is an older archive. The current team release follows the model index. No model, prediction file or submission is replaced by this presentation.

Design: 16:9 Beamer; white background, dark navy type, blue Off-market and teal List-aware accents; large text, minimal prose, vector results figure. Unknown team and presenter names are omitted.

## Slide 1 — Toronto Home-Price Prediction

**Target: 0:00-0:25**

### On screen

SSU Data Dash 2026

**Off-market:** What is this home worth before listing? No asking price.

**List-aware:** What will this listing close at? Asking price available.

150,000 synthetic Toronto sales · January 2021–May 2026 · No outside prediction data

### Speaker notes

How much should a Toronto home sell for? Before listing, our Off-market model has no asking price. Once listed, our List-aware model can use the seller's advertised price. These support different decisions, so we evaluate them separately, using one hundred and fifty thousand synthetic sales and no outside prediction data.

### Visual description

A large project title above two clearly separated track descriptions. Blue identifies Off-market and teal identifies List-aware throughout the deck.

## Slide 2 - Official monthly and team fixed-four-month validation

**Target: 0:25-1:20**

### On screen

| Purpose | Training cutoff | Evaluation / forecast | Homes per track |
| --- | --- | --- | ---: |
| Official monthly | Before each month | Dec 2025-May 2026, monthly refits | 10,069 |
| Team fixed four-month | Before Feb 1, 2026 | Feb-May 2026, no updates | 7,773 |
| Final refit | Before Jun 1, 2026 | Jun-Sep 2026, hidden outcomes | 3,974 / 4,064 |

Historical feature lookbacks: 90 / 180 / 365 days. Training sample: several years. Prediction horizon: one month per official fold or four months per frozen batch.

No current-fold outcomes enter fitting or reference pools. Existing configuration selection overlapped December 2025. Both evaluations are cumulative development validation.

### Speaker notes

We use two validation designs. The official design predicts December twenty twenty-five through May twenty twenty-six, refitting before each month. It evaluates ten thousand and sixty-nine homes per track. Our fixed four-month design trains before February and predicts February through May without updates, covering seven thousand seven hundred and seventy-three homes. This mirrors the final four-month forecast. Both designs train on several years of data. The ninety-, one-hundred-eighty-, and three-hundred-sixty-five-day windows are feature lookbacks, rather than training lengths. Each historical query excludes future, same-day and self-sale outcomes.

### Technical boundary

For cutoff C, historical query time is min(listing date, C). Price queries exclude future, same-day and self-sale outcomes. Monthly folds update their reference pool only between months. Within the fixed-four-month batch, the model and pool remain frozen. Categories come only from the relevant training fold. Off-market excludes asking price. No evaluation-label early stopping is used. These fold-local restrictions do not remove past configuration-selection reuse.

## Slide 3 — LightGBM: efficient gradient-boosted trees

**Target: 1:20-2:00**

### On screen

LightGBM is a sequence of decision trees for tabular data.

1. **Boosting:** each new tree learns a loss-based correction to the current score.
2. **Leaf-wise growth:** split the leaf with the largest expected loss reduction.
3. **Histogram-based splits:** group numeric values into bins to reduce split-search cost.

**Why it fits our task:** nonlinear relationships between location, size and type; numeric and categorical inputs; controlled complexity through leaf limits, minimum samples and penalties.

Updated score = current score + learning rate × new tree correction.

Prediction combines the frozen trees; no per-home retraining. Flexible leaf-wise growth can overfit, so constraints matter. Task suitability is our modelling rationale, not a claim that LightGBM was proven best.

### Speaker notes

LightGBM combines decision trees. A tree follows rules about a home's features and assigns a score at a leaf. Later trees add corrections guided by the loss. LightGBM grows the leaf with the largest expected improvement and groups numeric values into bins to search efficiently. That suits nonlinear interactions between location, size and property type. We constrain tree size and apply penalties to limit complexity. Prediction combines the frozen trees.

### Visual description

Three numbered mechanisms occupy the left half. The right half connects them to the project's tabular features and complexity controls. A large additive update equation links the current score, learning rate and new tree correction.

### Method source

[LightGBM official feature guide](https://lightgbm.readthedocs.io/en/stable/Features.html), sections on histogram-based split search and leaf-wise tree growth. This is algorithm background, not outside prediction data. The loss guides the corrections; with L1, they should not be described as simply fitting raw residual magnitudes.

## Slide 4 — From tree scores to home prices

**Target: 2:00-2:40**

### On screen

**Off-market:** property-type routing; LightGBM for houses and townhouses, CatBoost for apartments. Target log(P); predicted price exp(f(X)), rounded. Related types share training examples where specified, while routes allow different price relationships.

**List-aware:** one global LightGBM across all six types. Target log(P/L); predicted price L × exp(f(X)), rounded. The global model shares information while permitting type interactions.

Both use absolute error in log space: (1/N) Σ |f(X_i) − t_i|. A relative-error surrogate; less sensitive to extreme residuals than squared loss.

### Speaker notes

Off-market predicts log sale price using property-type routes, with related types sharing training examples. Houses and townhouses use LightGBM, while apartments use CatBoost. List-aware uses one global LightGBM to predict the log sale-to-asking-price ratio. We multiply that adjustment by the asking price. Both models minimize absolute error in log space. This emphasizes relative differences and reduces the influence of extreme errors compared with squared loss. It approximates our goal, without directly optimizing median percentage error.

### Visual description

Training-population choices on the left explain pooling and routing. Two large target-and-price equations on the right explain the prediction paths. The shared L1 loss appears below, with the limitation that no loss ablation establishes optimality.

## Slide 5 - Results under both validation protocols

**Target: 2:40-3:40**

### On screen

| Track / model (validation protocol) | Homes | MdAPE | MAPE |
| --- | ---: | ---: | ---: |
| Off-market trees (official monthly) | 10,069 | 7.62% | 10.19% |
| List-aware asking price (official monthly) | 10,069 | 7.26% | 8.09% |
| List-aware ratio (official monthly) | 10,069 | 3.93% | 5.04% |
| Off-market trees (team fixed four-month) | 7,773 | 7.95% | 10.38% |
| List-aware asking price (team fixed four-month) | 7,773 | 7.20% | 8.03% |
| List-aware ratio (team fixed four-month) | 7,773 | 3.93% | 5.19% |

The asking-price baseline directly uses the listed price as the prediction. Compare models within one protocol. Samples and updating differ across protocols. Hidden-test accuracy is unknown.

### Speaker notes

Under official monthly validation, Off-market achieves seven point six two percent median absolute percentage error. For List-aware, simply predicting the asking price gives seven point two six percent. Our ratio model reduces that to three point nine three percent. Under our fixed four-month design, Off-market gives seven point nine five percent, and List-aware gives three point nine three percent. Mean errors also appear in the table. These are the same selected model specifications. Different homes, cutoffs and updating rules prevent a direct ranking across protocols. Configuration selection overlapped December, so both results remain cumulative development validation. Hidden-test performance is unknown.

### Evidence

Monthly metrics and predictions are in `evidence/official_monthly/`. The indexed fixed-four-month evidence, including preceding complete-model comparisons, remains in backup B. Both evaluate the same selected specifications. No new model is selected, and different validation protocols are not ranked as a model improvement.

## Slide 6 - High-price homes remain difficult (fixed-four-month validation)

**Target: 3:40-4:15**

### On screen

Embed the original vector figure `result-price-deciles.pdf` from the November 7 figure previews, unchanged.

Fixed-four-month February-May cohort, 7,773 homes. Highest-price decile: **16.67% Off-market MdAPE**, versus **6.42%** in the lowest decile. List-aware: **5.04%**, versus **7.38%** for asking price in the highest decile.

**Use extra review for expensive or unusual homes.**

### Speaker notes

This figure uses our fixed four-month validation. Off-market error rises to sixteen point six seven percent in the highest-price decile, compared with six point four two in the lowest. Missing quality information or shrinkage toward typical homes could contribute, but this chart cannot establish the cause. Realized-price groups diagnose past errors and cannot directly route future predictions. Expensive or unusual homes deserve extra review.

### Figure description and uncertainty

Two panels plot MdAPE against realized sale-price decile. Solid blue circles show the current complete model; dashed grey squares show the preceding complete model. The List-aware panel also shows the asking-price baseline as dotted orange diamonds. All use identical homes within the cohort; deciles have 775–780 homes. Shading is pointwise 95% within-decile row-bootstrap intervals from 2,000 resamples. Intervals omit temporal dependence, repeated selection and future regime changes. Original chart title and legend are retained. Numeric data are copied into the evidence folder.

## Slide 7 — Use the estimate with its limits

**Target: 4:15-4:40**

### On screen

**Before listing:** Off-market supports initial valuation.

**After listing:** List-aware adjusts the asking-price anchor.

**Extra review:** expensive, unusual or weakly supported homes.

Next: unused-period testing and past-only uncertainty intervals.

Official monthly MdAPE: 7.62% / 3.93% · Hidden-test accuracy unknown

### Speaker notes

Off-market supports initial valuation before listing. List-aware adjusts the asking-price anchor once a home is listed. We recommend extra review for unusual homes. Next, we would test an unused period and calibrate uncertainty intervals from past data. Model explanations describe fitted behavior, while hidden-test accuracy remains unknown. Thank you.

## Backup A — Frozen models and fitting

Outside the timed talk. The current Off-market pipeline has six routes:

| Prediction type | Model | Training population |
| --- | --- | --- |
| Condo Apt | CatBoost | Shared apartment pool |
| Co-Op Apt | CatBoost | Shared apartment pool |
| Condo Townhouse | LightGBM | Own type |
| Detached | LightGBM | All freehold types |
| Semi-Detached | LightGBM | Semi-Detached + Freehold Townhouse |
| Freehold Townhouse | LightGBM | All freehold types |

Related populations add examples; routes allow different property-price relationships. One global List-aware LightGBM uses all six types and 150 inputs, sharing information while allowing interactions. Its frozen configuration has 680 trees, 21 maximum leaves and learning rate 0.04208. No evaluation-label early stopping is used.

| LightGBM route | Trees | Max leaves | Learning rate |
| --- | --- | --- | --- |
| Detached | 830 | 11 | 0.08206 |
| Semi-Detached | 867 | 59 | 0.05369 |
| Freehold Townhouse | 386 | 18 | 0.14491 |
| Condo Townhouse | 359 | 15 | 0.05579 |
| List-aware global | 680 | 21 | 0.04208 |

Exact unrounded configuration remains in saved schemas. Feature definitions, routes and settings are held fixed for the final through-May refit. L1 in log space is a surrogate; no loss ablation proves it optimal.

## Backup B — Complete-model comparisons

All rows below use identical IDs and the same frozen four-month forecast. Gap = row MdAPE minus current complete model MdAPE in that track, in percentage points.

| Track / complete model | MdAPE | MAPE | Gap to current |
| --- | --- | --- | --- |
| Off-market: previous enhanced | 7.999% | 10.401% | +0.050 pp |
| Off-market: current routed trees | 7.950% | 10.379% | 0.000 pp |
| List-aware: asking price | 7.195% | 8.032% | +3.261 pp |
| List-aware: previous ratio | 3.939% | 5.185% | +0.005 pp |
| List-aware: current global ratio | 3.934% | 5.190% | 0.000 pp |

Paired 95% intervals for current-minus-reference MdAPE: Off-market versus preceding pipeline [-0.17, +0.07] pp; List-aware versus preceding ratio [-0.05, +0.06] pp; List-aware versus asking price [-3.44, -3.07] pp. They use 2,000 paired row-bootstrap resamples and omit temporal dependence and model-selection bias.

## Backup C — Explaining one prediction

ID-selected Detached example `SYN-0000026`, chosen without examining errors. Asking price $1,688,200. January-cutoff evaluation models, distinct from final through-May models.

| Component | Off-market log price | List-aware log ratio |
| --- | --- | --- |
| Expected-value term | +14.1492 | -0.0077 |
| Sum of feature contributions | -0.1627 | -0.0791 |
| Reconstructed price | $1,186,505 | $1,547,898 |

Full-precision additive contributions exactly reconstruct the archived predictions. Display values are rounded. Asking price multiplies List-aware output outside the trees; tree gain alone does not capture its complete role. Correlated inputs share credit. These explanations describe fitted behaviour, not causal effects or guaranteed accuracy. No outcome or example error is used to select this case.

## Backup D - Sources and reproducibility

The v17 report, current model index, six-fold official monthly evidence, fixed-four-month complete-model comparisons, local contribution evidence, and the unchanged November 7 price-decile figure support this deck.

- Official validation requirement: https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/handout.md
- Official split/scoring implementation: https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/run.py
- Team code: https://github.com/hmengfan693/SSU-Data-Dash-2026
- Method: https://lightgbm.readthedocs.io/en/stable/Features.html

The official monthly run completed six folds and 42 fitted models. All February predictions exactly match the indexed current-best model at the same training cutoff. Final submission hashes remain unchanged. Existing configuration selection overlapped December 2025. Hidden-test error and competition points remain unknown.
