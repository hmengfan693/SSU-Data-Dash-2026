# Selected method and limitations

We predict synthetic Toronto home sale prices in two tracks. Off-market has no asking price. List-aware knows the asking price and learns an adjustment. No outside prediction data or real listing records are used.

## How the trees work

A tree asks a sequence of questions about a home, such as its property type, area or recent local prices. Each answer leads to another branch until a leaf supplies a numerical contribution. Boosting builds trees in sequence: later trees learn corrections to the ensemble's earlier errors. Prediction applies the fixed trees; it does not retrain for each home.

Off-market models learn log sale price and transform their combined output back with an exponential. List-aware learns `log(sold_price / list_price)` and multiplies the asking price by the exponential correction. Predictions are rounded to whole dollars. A log target treats multiplicative price differences more evenly across cheap and expensive homes. L1 loss reduces the influence of unusually large transformed errors. This is a practical surrogate; L1 training does not directly optimize the median percentage error used for competition ranking.

## Training and routing

The Off-market route is frozen by property type. Condo Apt uses the inherited base CatBoost apartment model; Co-Op Apt uses the inherited CatBoost branch with its original text enrichment. Detached and Semi-Detached use enhanced history, land and text features in LightGBM. Freehold Townhouse retains its original feature representation; Condo Townhouse removes lot features. Pooling related property types supports sparse groups while routing allows different market structures. Actual training code and model schemas document the pooling and loss configuration; no routing or parameter changes were made for this release.

LightGBM branches use `regression_l1` on the transformed target. The inherited CatBoost loss is recorded in each saved schema. The exact number of trees, learning rates, leaf constraints, regularization, feature order and training category vocabularies are frozen in `saved_models/**/schema.json`. List-aware uses one global 680-tree LightGBM model with 21 leaves and learning rate 0.04208449292893393. The release loads these trained models rather than repeating model selection.

## Three different time spans

1. **History lookback:** input summaries use prior 90, 180 and 365-day windows (and 365 days for coordinate-type groups). These are feature windows.
2. **Training sample span:** models learn from eligible earlier sales across multiple years. The official data begin in January 2021. Training stops before February 1 for the development forecast and before June 1 for the final forecast. A history window does not restrict training to that many days.
3. **Forecast horizon:** the development forecast covers February–May 2026 and the final forecast covers June–September, four months each. The reference pool receives no true sale-label updates during those forecast blocks.

Sale dates determine label eligibility and validation membership; they are not prediction features. Historical features query earlier eligible sales and exclude the home's own label and same-day labels. After-sale fields and external AVM columns are hidden. Off-market also hides asking price. Test sale labels are unavailable and were not recovered or used.

## Interpretation and limits

Past market summaries, property attributes and listing-text indicators provide explainable inputs. Tree contributions describe a model's numerical adjustment, not a causal effect. Text indicators and lot-area proxies can be wrong. Locations are synthetic and cannot be joined to real maps.

Both selected-model scores are fixed four-month development backtests on the same 7,773 homes. Repeated development on these periods creates selection optimism. Hidden-test errors remain unknown. Final-model monthly walk-forward validation is outstanding, and no calibrated uncertainty interval for an individual home is claimed. Expensive homes, sparse groups and future changes require particular care. See the archived slice and failure analyses in `evidence/report/`.
