# SSU Data Dash 2026: Toronto Home-Price Prediction Challenge

## The problem

When a home in Toronto goes on the market, everyone wants to know one thing: what will it actually sell
for? Buyers, sellers, banks and tax assessors all lean on automated valuation models (AVMs) to answer
that. Your team will build one, and test on sales it has never seen.

There are two different questions a pricing model can answer.

- **Off-market: what is this home worth?** You value a home before it is listed, so there is no asking
  price. This is the hard question, and it counts double.
- **List-aware: what will this listing close at?** The home is on the market and you know its list
  price. The list price already carries most of what the market knows, so this is much easier.

Pitfalls:

1. Mixing the questions up. A model that uses the list price looks far more accurate than one that
   does not. Reporting a list-aware result as an answer to the off-market question overstates what the
   model can do.
2. Time leakage. A model that learns from sales after the one it predicts looks excellent in
   testing and fails in use. A random train/test split does exactly this. Always train on the past and
   test on the future.

## Your task

Start from the working pipeline we give you and make it better: more accurate, more honest about its
uncertainty, or both. Strong submissions might:

- build better features: location, comparable recent sales, market trends, the listing descriptions;
- try other models or combine them;
- estimate uncertainty, not just a single price;
- find where the model fails (by price, property type, neighbourhood) and explain or fix it.

You will predict prices for homes that sold June to September 2026, and we score those predictions
against the real (held-back) sale prices after the deadline.

## The data

All data is synthetic: no real listing, address or price appears. It was generated from a model
calibrated to the Toronto market (property-type price levels, neighbourhoods with their own levels and
trends, the 2021 to 2026 boom and correction, sold-to-list ratios), so modelling moves that help on real
data help here too.

Download it from [huggingface.co/datasets/Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash)
(the starter code downloads it for you).

| File | Rows | What it is |
|---|---|---|
| `listings` | 150,000 | Sales from January 2021 to May 2026, with sold prices. Train on these. |
| `test_off_market` | 3,974 | Homes to value before listing: no list price. Predict their sale price. |
| `test_list_aware` | 4,064 | Listed homes with their list price. Predict their sale price. |

Each file comes as `.parquet` and `.csv`. The two test files are different homes.

| Group | Columns |
|---|---|
| Target | `sold_price` (listings only) |
| Listing | `list_price` (not in `test_off_market`), `date_listed`, `date_sold` and `days_on_market` (listings only) |
| Property | `house_type_name` (6 types), `house_style`, `bedrooms`, `washrooms`, `parking_total`, `garage`, `house_area` (sq ft, sometimes a range), `lot_front`, `lot_depth` (ft) |
| Location | `municipality`, `community` (neighbourhood), `latitude`, `longitude` |
| Text | `remarks`: a listing description, written by a language model |
| Third-party estimate | `external_avm_estimate`, `external_avm_estimate_date` (listings only) |
| ID | `ml_num` |

Two things to know:

- Location is synthetic. Neighbourhood names, coordinates and streets do not match real maps, so
  outside map or census data will not line up.
- The third-party estimate. It appears only in `listings`, not in the test files. A real estimate is
  only usable if it existed before the sale: check `external_avm_estimate_date` against the sale date
  before you rely on it, and think about whether your model could have it at prediction time.

## Your starting point

The starter repository, [github.com/erwinzhang7/SSU-Data-Dash](https://github.com/erwinzhang7/SSU-Data-Dash),
runs end to end. A Colab notebook and an R version are linked from its README.

```
pip install -r requirements.txt
python run.py validate   # walk-forward test on December 2025 to May 2026
python run.py report     # results table and a figure for your report
python run.py predict    # write both submission files
python run.py check      # check them before you hand in
```

Edit `features.py` (features) and `models.py` (models). `validate` predicts each month from the sales
before it, the same way we test you. Its numbers are the ones to report.

The bar. On the starter's walk-forward validation (December 2025 to May 2026):

| Track | Model | MdAPE | MAPE |
|---|---|---|---|
| Both | Maximum error: a simple baseline, the median sale of the same type in the same neighbourhood over the previous 12 months | 17.7% | 22.0% |
| Off-market | Starter baseline: median of the 10 nearest earlier sales | 20.8% | 27.4% |
| List-aware | Starter baseline: the list price | 7.3% | 8.1% |

MdAPE is the median absolute percentage error, |predicted − sold| / sold; MAPE is the mean. To earn
accuracy points on a track, your hidden-test MdAPE must be below the maximum error: the simple
baseline's error on the same hidden homes. The starter's off-market baseline is above it, so you have to
improve on it. A gradient-boosted model (`FeatureModel`) is ready to switch on in `models.py`.

## What to submit

By **November 8, 11:59 pm EST**, through the form we will email and post on Discord: one PDF report
(4 pages max) that links to:

1. A video, 5 minutes max, on YouTube (unlisted is fine, private is not). Judges stop at 5:00. Format and what to
   cover: see [The video](#the-video) below.
2. A GitHub repository (public, or shared with the organisers) holding your code and two files,
   written by `python run.py predict`:

```
submissions/submission_off_market.csv    one row per home in test_off_market
submissions/submission_list_aware.csv    one row per home in test_list_aware

ml_num,predicted_price
SYN-0123456,815000
```

Use the report template in the starter (`report/report.tex`; it compiles on Overleaf). Put every
team member's full name on it. Report your walk-forward validation MdAPE for each track: we compare
it with your hidden-test error. Late submissions are not reviewed.

### The video

Any format works: slides with a voice-over, a screen recording of your notebook or code, or you on camera.
Slides are fine and are the most common choice. Please don't read your report aloud. The video is where you
show you understand what you built, so talk it through in your own words.

In five minutes, cover:

1. **What you built:** your features and model, in plain words.
2. **Why:** the choices that mattered, and what you tried that didn't help.
3. **How you validated it:** your walk-forward split and your validation MdAPE for each track.
4. **Where it fails:** which homes, prices or neighbourhoods it gets wrong, and why you think so.

Judges stop at 5:00, so anything after that isn't seen.

## How you are judged

The full rubric is in the starter repository. In short:

**Round 1, online review (November 9 to 15).**

| Criterion | Points | What earns it |
|---|---|---|
| Accuracy on the hidden test | 30 | Off-market 20, list-aware 10. Ranked by MdAPE; MAPE breaks ties. |
| Method and validation | 25 | Train on the past, test on the future. Nothing from after the sale or from the test set. |
| Insight and error analysis | 20 | Where the model fails (price, type, neighbourhood) and why. Honest uncertainty. |
| Honesty about performance | 10 | Your validation error should predict your hidden-test error. |
| Communication | 10 | A clear five-minute video on what you built and why. Slides with a voice-over are fine; don't just read your report. See [The video](#the-video). |
| Reproducibility | 5 | We can rerun your code and get your predictions. |

A smaller error that you cannot explain loses to a slightly larger one with a clean, honest method.

**Round 2, finale (in person, November 20).** The top five teams present. Scores start fresh:

| Criterion | Points |
|---|---|
| The problem and who it serves: a real user, a real decision | 25 |
| Recommendations that follow from your results | 25 |
| Risks and limits: when not to trust the model | 20 |
| Technical credibility under questions | 15 |
| Presentation and Q&A | 15 |

## Timeline

| Date | What |
|---|---|
| October 8 | Case and data released. Workshop, 1 to 3 pm, room 9199, 700 University Ave. |
| November 8, 11:59 pm EST | Submissions due. |
| November 9 to 15 | Round 1 review. |
| November 16 | Finalists announced. |
| November 20 | Finale, in person, followed by a networking event. Everyone who submits is invited. |

Prizes, per team: $900 first, $600 second, $300 third.

## Rules

- Outside data only if it covers May 2026 or earlier. List every source with its URL in your report.
- No real listing or sales records. No attempts to recover the hidden test prices. Either
  disqualifies the team.
- Every number in your video comes from your own validation.
- The off-market model may not use the list price.

## Getting help

Join the Discord: [discord.gg/TE2UPPj6G](https://discord.gg/TE2UPPj6G). Post in the forum that fits:

- case-questions: the problem, the data, the rules, judging;
- code-help: the starter code, setup, errors;
- workshop-questions: anything from the workshop.

Search before you post: someone may already have asked. Answers posted there apply to every team.

## Common mistakes

- A random split. It lets the model see the future. Use `run.py validate` or the same walk-forward
  idea.
- List price in the off-market model. The off-market test file has none.
- Features built from the whole dataset, such as a neighbourhood average that includes later sales.
  Compute them from earlier sales only.
- The third-party estimate without checking its date.
- Reporting the best of many tries on the validation months as if it were a fresh test. Your
  hidden-test error will be worse, and the honesty score will show it.
