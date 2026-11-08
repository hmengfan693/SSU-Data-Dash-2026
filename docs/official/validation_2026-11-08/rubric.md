# SSU Data Dash: how you are judged

## Submit by November 8 11:59 pm EST

An email will be sent out with a form, as well as posted on the Discord.

One PDF report (4 pages max, template in the starter repo) linking to:
1. **a video**, 5 minutes max, on YouTube (unlisted is fine). Judges stop at 5:00. Format and what to cover:
   see "The video" in [handout.md](handout.md#the-video).
2. **a GitHub repository** with your code and `submission_off_market.csv` and
   `submission_list_aware.csv` (`python run.py predict` writes them, `python run.py check` checks them).

## Round 1: online review (November 9-15)

| Criterion | Points | What earns it |
|---|---|---|
| Accuracy on the hidden test | 30 | Off-market 20, list-aware 10. Median absolute percentage error (MdAPE); mean (MAPE) breaks ties. |
| Method and validation | 25 | Train on the past, test on the future. Nothing from after the sale or from the test set. |
| Insight and error analysis | 20 | Where the model fails (price, type, neighbourhood) and why. Honest uncertainty. |
| Honesty about performance | 10 | Your validation error should predict your hidden-test error. |
| Communication | 10 | A clear five-minute video on what you built and why. Slides with a voice-over are fine; don't just read your report. See "The video" in the handout. |
| Reproducibility | 5 | We can rerun your code and get your predictions. |

Off-market counts double: it is the harder question.

## Round 2: finale (in person, November 20)
Top five teams present. Scores start fresh.

| Criterion | Points |
|---|---|
| The problem and who it serves: a real user, a real decision | 25 |
| Recommendations that follow from your results | 25 |
| Risks and limits: when not to trust the model | 20 |
| Technical credibility under questions | 15 |
| Presentation and Q&A | 15 |

## Rules
- Outside data only if it covers May 2026 or earlier. List every source with its URL.
- No real listing or sales records. No attempts to recover hidden test prices. Either disqualifies.
- Every number in your video comes from your own validation.
- Late submissions will not be reviewed.
