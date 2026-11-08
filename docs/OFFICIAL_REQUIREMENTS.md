# Official code-upload requirements

Rechecked online on November 7, 2026 (America/Toronto), against the official [handout](https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/handout.md#what-to-submit), [rubric](https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/rubric.md) and [README](https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/README.md#submission). Downloaded snapshots and hashes are in `docs/official/`.

## Explicit official requirements

- A GitHub repository, public or shared with the organizers, must hold the code and both predictions.
- Required paths: `submissions/submission_off_market.csv` and `submissions/submission_list_aware.csv`.
- Each CSV has columns `ml_num,predicted_price`, with one row for each home in its respective hidden test file. The datasets contain 3,974 and 4,064 homes respectively.
- The official workflow uses `python run.py predict` and `python run.py check`. Reproducibility is worth five points: judges must be able to rerun code and obtain predictions.
- Submit a PDF report of at most four pages via the organizer's form by **November 8, 2026, 11:59 pm EST**. The report links to the team's accessible GitHub repository and a YouTube video of at most five minutes. The video may be unlisted, but cannot be private. Use the official report template, include every team member's full name and report walk-forward validation MdAPE for both tracks. The template excludes the outside-data table and appendices from its four-page body limit.
- Validation must train on earlier sales and predict later sales. The starter uses December 2025–May 2026 monthly walk-forward validation. Off-market cannot use asking price. Do not use real listing/sale records or attempt to recover hidden labels. Outside data must cover May 2026 or earlier and have listed source URLs.

The official material does not require uploading raw datasets, all failed experiments, a virtual environment or temporary files. The file hash manifest, saved models, compact research source and setup instructions in this repository are packaging choices to support reproducibility.

## Release checks and remaining submission work

The code release uses the current research model index and includes both current CSVs. A separate clean Python 3.11 environment was used to rebuild raw features and rerun all seven saved branches. The resulting CSVs matched both indexed hashes byte for byte and passed official format checks. See `evidence/release_verification.json` for the recorded check results.

This does not complete the full competition submission. The selected specifications have now completed the official monthly schedule: Off-market MdAPE 7.62%, List-aware 3.93%, 10,069 homes per track. These remain cumulative development-validation scores; existing configuration selection overlapped December 2025. The final v19 report includes team hmf, member Frank hu, and the supplied YouTube link. Video metadata reports 4 minutes 31 seconds and playable status. Review the report and confirm video playback before submitting through the organizer’s form. Fixed four-month development scores must be labelled as such. Publishing the repository does not submit the organizer's form or establish a hidden-test score.
