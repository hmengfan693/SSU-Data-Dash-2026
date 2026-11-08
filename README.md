# SSU Data Dash 2026 — selected prediction pipeline

Team code repository: https://github.com/hmengfan693/SSU-Data-Dash-2026

This repository contains the current selected property-type tree models for Off-market valuation and a global LightGBM asking-price adjustment for List-aware valuation. The required CSVs are in [`submissions/`](submissions/). Their hashes match the research project's current model index, copied unchanged to [`CURRENT_BEST_MODEL.json`](CURRENT_BEST_MODEL.json).

## Reproduce both submissions

Use Python 3.11. On macOS, LightGBM also needs the OpenMP runtime (`brew install libomp`).

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_package.py
python run.py predict
python run.py check
```

On Windows, activate with `.venv\Scripts\activate`. The official synthetic data download automatically from [Hugging Face](https://huggingface.co/datasets/Baliyang/SSU_data_dash). To use downloaded data and write to a separate folder:

```sh
python run.py predict --data-dir /path/to/data --output reproduced_submission
python run.py check --data-dir /path/to/data --output reproduced_submission
```

The data directory must contain `listings`, `test_off_market` and `test_list_aware`, each as Parquet or CSV. Raw datasets are not redistributed here. [`DATA_MANIFEST.json`](DATA_MANIFEST.json) records the data files used for the release verification. The Hugging Face download follows the dataset's current default revision; exact replay is verified against the selected prediction hashes and fails if the result changes.

`predict` reconstructs features from raw data and runs all seven saved model branches, including both apartment models. It does not read the submitted CSVs as model inputs. It writes results only after both forecasts match their indexed SHA-256 hashes and pass the official checker. Default output is `submissions/`.

## Results and validation scope

The same selected Off-market property-type boosted trees and List-aware global LightGBM ratio specification are evaluated under both protocols. Each Model entry identifies its validation protocol.

| Track | Model (validation protocol) | Homes | MdAPE | MAPE |
| --- | --- | ---: | ---: | ---: |
| Off-market | Property-type trees (official monthly) | 10,069 | 7.62% | 10.19% |
| List-aware | Asking-price baseline (official monthly) | 10,069 | 7.26% | 8.09% |
| List-aware | Global ratio model (official monthly) | 10,069 | 3.93% | 5.04% |
| Off-market | Property-type trees (team fixed four-month) | 7,773 | 7.95% | 10.38% |
| List-aware | Asking-price baseline (team fixed four-month) | 7,773 | 7.20% | 8.03% |
| List-aware | Global ratio model (team fixed four-month) | 7,773 | 3.93% | 5.19% |

**Official monthly validation:** December 2025–May 2026, refitting on eligible past sales before each month and refreshing history between months. **Team fixed-four-month validation:** train before February 1 and forecast February–May without updates. The asking-price baseline directly predicts the listed price. Compare models within one protocol; different samples, cutoffs and updating prevent a direct ranking across the two overall results.

The original indexed fixed-four-month MdAPE remains 7.949597% Off-market and 3.934101% List-aware, with a 0 pp difference from the indexed current best because no new model was selected. The final saved models use labels before June 1, 2026 for the June–September forecast. Both evaluations are cumulative development validation; hidden-test error is unknown.

The official starter evaluates December–May with monthly walk-forward updating. The selected frozen specifications can be refitted on that schedule with:

```sh
python run.py validate --data-dir /path/to/data --output /new/validation/folder
python verify_official_validation.py --data-dir /path/to/data --output /new/validation/folder
python -m unittest discover -s tests -v
```

The output folder must be new. Validation trains seven fresh models per month, rebuilding historical features from only earlier sales and learning fresh training-only category vocabularies. It writes the 42 fitted models, training IDs, per-fold audits, individual-home predictions, precise and officially rounded scores, and an English Markdown report. The independent verifier checks all fold IDs/cutoffs, frozen parameters, artifact hashes and pooled metrics; February predictions must exactly reproduce the indexed development models with the same training cutoff. Tests compare all scoring/bootstrap fields against the downloaded official implementation in `docs/official/validation_2026-11-08/`.

These are cumulative development results: the existing configuration-selection window includes December 2025, overlapping the first official fold. Fold-local training and reference features exclude current/future sale labels, but this does not remove configuration-selection reuse. A fixed four-month forecast, or its monthly slices, must not be presented as monthly refitting. Archived starter monthly scores in `evidence/starter_monthly_validation.json` describe the starter models.

### Report, slides and preliminary speaking script

- [Report PDF](report/SSU_Data_Dash_2026_Report_v18.pdf) and [English Markdown](report/SSU_Data_Dash_2026_Report_v18.md)
- [Seven-slide presentation](presentation/SSU_Data_Dash_5min_Both_Protocols_2026-11-08.pdf) and [presentation with four backups](presentation/SSU_Data_Dash_with_Backup.pdf)
- [Preliminary English speaking script](presentation/Initial_Speaking_Script_EN.md) and [slide-by-slide notes](presentation/Slides_and_Speaker_Notes.md)
- [Editable Beamer source package](presentation/SSU_Data_Dash_LaTeX_Package.zip)

The report and slides show both protocols. The price-decile figure remains the fixed-four-month diagnostic, clearly labelled. The script targets 4 minutes 40 seconds; that is a rehearsal estimate, not a measured recording. Team hmf and member Frank hu are named in the report. [YouTube video](https://youtu.be/_39kVEx1bFA): 4 minutes 31 seconds according to video metadata. See [compact monthly evidence](evidence/official_monthly_summary/README.md) for completed run scope.

To independently recompute the indexed development scores from archived predictions and the official observed training labels:

```sh
python score_archive.py --data-dir /path/to/data
```

This command checks IDs, forecast hashes and scores; it does not fit models or recreate walk-forward validation.

## Repository contents

| Location | Purpose |
| --- | --- |
| `run.py`, `data.py`, `check_submission.py` | Supported final-prediction entry point, official data loading and official CSV checking |
| `validate_official.py`, `verify_official_validation.py`, `tests/` | Monthly refitting of selected frozen specifications, independent audits and boundary/scoring tests |
| `experiments/` | Feature builders and saved-model loaders required at prediction time |
| `saved_models/` | Six Off-market property-type branches and one List-aware model, with frozen schemas and parameters |
| `submissions/` | Off-market: 3,974 rows; List-aware: 4,064 rows; columns `ml_num,predicted_price` |
| `evidence/` | Current selected development predictions, comparisons, configurations and historical verification records |
| `research_source/` | Historical training and experiment code for inspection; see its [scope note](research_source/README.md) |
| `docs/` | Method explanation and official requirement snapshots |
| `report/`, `presentation/` | Latest report draft, slides, editable source and preliminary speaking script |
| `PACKAGE_MANIFEST.json` | Actual portable file hashes for this release |

See [method and limitations](docs/METHOD.md) and [official upload requirements](docs/OFFICIAL_REQUIREMENTS.md). Historical research IDs and tool names are retained where needed for traceability. `evidence/*/protocol_portable.json` replaces the original local workspace prefix with `ORIGINAL_WORKSPACE`; original protocols and experiments remain preserved locally. Historical verification files are records of earlier checks, not new checks of this reorganized directory. The release checks are in `evidence/release_verification.json`.

## Submission status

Publishing this repository is one part of the competition submission. The v18 report names team hmf and member Frank hu and links to the team repository and [YouTube video](https://youtu.be/_39kVEx1bFA). Video metadata reports 4 minutes 31 seconds and a playable status. The report and presentation identify both validation protocols. Review the complete report and confirm video playback before submitting. Publishing these files does not submit the organizer’s form or establish a hidden-test score.

Starter code is credited to [Erzheng Zhang's official repository](https://github.com/erwinzhang7/SSU-Data-Dash); its MIT license is preserved in [`LICENSE`](LICENSE).
