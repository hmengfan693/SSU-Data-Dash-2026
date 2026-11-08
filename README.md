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

| Track | Current selected complete model | Development MdAPE | Development MAPE | Difference from indexed current best |
| --- | --- | ---: | ---: | ---: |
| Off-market | Property-type boosted trees | 7.949597% | 10.379028% | 0 pp |
| List-aware | Global LightGBM log sold/list ratio | 3.934101% | 5.190150% | 0 pp |

Both scores use the same 7,773 February–May 2026 homes, with model fitting and the reference pool frozen before February 1. These are repeatedly used development backtests; hidden-test performance is unknown. The final saved models use labels before June 1, 2026, for the June–September forecast.

The official starter evaluates December–May with monthly walk-forward updating. **The selected models' official monthly walk-forward scores have not yet been produced.** A fixed four-month forecast, or its monthly slices, must not be presented as monthly refitting. Archived starter monthly scores in `evidence/starter_monthly_validation.json` describe the starter models and use different samples and updating rules.

To independently recompute the indexed development scores from archived predictions and the official observed training labels:

```sh
python score_archive.py --data-dir /path/to/data
```

This command checks IDs, forecast hashes and scores; it does not fit models or recreate walk-forward validation.

## Repository contents

| Location | Purpose |
| --- | --- |
| `run.py`, `data.py`, `check_submission.py` | Supported final-prediction entry point, official data loading and official CSV checking |
| `experiments/` | Feature builders and saved-model loaders required at prediction time |
| `saved_models/` | Six Off-market property-type branches and one List-aware model, with frozen schemas and parameters |
| `submissions/` | Off-market: 3,974 rows; List-aware: 4,064 rows; columns `ml_num,predicted_price` |
| `evidence/` | Current selected development predictions, comparisons, configurations and historical verification records |
| `research_source/` | Historical training and experiment code for inspection; see its [scope note](research_source/README.md) |
| `docs/` | Method explanation and official requirement snapshots |
| `PACKAGE_MANIFEST.json` | Actual portable file hashes for this release |

See [method and limitations](docs/METHOD.md) and [official upload requirements](docs/OFFICIAL_REQUIREMENTS.md). Historical research IDs and tool names are retained where needed for traceability. `evidence/*/protocol_portable.json` replaces the original local workspace prefix with `ORIGINAL_WORKSPACE`; original protocols and experiments remain preserved locally. Historical verification files are records of earlier checks, not new checks of this reorganized directory. The release checks are in `evidence/release_verification.json`.

## Submission status

Publishing this code repository is one part of the competition submission. The final report is maintained separately; an old report PDF is intentionally not included. Team identity, the YouTube link, the final PDF and submission through the organizer's form still need completion. The PDF must link to this accessible team repository. No form submission has been made by publishing this code.

Starter code is credited to [Erzheng Zhang's official repository](https://github.com/erwinzhang7/SSU-Data-Dash); its MIT license is preserved in [`LICENSE`](LICENSE).
