# Monthly validation summary

These files summarize the completed December 2025–May 2026 monthly re-evaluation of the current selected specifications. They include all scored home predictions, metrics, diagnostics and the report wording. They are cumulative development results; configuration selection overlapped December 2025, and hidden-test error is unknown.

The 42 refitted models, training IDs and full fold audits are not bundled in this compact summary. Generate a complete new run with `python run.py validate --data-dir /path/to/data --output /new/run`, then audit that full output folder with `python verify_official_validation.py --data-dir /path/to/data --output /new/run`. The full verifier expects the generated run, not this summary folder.
