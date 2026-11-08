# Historical research source

This directory preserves the original experiment and training code for inspection, including both selected tracks and earlier comparisons. It is separate from the supported runtime code at the repository root.

Historical scripts depend on the original research archive, feature caches and artifact paths. Those complete caches and all earlier model runs are not redistributed in this compact repository. Installing the root requirements does not install every historical experiment's optional dependency. This directory is not advertised as a portable command to repeat the full research process.

Use root `run.py predict` to independently rebuild features from raw official data and rerun all seven selected final-model branches. Use root `score_archive.py` to check the indexed development scores. Frozen schemas, configurations and selected predictions are included separately. Full original experiments remain preserved in the local research project.
