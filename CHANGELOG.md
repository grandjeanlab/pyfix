# 0.4.0 (Monday 22nd January 2024)

 - Now using the [xgboost](https://xgboost.readthedocs.io/) classifier by
   default.
 - Added a set of pre-trained models to replace those included with old FIX -
   included in this release are `Standard`, `UKBiobank`, `HCP25_2000`,
   `HCP7T_2000`, `WhII_Standard` and `WhII_MB6`.
 - Fixed a range of bugs in feature extraction.
 - Implemented aggressive cleanup.
 - The training stage now has an optional hyper-parameter optimisation step
   using [optuna](https://optuna.readthedocs.io/).
