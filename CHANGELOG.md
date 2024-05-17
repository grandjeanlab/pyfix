This file contains the pyFIX release history in reverse chronological order.

# 0.8.0 (Friday 17th May 2024)

 - A directory containing custom mask images for feature extraction can be
   passed via the `--species` argument.


# 0.7.0 (Tuesday 20th February 2024)

 - New pre-trained `NHP_HCP_MacaqueCyno` model, for NHP-HCP BOLD fMRI of *Macaca
   fascicularis*, provided by Takuya Hayashi.
 - Added the macaque mask images used with the `NHP_HCP_Macaque` model, as they
   had been accidentally omitted in the previous two versions.
 - Built-in models can now be referred to by their alias (as specified in
   `pyfix/resources/models/info.json`), or their file name prefix.
 - The species ID that is specified during feature extraction/training, and
   saved in model files, is used when running feature extraction,
   classification, and clean-up.


# 0.6.0 (Thursday 15th February 2024)

 - New `convert_fix_model` script which will convert an old FIX `.RData`
   model file into a pyFIX model file. Note that this script requires
   R to be installed.


# 0.5.0 (Tuesday 13th February 2024)

 - Adjust pyFIX so that it can be used on existing data sets (where features
   have already beeen extracted) which do not contain all features.
 - Add new `--species` option, used during feature extraction and model
   training, which (at present) controls which brain mask images are used
   on input data sets.
 - Add new brain masks for use with NHP-HCP Macaque data sets.
 - Add new `NHP_HCP_Macaque` and `HCP_Style_Single_Multirun_Dedrift` built-in
   models - these are equivalents of the analagous models that were released
   with old FIX.


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
