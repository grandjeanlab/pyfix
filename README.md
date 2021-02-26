# pyFIX

This package contains a python port of [FMRIB's ICA-based Xnoiseifier](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIX) (`FIX`).

FIX attempts to auto-classify ICA components into "good" vs "bad" components, so that the bad components can be removed from the 4D FMRI data. FIX is intended to be run on single-session [MELODIC](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC) ICA output. 

> This python port attempts to emulate `FIX` as closely as possible but is *not* a like-for-like replacement.  The primary difference is that the ensemble classifier that was used in `FIX`, and was implemented in `R`, is *not* used by `pyFIX`.  Instead we allow the user to use any classifier that conforms to the [scikit-learn](https://scikit-learn.org/stable/) classifier API.  We may, at a later stage, implement an ensemble classifier in `pyFIX` that is inspired by the classifier in `FIX`.

## Installation
```shell
# install development version from gitlab with pip
pip install git+https://git.fmrib.ox.ac.uk/seanf/pyfix.git

# OR, clone locally and install with pip
git clone https://git.fmrib.ox.ac.uk/seanf/pyfix.git
pip install -e pyfix
```
We recommend that you create your own virtual environment first (e.g.):
```shell
conda create -n pyfix_env python=3.8
```

## Usage

### Python Interface

PyFIX was designed to be more flexible than `legacy FIX`, and the recommended way to interact with `pyFIX` is via the python API.

1. [fMRI Feature Extraction](pyfix/examples/fmri_feature_extraction.md): Example of creating a `FixData` object for fMRI and extracting `legacy FIX` features.
2. [fMRI Training and Classification](pyfix/examples/fmri_train.md): Example of training a `pyFIX` model on hand labelled fMRI data, and applying that model to the data from a single subject.
3. [Custom Feature Extractors](pyfix/examples/custom_feature_extractors.md): Example of how to create your own custom feature extractors.

### Legacy CLI

However, there is a drop-in command line replacement for `legacy FIX` for fMRI.  Simply call `fix` on the command line:
```shell
>> fix

FIX (FMRIB's ICA-based Xnoiseifier)

Simple usage, assuming training data already exists:

  fix <mel.ica> <training.pyfix_model> <thresh>  [fix -a options]
    Extract features, classify and run cleanup - equivalent to:
      fix -f <mel.ica> ; fix -c <mel.ica> <train>.RData <thresh> ; fix -a <mel.ica>/fix4melview_<train>_<thresh>.txt [-a options]
    For the [fix -a options], see "fix -a" usage below.
    The first argument should be a FEAT/MELODIC output directory that includes a filtered_func_data.ica melodic output.
    Example:    fix study1/subject1.ica /usr/local/FIX/HCP_noHP.RData 20 -m -h 200
    Example:    fix subject2.feat /home/FIX/Standard_noHP.RData 20 -m   (ICA must have been turned on in FEAT pre-stats)

Usage for each stage separately, including creation of training data:

  fix -f <mel.ica>
    Feature extraction (for later training and/or classifying).

  fix -t <training-output-basename> [-l] <mel1.ica> <mel2.ica> ...
    Train classifier (Training output basename followed by list of Melodic directories).
    Every Melodic directory must contain hand_labels_noise.txt listing the artefact components, e.g.:  [1, 4, 99, ... 140].
    -l : optional LOO accuracy testing.

  fix -c <mel.ica> <training.pyfix_model> <thresh>
    Classify ICA components.

  fix -a <mel.ica/fix4melview_TRAIN_thr.txt>  [-m] [-h <highpass>]  [-A]
    Apply cleanup, using artefacts listed in the .txt file, to the data inside the enclosing Feat/Melodic directory.
    -h -1          apply no highpass filtering.
    -h 0           apply linear detrending only.
    -h <highpass>  with a positive <highpass> value, apply highpass with <highpass> being full-width (2*sigma) in seconds.
    -m : optionally also cleanup motion confounds, with highpass filtering of motion confounds controlled by -h:
       - if -h is omitted, fix will look to see if a design.fsf file is present, to find the highpass cutoff.
       - if -h is omitted, and no design.fsf is present, no filtering of the motion confounds will take place.
    -A : apply aggressive (full variance) cleanup, instead of the default less-aggressive (unique variance) cleanup.
```



## Citation
1. G. Salimi-Khorshidi, G. Douaud, C.F. Beckmann, M.F. Glasser, L. Griffanti S.M. Smith. Automatic denoising of functional MRI data: Combining independent component analysis and hierarchical fusion of classifiers. NeuroImage, 90:449-68, 2014
2. L. Griffanti, G. Salimi-Khorshidi, C.F. Beckmann, E.J. Auerbach, G. Douaud, C.E. Sexton, E. Zsoldos, K. Ebmeier, N. Filippini, C.E. Mackay, S. Moeller, J.G. Xu, E. Yacoub, G. Baselli, K. Ugurbil, K.L. Miller, and S.M. Smith. ICA-based artefact removal and accelerated fMRI acquisition for improved resting state network imaging. NeuroImage, 95:232-47, 2014 
