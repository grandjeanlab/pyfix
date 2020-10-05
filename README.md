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

## Usage

See usage examples [here](pyfix/examples/examples.ipynb).

## Citation
1. G. Salimi-Khorshidi, G. Douaud, C.F. Beckmann, M.F. Glasser, L. Griffanti S.M. Smith. Automatic denoising of functional MRI data: Combining independent component analysis and hierarchical fusion of classifiers. NeuroImage, 90:449-68, 2014
2. L. Griffanti, G. Salimi-Khorshidi, C.F. Beckmann, E.J. Auerbach, G. Douaud, C.E. Sexton, E. Zsoldos, K. Ebmeier, N. Filippini, C.E. Mackay, S. Moeller, J.G. Xu, E. Yacoub, G. Baselli, K. Ugurbil, K.L. Miller, and S.M. Smith. ICA-based artefact removal and accelerated fMRI acquisition for improved resting state network imaging. NeuroImage, 95:232-47, 2014 
