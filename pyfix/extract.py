#!/usr/bin/env python
import logging
import os
import os.path as op
import numpy as np
import shutil
import nibabel as nb
import pyfix.feature as feature
import scipy
from pyfix.util import SegType
from pyfix import util
from fsl.data.image import Image
from fsl.wrappers import bet, fslmaths, flirt
from typing import Optional, Union, List
from fsl.utils.filetree import FileTree
import pandas as pd

log = logging.getLogger(__name__)
DEFAULTS = FileTree.read(util.get_resource('pyfix.tree'), partial_fill=False)


def from_feat_dir(featdir: str):
    return from_melodic_dir(featdir)


def from_melodic_dir(meldir: str):
    args = {
        'func': op.join(meldir, 'filtered_func_data.nii.gz'),
        'func_brainmask': op.join(meldir, 'mask.nii.gz'),
        'struct': op.join(meldir, 'reg/highres.nii.gz'),
        'struct2func_xfm': op.join(meldir, 'reg/highres2example_func.mat')

    }

    util.assert_file_exists(*(args.values()))

    struct_fast_seg = op.join(meldir, 'reg/highres_pveseg.nii.gz')
    # struct_fs_wmparc = op.join(meldir, 'reg/wmparc.nii.gz')

    if op.exists(struct_fast_seg):
        args['struct_dseg'] = struct_fast_seg
        args['struct_dseg_type'] = SegType.fsl_fast

    # TODO: add support for wmparc
    # if op.exists(struct_fs_wmparc):
    #     args['struct_dseg'] = struct_fs_wmparc
    #     args['struct_dseg_type'] = SegType.freesurfer_wmparc

    std2func_mat = op.join(meldir, 'reg/standard2example_func.mat')
    std2func_warp = op.join(meldir, 'reg/standard2example_func_warp.nii.gz')

    if op.exists(std2func_mat):
        args['standard2func_xfm'] = std2func_mat

    if op.exists(std2func_warp):
        args['standard2func_xfm'] = std2func_warp

    args['ic_timeseries'] = op.join(
        meldir, 'filtered_func_data.ica', 'melodic_mix')
    args['ic_map'] = op.join(
        meldir, 'filtered_func_data.ica', 'melodic_IC.nii.gz')

    args['mot_param'] = op.join(meldir, 'mc/prefiltered_func_data_mcf.par')

    fixdir = op.join(meldir, 'pyfix')

    return extract_features(fixdir=fixdir, **args)

# TODO: only ica_map and ica_timeseries should be req'd
# TODO: fmri (currently func), eeg, and meg should all be options
def extract_features(fixdir: str,
                     func: str,  # native functional space
                     struct2func_xfm: str,  # native structural to native functional space xfm (warp or affine)
                     ic_timeseries: str,  # make optional, run ICA
                     ic_map: str,  # make optional, run ICA
                     mot_param: str,  # make optional, run mcflirt if not provided
                     func_mean: Optional[str] = None,
                     ic_threshold: Optional[float] = 2.5,  # magic number taken from original fix
                     hp_fwhm: Optional[float] = 0,
                     func_brainmask: Optional[str] = None,  # native functional space
                     struct: Optional[str] = None,  # native structural space
                     struct_dseg: Optional[str] = None,  # native structural space
                     struct_dseg_type: Optional[SegType] = SegType.fsl_fast,
                     standard2func_xfm: Optional[str] = None,  # mni standard to native functional space xfm (warp or affine) -> None
                     overwrite: Optional[bool] = False,  # overwrite existing features if true
                     debug=False,
                     include_features: Union[str, List[str]] = 'all',
                     ):
    log.info("Extract features")

    # check main inputs
    util.assert_file_exists(func, struct2func_xfm, ic_timeseries, ic_map)

    if (struct_dseg is None) and (struct is None):
        raise RuntimeError('Either struct_dseg OR struct must be provided')

    # update defaults
    defaults = DEFAULTS.update(workdir=fixdir)

    if not op.exists(fixdir):
        os.makedirs(fixdir)

    # demean func if necessary
    # if func_mean is None:
    #     fslmaths(func).Tmean().run(defaults.get('func_mean', make_dir=True))
    # else:
    #     util.assert_file_exists(func_mean)
    #     shutil.copy(func_mean, defaults.get('func_mean', make_dir=True))
    # func_mean = defaults.get('func_mean')

    # create func_std and func_tsnr
    # fslmaths(func).Tstd().run(defaults.get('func_std', make_dir=True))
    # fslmaths(func_mean).div(defaults.get('func_std')).run(defaults.get('func_tsnr', make_dir=True))

    # prepare func_dseg and transform to func space

    # TODO: move to feature with caching
    if struct_dseg is None:
        log.info('Creating structural segmentation with FSL FAST')

        util.run([
            'fast', '-t', '1', '-o',
            op.join(fixdir, 'fastsg'),
            struct,
        ])
        struct_dseg = op.join(fixdir, 'fastsg_pveseg.nii.gz')

    # TODO: add support for freesurfer wmparc
    if struct_dseg_type is not SegType.fsl_fast:
        raise RuntimeError('Only fsl_fast segtype is currently supported. Will change in future.')

    # TODO: if brainmask not supplied, run bet or create from dseg
    if func_brainmask is None:
        raise RuntimeError('Currently func_brain_mask is always required. Will change in future.')

    # TODO: look at moving this to a feature, with caching
    # log.info('Transform structural segmentation to native functional space')
    # flirt(
    #     src=struct_dseg,
    #     ref=func_mean,
    #     applyxfm=True,
    #     init=struct2func_affine,
    #     interp='nearestneighbour',
    #     out=defaults.get('func_dseg', make_dir=True)
    # )

    # prepare data for feature extractors

    ic_map = nb.load(ic_map)
    ic_timeseries = np.loadtxt(ic_timeseries)

    func = nb.load(func)
    # func_dseg = nb.load(defaults.get('func_dseg'))

    TR = func.header.get_zooms()[3]
    sigma = np.round(hp_fwhm / (2 * TR))

    nIC = ic_map.shape[3]

    # TODO: provide a workdir (fixdir+funcname) for storing feature assets with collision
    # TODO: this should be a dataclass
    data = {
        'nIC': nIC,
        'func': func,
        'struct': nb.load(struct) if struct is not None else struct,
        'struct_dseg': struct_dseg,
        'struct2func_xfm': struct2func_xfm,
        'standard2func_xfm': standard2func_xfm,
        'func_brainmask': nb.load(func_brainmask),
        'ic_threshold': ic_threshold,
        'pixdim': func.header.get_zooms(),
        'TR': TR,
        'hp_fwhm': hp_fwhm,
        'motion_parameters': mot_param,
        # 'func_mean': nb.load(func_mean),
        'fixdir': defaults.get('fixdir', make_dir=True),
        'workdir': defaults.get('fixdir', make_dir=True),
        # 'motion_parameters_norm': feature.motionconfounds(mot_param, sigma, TR),
        # 'csf_timeseries': np.mean(func.get_fdata()[func_dseg.get_fdata() == 1], axis=0),
        # 'gm_timeseries': np.mean(func.get_fdata()[func_dseg.get_fdata() == 2], axis=0),
        # 'wm_timeseries': np.mean(func.get_fdata()[func_dseg.get_fdata() == 3], axis=0),
    }

    # loop through ICs and extract features

    # TODO: add legacy include_features list
    if include_features == 'all':
        include_features = feature.FEATURE_EXTRACTORS.keys()

    for f in include_features:
        if f not in feature.FEATURE_EXTRACTORS.keys():
            raise RuntimeError(f'Unknown feature: {f}')

    rows = []
    for ic_idx in range(nIC):

        log.info(f"Extract features for IC {ic_idx}")

        data['ic_index'] = ic_idx

        # select single IC
        data['ic_timeseries'] = ic_timeseries[:, ic_idx]

        ic_map0 = _index_img(ic_map, ic_idx)
        ic_map0.to_filename(op.join(defaults.get('fixdir'), 'icmap0.nii.gz'))  # some features require the icmap on disk
        data['ic_map'] = ic_map0

        ic_map0_abs = nb.Nifti1Image(np.abs(ic_map0.get_fdata()), ic_map0.affine, ic_map0.header)
        ic_map0_abs.to_filename(op.join(defaults.get('fixdir'), 'icmap0_abs.nii.gz'))
        data['ic_map_abs'] = ic_map0_abs

        # TODO: change to loop instead of list comprehension
        # loop through feature extractors
        row0 = [feature.FEATURE_EXTRACTORS[fe](data) for fe in include_features]
        rows.append(pd.concat(row0, verify_integrity=True))

    features = pd.concat(rows, axis=1).T

    features.to_csv(defaults.get('features', make_dir=True), index=None)

    return features


def _index_img(img, idx):
    return nb.Nifti1Image(img.get_fdata()[:, :, :, idx], img.affine, img.header)

# def _apply_fast(
#         self,
#         struct: nb.Nifti1Image,
#         struct_seg_type,
#         struct_seg: nb.Nifti1Image,
#         func: nb.Nifti1Image,
#         struct2func_rigid: str) -> None:
#
#     # create structual segmentation if it does not exist
#     if struct_seg is None:
#         log.info('Creating structural segmentation with FSL FAST')
#         util.run([
#             'fast', '-t', '1', '-o',
#             op.join(self.workdir, 'fastsg'),
#             struct.get_filename()
#         ])
#         struct_seg = nb.load(op.join(self.workdir, 'fastsg_pveseg.nii.gz'))
#
#     # TODO: add support for freesurfer wmparc
#     assert struct_seg_type is SegType.fsl_fast, (
#         'Segmentation type not currently supported')
#
#     # transform structural segmentation to native functional space
#     log.info('Transform structural segmentation to native functional space')
#     func_seg = op.join(self.workdir, 'func_seg.nii.gz')
#     util.run([
#         'flirt',
#         '-in', struct_seg.get_filename(),
#         '-ref', func.get_filename(),
#         '-applyxfm',
#         '-init', struct2func_rigid,
#         '-interp', 'nearestneighbour',
#         '-out', func_seg
#     ])
#     func_seg = nb.load(func_seg)
#
#     return func_seg
