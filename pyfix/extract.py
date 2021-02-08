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
from pyfix import util, clean
from fsl.data.image import Image
from fsl.wrappers import bet, fslmaths, flirt
from typing import Optional, Union, List, Type
from fsl.utils.filetree import FileTree
import pandas as pd
from dataclasses import dataclass, field
from scipy import io
from collections import namedtuple
from fsl.data import featanalysis
from cachetools import cached, cachedmethod
from cachetools.keys import hashkey
from functools import lru_cache

IC = namedtuple('IC', ['map', 'timeseries'])

log = logging.getLogger(__name__)
DEFAULTS = FileTree.read(util.get_resource('pyfix.tree'), partial_fill=False)


@dataclass
class SpatialMap:
    """Container for spatial map data."""
    data: np.ndarray

    # def __init__(self, data: np.ndarray):
    #     self._data = data

    # @property
    # def data(self):
    #     return self._data

    @property
    def nmap(self) -> int:
        return self.data.shape[-1]

    @property
    def nspace(self) -> int:
        return np.prod(self.data.shape[:-1])

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def get_map(self, idx) -> np.ndarray:
        map0 = self.data[..., idx]
        return map0


@dataclass
class NiftiMap(SpatialMap):
    """SpatialMap subclass container for NIFTI spatial maps"""
    data: nb.Nifti1Image
    xfm: Optional[str] = None

    @cached(cache={}, key=lambda self, idx: hashkey(id(self), idx))
    def get_map(self, idx) -> nb.Nifti1Image:
        # print(f'get_map: {idx}')
        dd = self.data.get_fdata()[..., idx]
        return nb.Nifti1Image(dd, affine=self.data.affine, header=self.data.header)

    def to_filename(self, fname: str) -> str:
        self.data.to_filename(fname)
        return fname

    # def get_data(self, idx, unfold=False) -> np.ndarray:
    #     map0 = self.data.get_fdata()[..., idx]
    #     if unfold:
    #         map0 = map0.ravel()
    #     return map0

    @property
    def pixdim(self):
        return self.data.header.get_zooms()


@dataclass(frozen=True)
class TimeCourse:
    """Container for timecourse data."""
    data: np.ndarray     # frames (time) x channels
    fs: float                   # sample rate (Hz)

    @property
    def tr(self) -> float:
        return 1 / self.fs

    @property
    def srate(self) -> float:
        return self.fs

    @property
    def ntime(self) -> int:
        return self.data.shape[0]

    @property
    def nchan(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self):
        return self.data.shape

    def get_timecourse(self, idx) -> np.ndarray:
        return self.data[:, idx] if self.data.ndim > 1 else self.data[:, np.newaxis][:, idx]


@dataclass(frozen=True)
class NiftiTimeCourse(TimeCourse):
    """TimeCourse subclass container for NIFTI timecourse data."""
    data: nb.Nifti1Image
    fs: float
    xfm: Optional[str] = None

    @property
    def ntime(self) -> int:
        return self.data.shape[-1]

    @property
    def nchan(self) -> int:
        return np.prod(self.data.shape[:-1])

    def get_timecourse(self, idx) -> np.ndarray:
        dd = self.data.get_fdata()
        dd = np.reshape(dd, (-1, self.ntime))
        return dd[idx, :]

    # @cached(cache={}, key=lambda self: hashkey(id(self), 'get_tmean'))
    # def get_tmean(self) -> nb.Nifti1Image:
    #     print(f'get_nifti_timecourse_tmean')
    #     dd = np.mean(self.data.get_fdata(), axis=-1)
    #     return nb.Nifti1Image(dd, affine=self.data.affine, header=self.data.header)

    @property
    def pixdim(self):
        return self.data.header.get_zooms()


@dataclass(frozen=True)
class Component:
    """Container for single IC"""
    spatialmap: np.ndarray
    timecourse: np.ndarray
    idx: int


@dataclass(frozen=True)
class ICA:
    """Container for full ICA decomposition"""
    maps: SpatialMap  # space[x,y,z] x ics
    timecourses: TimeCourse  # time x ics

    @property
    def nic(self) -> int:
        return self.maps.nmap

    @property
    def ntime(self) -> int:
        return self.timecourses.ntime

    @property
    def nspace(self) -> int:
        return self.maps.nspace

    @property
    def tr(self) -> float:
        return 1 / self.timecourses.fs

    @property
    def srate(self) -> float:
        return self.timecourses.fs

    def __post_init__(self):
        spatial_nic = self.maps.nmap
        temporal_nic = self.timecourses.nchan

        assert spatial_nic == temporal_nic, \
            f'The number of ICs in the spatial components ({spatial_nic}) and timeseries do not match ({temporal_nic}).'

    def get_ic(self, idx):
        return Component(self.maps.get_map(idx), self.timecourses.get_timecourse(idx), idx)

    def get_timecourse(self, idx):
        return self.timecourses.get_timecourse(idx)

    def get_spatialmap(self, idx):
        return self.maps.get_map(idx)



@dataclass(frozen=True)
class FixData:
    """Container for FIX input data."""
    fixdir: str
    ica: ICA
    input: Optional[Type[TimeCourse]] = None
    timecourses: Optional[dict] = field(default_factory=dict)
    spatialmaps: Optional[dict] = field(default_factory=dict)
    hp_fwhm: Optional[float] = None

    @property
    def nic(self) -> int:
        return self.ica.nic

    def __post_init__(self):

        for k, v in self.timecourses.items():
            assert v.srate == self.ica.srate, \
                f'srate of {k} timecourse ({v.srate}) must equal ICA ({self.ica.srate})'

            assert v.ntime == self.ica.ntime, \
                f'number time samples of {k} timecourse ({v.ntime}) must equal ICA ({self.ica.ntime})'

        # for k, v in self.spatialmaps.items():
        #     assert v.nspace == self.ica.nspace, \
        #         f'number spatial samples of {k} spatialmap ({v.nspace}) must equal ICA ({self.ica.nspace})'

        if self.input is not None:
            assert self.input.ntime == self.ica.ntime, \
                f'number of time samples in input data ({self.input.ntime}) must equal ICA ({self.ica.ntime})'


    @classmethod
    def from_melodic_dir(cls, meldir: str, fixdir: Optional[str] = None, highpass_fwhm=None):

        # fixdir

        if fixdir is None:
            fixdir = op.join(meldir, 'pyfix')

        # melodic data

        func = nb.load(op.join(meldir, 'filtered_func_data.nii.gz'))
        tr = func.header.get_zooms()[3]

        func_brainmask = nb.load(op.join(meldir, 'mask.nii.gz'))

        ic_timeseries = np.loadtxt(op.join(meldir, 'filtered_func_data.ica', 'melodic_mix'))
        ic_map = nb.load(op.join(meldir, 'filtered_func_data.ica', 'melodic_IC.nii.gz'))

        struct_fast_seg = op.join(meldir, 'reg/highres_pveseg.nii.gz')
        struct_fs_wmparc = op.join(meldir, 'reg/wmparc.nii.gz')

        struct = nb.load(op.join(meldir, 'reg/highres.nii.gz'))
        struct2func_xfm = op.join(meldir, 'reg/highres2example_func.mat')

        veins_exf = op.join(meldir, 'reg/veins_exf.nii.gz')

        #  structural segmentations

        if ~op.exists(struct_fast_seg):
            # TODO: move to feature with caching so it is only run JIT
            log.info('Creating structural segmentation with FSL FAST')

            if not op.exists(fixdir):
                os.makedirs(fixdir)

            util.run([
                'fast', '-t', '1', '-o',
                op.join(fixdir, 'fastsg'),
                struct.get_filename(),
            ])
            struct_fast_seg = op.join(fixdir, 'fastsg_pveseg.nii.gz')

        dseg = nb.load(struct_fast_seg)

        # Load subcortical mask (if it exists) and add to fast pveseg
        if op.exists(struct_fs_wmparc):
            subcort = util.create_mask(
                dseg=struct_fs_wmparc,
                labels=(10, 11, 12, 13, 49, 50, 51, 52, 26, 58),
                outname=op.join(fixdir, 'subcort.nii.gz')
            )

            dseg = nb.load(struct_fast_seg)
            dseg_d = dseg.get_fdata()
            dseg_d[subcort.get_fdata().astype(bool)] = 2
            dseg = nb.Nifti1Image(dseg_d.astype(int), header=dseg.header, affine=dseg.affine)
            dseg.to_filename(struct_fast_seg)


        # standard-to-func transforms

        standard = op.join(meldir, 'reg/standard.nii.gz')
        std2func_xfm = None

        std2func_mat = op.join(meldir, 'reg/standard2example_func.mat')
        if op.exists(std2func_mat):
            std2func_xfm = std2func_mat

        std2func_warp = op.join(meldir, 'reg/standard2example_func_warp.nii.gz')
        if op.exists(std2func_warp):
            std2func_xfm = std2func_warp

        # highpass settings

        if op.exists(op.join(meldir, 'design.fsf')):
            fsf = featanalysis.loadSettings(meldir)
            if int(fsf['temphp_yn']) == 1:
                highpass_fwhm = float(fsf['paradigm_hp'])

        # motion parameters

        mp = None
        for fname in ('prefiltered_func_data_mcf.par', 'prefiltered_func_data_mcf_conf_hp.nii.gz'):
            if op.exists(op.join(meldir, 'mc', fname)):
                mp = op.join(meldir, 'mc', fname)

        if mp is not None:
            mp = clean.motion_confounds(mp, tr=tr, temporal_fwhm=highpass_fwhm, tmpdir=meldir)
            mp = clean._normalise(mp)


        # instantiate class

        d = cls(
            fixdir=fixdir,
            ica=ICA(
                maps=NiftiMap(ic_map),
                timecourses=TimeCourse(ic_timeseries, fs=1/tr),
            ),
            spatialmaps={
                'dseg': NiftiMap(dseg, xfm=struct2func_xfm),
                'brainmask': NiftiMap(func_brainmask),
                'struct': NiftiMap(struct, xfm=struct2func_xfm),
                # 'standard': NiftiMap(standard, xfm=std2func_xfm)
            },
            hp_fwhm=highpass_fwhm,
            input=NiftiTimeCourse(func, fs=1/tr),
        )

        if op.exists(standard):
            d.spatialmaps['standard'] = NiftiMap(standard, xfm=std2func_xfm)

        # motion parameters
        if mp is not None:
            d.timecourses['motparams'] = mp

        # veins
        if op.exists(veins_exf):
            d.spatialmaps['veins_exf'] = NiftiMap(nb.load(veins_exf))

        return d

    @classmethod
    def from_osl(cls, fname, fixdir=None):

        ica = io.loadmat(fname)['ica']

        # IC time courses [time x numICs]
        ic_timeseries = ica[0, 0]['tc']

        # IC spatial maps [sensors x numICs]
        ic_map = ica[0, 0]['sm']

        # sampling rate
        srate = ica[0, 0]['fsample'][0][0]

        # EOG, EMG, ECG
        eog = ica[0, 0]['EOG'].T
        emg = ica[0, 0]['EMG'].T
        ecg = ica[0, 0]['ECG'].T

        # component labels
        labels = ica[0, 0]['bad_components']
        labels = np.array(labels)

        # fixdir

        if fixdir is None:
            fixdir = op.join(op.splitext(fname)[0], 'pyfix')

        # instantiate class

        d = cls(
            fixdir=fixdir,
            ica=ICA(
                maps=SpatialMap(ic_map),
                timecourses=TimeCourse(ic_timeseries, fs=srate),
            ),
            timecourses={
                'eog': TimeCourse(eog, fs=srate),
                'ecg': TimeCourse(ecg, fs=srate),
                'emg': TimeCourse(emg, fs=srate),
            },
        )

        return d, labels


def extract_features(data: FixData,
                         features: List[str],
                         # fixdir: Optional[str]=None,
                         # feature_fname: Optional[str] = None,
                         ):

    log.info("Extract features")

    # setup fixdir

    # if fixdir is not None:
    #     data.fixdir = fixdir

    if not op.exists(data.fixdir):
        os.makedirs(data.fixdir)

    # # update defaults
    # defaults = DEFAULTS.update(workdir=fixdir)

    # extract features
    rows = []
    for ic_idx in range(data.nic):

        row0 = [feature.FEATURE_EXTRACTORS[fe](data, ic_idx) for fe in features]
        rows.append(pd.concat(row0, verify_integrity=True))

    features = pd.concat(rows, axis=1).T

    # if feature_fname is None:
    #     feature_fname = op.join(data.fixdir, 'features.csv')
    #
    # features.to_csv(feature_fname, index=None)

    return features


# # TODO: only ica_map and ica_timeseries should be req'd
# # TODO: fmri (currently func), eeg, and meg should all be options
# def extract_features(fixdir: str,
#                      func: str,  # native functional space
#                      struct2func_xfm: str,  # native structural to native functional space xfm (warp or affine)
#                      ic_timeseries: str,  # make optional, run ICA
#                      ic_map: str,  # make optional, run ICA
#                      mot_param: str,  # make optional, run mcflirt if not provided
#                      func_mean: Optional[str] = None,
#                      ic_threshold: Optional[float] = 2.5,  # magic number taken from original fix
#                      hp_fwhm: Optional[float] = 0,
#                      func_brainmask: Optional[str] = None,  # native functional space
#                      struct: Optional[str] = None,  # native structural space
#                      struct_dseg: Optional[str] = None,  # native structural space
#                      struct_dseg_type: Optional[SegType] = SegType.fsl_fast,
#                      standard2func_xfm: Optional[str] = None,  # mni standard to native functional space xfm (warp or affine) -> None
#                      overwrite: Optional[bool] = False,  # overwrite existing features if true
#                      debug=False,
#                      include_features: Union[str, List[str]] = 'all',
#                      ):
#     log.info("Extract features")
#
#     # check main inputs
#     util.assert_file_exists(func, struct2func_xfm, ic_timeseries, ic_map)
#
#     if (struct_dseg is None) and (struct is None):
#         raise RuntimeError('Either struct_dseg OR struct must be provided')
#
#     # update defaults
#     defaults = DEFAULTS.update(workdir=fixdir)
#
#     if not op.exists(fixdir):
#         os.makedirs(fixdir)
#
#     # demean func if necessary
#     # if func_mean is None:
#     #     fslmaths(func).Tmean().run(defaults.get('func_mean', make_dir=True))
#     # else:
#     #     util.assert_file_exists(func_mean)
#     #     shutil.copy(func_mean, defaults.get('func_mean', make_dir=True))
#     # func_mean = defaults.get('func_mean')
#
#     # create func_std and func_tsnr
#     # fslmaths(func).Tstd().run(defaults.get('func_std', make_dir=True))
#     # fslmaths(func_mean).div(defaults.get('func_std')).run(defaults.get('func_tsnr', make_dir=True))
#
#     # prepare func_dseg and transform to func space
#
#     # TODO: move to feature with caching
#     if struct_dseg is None:
#         log.info('Creating structural segmentation with FSL FAST')
#
#         util.run([
#             'fast', '-t', '1', '-o',
#             op.join(fixdir, 'fastsg'),
#             struct,
#         ])
#         struct_dseg = op.join(fixdir, 'fastsg_pveseg.nii.gz')
#
#     # TODO: add support for freesurfer wmparc
#     if struct_dseg_type is not SegType.fsl_fast:
#         raise RuntimeError('Only fsl_fast segtype is currently supported. Will change in future.')
#
#     # TODO: if brainmask not supplied, run bet or create from dseg
#     if func_brainmask is None:
#         raise RuntimeError('Currently func_brain_mask is always required. Will change in future.')
#
#     # TODO: look at moving this to a feature, with caching
#     # log.info('Transform structural segmentation to native functional space')
#     # flirt(
#     #     src=struct_dseg,
#     #     ref=func_mean,
#     #     applyxfm=True,
#     #     init=struct2func_affine,
#     #     interp='nearestneighbour',
#     #     out=defaults.get('func_dseg', make_dir=True)
#     # )
#
#     # prepare data for feature extractors
#
#     ic_map = nb.load(ic_map)
#     ic_timeseries = np.loadtxt(ic_timeseries)
#
#     func = nb.load(func)
#     # func_dseg = nb.load(defaults.get('func_dseg'))
#
#     TR = func.header.get_zooms()[3]
#     sigma = np.round(hp_fwhm / (2 * TR))
#
#     nIC = ic_map.shape[3]
#
#     # TODO: provide a workdir (fixdir+funcname) for storing feature assets with collision
#     # TODO: this should be a dataclass
#     data = {
#         'nIC': nIC,
#         'func': func,
#         'struct': nb.load(struct) if struct is not None else struct,
#         'struct_dseg': struct_dseg,
#         'struct2func_xfm': struct2func_xfm,
#         'standard2func_xfm': standard2func_xfm,
#         'func_brainmask': nb.load(func_brainmask),
#         'ic_threshold': ic_threshold,
#         'pixdim': func.header.get_zooms(),
#         'TR': TR,
#         'hp_fwhm': hp_fwhm,
#         'motion_parameters': mot_param,
#         # 'func_mean': nb.load(func_mean),
#         'fixdir': defaults.get('fixdir', make_dir=True),
#         'workdir': defaults.get('fixdir', make_dir=True),
#         # 'motion_parameters_norm': feature.motionconfounds(mot_param, sigma, TR),
#         # 'csf_timeseries': np.mean(func.get_fdata()[func_dseg.get_fdata() == 1], axis=0),
#         # 'gm_timeseries': np.mean(func.get_fdata()[func_dseg.get_fdata() == 2], axis=0),
#         # 'wm_timeseries': np.mean(func.get_fdata()[func_dseg.get_fdata() == 3], axis=0),
#     }
#
#     # loop through ICs and extract features
#
#     # TODO: add legacy include_features list
#     if include_features == 'all':
#         include_features = feature.FEATURE_EXTRACTORS.keys()
#
#     for f in include_features:
#         if f not in feature.FEATURE_EXTRACTORS.keys():
#             raise RuntimeError(f'Unknown feature: {f}')
#
#     rows = []
#     for ic_idx in range(nIC):
#
#         log.info(f"Extract features for IC {ic_idx}")
#
#         data['ic_index'] = ic_idx
#
#         # select single IC
#         data['ic_timeseries'] = ic_timeseries[:, ic_idx]
#
#         ic_map0 = _index_img(ic_map, ic_idx)
#         ic_map0.to_filename(op.join(defaults.get('fixdir'), 'icmap0.nii.gz'))  # some features require the icmap on disk
#         data['ic_map'] = ic_map0
#
#         ic_map0_abs = nb.Nifti1Image(np.abs(ic_map0.get_fdata()), ic_map0.affine, ic_map0.header)
#         ic_map0_abs.to_filename(op.join(defaults.get('fixdir'), 'icmap0_abs.nii.gz'))
#         data['ic_map_abs'] = ic_map0_abs
#
#         # TODO: change to loop instead of list comprehension
#         # loop through feature extractors
#         row0 = [feature.FEATURE_EXTRACTORS[fe](data) for fe in include_features]
#         rows.append(pd.concat(row0, verify_integrity=True))
#
#     features = pd.concat(rows, axis=1).T
#
#     features.to_csv(defaults.get('features', make_dir=True), index=None)
#
#     return features


def _index_img(img, idx):
    return nb.Nifti1Image(img.get_fdata()[:, :, :, idx], img.affine, img.header)