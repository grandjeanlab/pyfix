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
    def from_melodic_dir(cls, meldir: str, fixdir: Optional[str] = None):

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

        #  structural segmentations

        if ~op.exists(struct_fast_seg):
            # TODO: move to feature with caching so it is only run JIT
            log.info('Creating structural segmentation with FSL FAST')

            util.run([
                'fast', '-t', '1', '-o',
                op.join(fixdir, 'fastsg'),
                struct.get_filename(),
            ])
            struct_fast_seg = op.join(fixdir, 'fastsg_pveseg.nii.gz')

        dseg = nb.load(struct_fast_seg)

        # TODO: add support for wmparc
        if op.exists(struct_fs_wmparc):
            pass

        # standard-to-func transforms

        standard = nb.load(op.join(meldir, 'reg/standard.nii.gz'))
        std2func_xfm = None

        std2func_mat = op.join(meldir, 'reg/standard2example_func.mat')
        if op.exists(std2func_mat):
            std2func_xfm = std2func_mat

        std2func_warp = op.join(meldir, 'reg/standard2example_func_warp.nii.gz')
        if op.exists(std2func_warp):
            std2func_xfm = std2func_warp

        # highpass settings

        fsf = featanalysis.loadSettings(meldir)

        hp = None
        if int(fsf['temphp_yn']) == 1:
            hp = float(fsf['paradigm_hp'])

        # motion parameters

        mp = np.loadtxt(op.join(meldir, 'mc/prefiltered_func_data_mcf.par'))

        # instantiate class

        d = cls(
            fixdir=fixdir,
            ica=ICA(
                maps=NiftiMap(ic_map),
                timecourses=TimeCourse(ic_timeseries, fs=1/tr),
            ),
            timecourses={
                'RotX': TimeCourse(mp[:, 0], fs=1/tr),
                'RotY': TimeCourse(mp[:, 1], fs=1/tr),
                'RotZ': TimeCourse(mp[:, 2], fs=1/tr),
                'X': TimeCourse(mp[:, 3], fs=1/tr),
                'Y': TimeCourse(mp[:, 4], fs=1/tr),
                'Z': TimeCourse(mp[:, 5], fs=1/tr),
            },
            spatialmaps={
                'dseg': NiftiMap(dseg, xfm=struct2func_xfm),
                'brainmask': NiftiMap(func_brainmask),
                'struct': NiftiMap(struct, xfm=struct2func_xfm),
                'standard': NiftiMap(standard, xfm=std2func_xfm)
            },
            hp_fwhm=hp,
            input=NiftiTimeCourse(func, fs=1/tr),
        )

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


# def from_feat_dir(featdir: str):
#     return from_melodic_dir(featdir)
#
#
# def from_melodic_dir(meldir: str):
#     args = {
#         'func': op.join(meldir, 'filtered_func_data.nii.gz'),
#         'func_brainmask': op.join(meldir, 'mask.nii.gz'),
#         'struct': op.join(meldir, 'reg/highres.nii.gz'),
#         'struct2func_xfm': op.join(meldir, 'reg/highres2example_func.mat')
#
#     }
#
#     util.assert_file_exists(*(args.values()))
#
#     struct_fast_seg = op.join(meldir, 'reg/highres_pveseg.nii.gz')
#     # struct_fs_wmparc = op.join(meldir, 'reg/wmparc.nii.gz')
#
#     if op.exists(struct_fast_seg):
#         args['struct_dseg'] = struct_fast_seg
#         args['struct_dseg_type'] = SegType.fsl_fast
#
#     # TODO: add support for wmparc
#     # if op.exists(struct_fs_wmparc):
#     #     args['struct_dseg'] = struct_fs_wmparc
#     #     args['struct_dseg_type'] = SegType.freesurfer_wmparc
#
#     std2func_mat = op.join(meldir, 'reg/standard2example_func.mat')
#     std2func_warp = op.join(meldir, 'reg/standard2example_func_warp.nii.gz')
#
#     if op.exists(std2func_mat):
#         args['standard2func_xfm'] = std2func_mat
#
#     if op.exists(std2func_warp):
#         args['standard2func_xfm'] = std2func_warp
#
#     args['ic_timeseries'] = op.join(
#         meldir, 'filtered_func_data.ica', 'melodic_mix')
#     args['ic_map'] = op.join(
#         meldir, 'filtered_func_data.ica', 'melodic_IC.nii.gz')
#
#     args['mot_param'] = op.join(meldir, 'mc/prefiltered_func_data_mcf.par')
#
#     fixdir = op.join(meldir, 'pyfix')
#
#     return extract_features(fixdir=fixdir, **args)


# # TODO: add get data mean, calculate when required, cache as appropriate
# # TODO: add get ic_map_abs, calculate when required, cache as appropriate
# # TODO: add maps in other spaces, with xfm to dataspace, only transform as required, cache as appropriate.
# @dataclass(frozen=True)
# class InputData:
#     """Container for input data."""
#     ic_timeseries: np.ndarray
#     ic_map: np.ndarray
#     fs: float
#     fixdir: str
#     data: Optional[np.ndarray] = None
#     standard2data_xfm: Optional[str] = None
#     timeseries: Optional[dict] = field(default_factory=dict)
#     hp_fwhm: Optional[float] = None
#     # TODO: make timeseries a namedtuple
#     map_pixdim: Optional[np.array] = None
#     _cache: dict = field(default_factory=dict, init=False)
#
#     #######
#     # PROPERTIES
#     #######
#
#     @property
#     def tr(self) -> float:
#         return 1/self.fs
#
#     @property
#     def srate(self) -> float:
#         return self.fs
#
#     @property
#     def nic(self) -> int:
#         return self.ic_map.shape[-1]
#
#     @property
#     def ntime(self) -> int:
#         return self.ic_timeseries.shape[0]
#
#     @property
#     def nspace(self) -> int:
#         return np.prod(self.ic_map.shape[:-1])
#
#     @property
#     def has_data(self) -> bool:
#         return self.data is not None
#
#     @property
#     def ic_map_abs(self):
#         if 'ic_map_abs' not in self._cache:
#             self._cache['ic_map_abs'] = np.abs(self.ic_map)
#         return self._cache['ic_map_abs']
#
#     @property
#     def data_tmean(self):
#         if 'data_tmean' not in self._cache:
#             self._cache['data_tmean'] = np.mean(self.data, axis=-1) if self.has_data else None
#         return self._cache['data_tmean']
#
#     #######
#     # METHODS
#     #######
#
#     def get_ic(self, idx):
#         return IC(self.ic_map[..., idx], self.ic_timeseries[:, idx])
#
#     def get_ic_map(self, idx, nifti=False):
#         icmap = self.ic_map[..., idx]
#
#         if nifti:
#             fname = op.abspath(op.join(self.fixdir, f'icmap{idx}.nii.gz'))
#             InputData.save_map(icmap, self.map_pixdim, fname)
#             icmap = fname
#
#         return icmap
#
#     def get_ic_map_abs(self, idx, nifti=False):
#         icmap = self.ic_map_abs[..., idx]
#
#         if nifti:
#             fname = op.abspath(op.join(self.fixdir, f'icmap{idx}_abs.nii.gz'))
#             InputData.save_map(icmap, self.map_pixdim, fname)
#             icmap = fname
#
#         return icmap
#
#
#     def has(self, prop) -> bool:
#         return (hasattr(self, prop)) & (getattr(self, prop) is not None)
#
#     def __post_init__(self):
#
#         nic_maps = self.ic_map.shape[-1]
#         nic_ts = self.ic_timeseries.shape[1]
#
#         assert nic_maps == nic_ts, \
#             f'The number of ICs in the spatial maps ({nic_maps}) and timeseries do not match ({nic_ts}).'
#
#         if self.has_data:
#             assert self.data.shape[-1] == self.ntime, \
#                 f'The number of time samples in the data ({self.data.shape[-1]}) does not match the ic_timeseries ({self.ntime})'
#
#             # assert self.data.shape[1] == self.nspace, \
#             #     f'The number of space samples in the data ({self.data.shape[1]}) does not match the ic_map ({self.nspace})'
#
#         for k, v in self.timeseries.items():
#             assert v.shape[0] == self.ntime, \
#                 f'The number of time samples in timeseries {k} ({v.shape[0]}) does not match the ic_timeseries ({self.ntime})'
#
#     #######
#     # STATIC METHODS
#     #######
#
#     @staticmethod
#     @cached(cache={}, key=lambda map0, pixdim, fname: hashkey(map0.tostring(), pixdim, fname))
#     def save_map(map0, pixdim, fname):
#
#         img = nb.Nifti1Image(np.array(map0), affine=np.eye(4))
#
#         print(f'save: {fname}')
#
#         pxd = img.header['pixdim']
#         pxd[1:1 + len(pixdim)] = pixdim
#
#         img.header['pixdim'] = pxd
#         img.to_filename(fname)
#
#     #######
#     # CLASS METHODS
#     #######
#
#     @classmethod
#     def from_osl(cls, fname, fixdir=None):
#
#         ica = io.loadmat(fname)['ica']
#
#         # IC time courses [time x numICs]
#         ic_timeseries = ica[0, 0]['tc']
#
#         # IC spatial maps [sensors x numICs]
#         ic_map = ica[0, 0]['sm']
#
#         # sampling rate
#         srate = ica[0, 0]['fsample'][0][0]
#
#         # EOG, EMG, ECG
#         eog = ica[0, 0]['EOG'].T
#         emg = ica[0, 0]['EMG'].T
#         ecg = ica[0, 0]['ECG'].T
#
#         # component labels
#         labels = ica[0, 0]['bad_components']
#         labels = np.array(labels)
#
#         # fixdir
#
#         if fixdir is None:
#             fixdir = op.join(op.splitext(fname)[0], 'pyfix')
#
#         # instantiate class
#
#         d = cls(
#             ic_map=ic_map,
#             ic_timeseries=ic_timeseries,
#             fs=srate,
#             timeseries={
#                 'eog': eog,
#                 'emg': emg,
#                 'ecg': ecg,
#             },
#             fixdir=fixdir,
#         )
#
#         return d, labels
#
#
#     @classmethod
#     def from_melodic_dir(cls, meldir: str, fixdir: Optional[str]=None):
#
#         func = nb.load(op.join(meldir, 'filtered_func_data.nii.gz'))
#         tr = func.header.get_zooms()[3]
#
#         ic_timeseries = np.loadtxt(op.join(meldir, 'filtered_func_data.ica', 'melodic_mix'))
#         ic_map = nb.load(op.join(meldir, 'filtered_func_data.ica', 'melodic_IC.nii.gz'))
#
#         struct_fast_seg = op.join(meldir, 'reg/highres_pveseg.nii.gz')
#         struct_fs_wmparc = op.join(meldir, 'reg/wmparc.nii.gz')
#
#         #  structural segmentations
#
#         if op.exists(struct_fast_seg):
#             # TODO: transform struct_fast_seg to func and add as dseg
#
#             pass
#
#             # func_mean = _calc_tmean(data.data)
#             # func_mean = save_map(func_mean, data.map_pixdim, op.join(data.fixdir, f'func_mean.nii.gz'))
#
#             # seg = feature._apply_xfm(
#             #     src=struct_fast_seg,
#             #     ref=func_mean,
#             #     xfm=struct2func_xfm,
#             #     interp='nearestneighbour',
#             #     out_name=defaults.get('func_dseg', make_dir=True)
#             # )
#
#         # TODO: add support for wmparc
#         if op.exists(struct_fs_wmparc):
#             pass
#
#         # standard-to-func transforms
#
#         xfm = None
#
#         std2func_mat = op.join(meldir, 'reg/standard2example_func.mat')
#         if op.exists(std2func_mat):
#             xfm = std2func_mat
#
#         std2func_warp = op.join(meldir, 'reg/standard2example_func_warp.nii.gz')
#         if op.exists(std2func_warp):
#             xfm = std2func_warp
#
#         # fixdir
#
#         if fixdir is None:
#             fixdir = op.join(meldir, 'pyfix')
#
#         # highpass settings
#
#         fsf = featanalysis.loadSettings(meldir)
#
#         hp = None
#         if int(fsf['temphp_yn']) == 1:
#             hp = float(fsf['paradigm_hp'])
#
#         # motion parameters
#
#         mp = np.loadtxt(op.join(meldir, 'mc/prefiltered_func_data_mcf.par'))
#
#         # instantiate class
#
#         d = cls(
#             fixdir=fixdir,
#             ic_map=ic_map.get_fdata(), #.reshape((-1, ic_map.shape[-1])),
#             ic_timeseries=ic_timeseries,
#             data=func.get_fdata(), #.reshape((-1, func.shape[-1])),
#             fs=1/tr,
#             timeseries={
#                 'RotX': mp[:, 0],
#                 'RotY': mp[:, 1],
#                 'RotZ': mp[:, 2],
#                 'X': mp[:, 3],
#                 'Y': mp[:, 4],
#                 'Z': mp[:, 5],
#             },
#             standard2data_xfm=xfm,
#             hp_fwhm=hp,
#             map_pixdim=func.header.get_zooms(),
#         )
#
#         return d



def new_extract_features(data: FixData,
                         features: List[str],
                         # fixdir: Optional[str]=None,
                         feature_fname: Optional[str] = None,
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

    if feature_fname is None:
        feature_fname = op.join(data.fixdir, 'features.csv')

    features.to_csv(feature_fname, index=None)

    return features


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
