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

        if not op.exists(struct_fast_seg):
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

        # check if HCP-style filtered motion parameters (*_mcf_conf_hp.nii.gz) exist
        for fname in ('prefiltered_func_data_mcf.par', 'prefiltered_func_data_mcf_conf_hp.nii.gz'):
            if op.exists(op.join(meldir, 'mc', fname)):
                mp = op.join(meldir, 'mc', fname)

        # filter, and normalise motion confounds, as required
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
            d.timecourses['motparams'] = TimeCourse(mp, fs=1/tr)

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
                     features: List[str]):

    log.info("Extract features")

    # setup fixdir

    if not op.exists(data.fixdir):
        os.makedirs(data.fixdir)

    # extract features
    rows = []
    for ic_idx in range(data.nic):

        log.info(f'Extracting features for IC {ic_idx}')

        row0 = [feature.FEATURE_EXTRACTORS[fe](data, ic_idx) for fe in features]
        rows.append(pd.concat(row0, verify_integrity=True))

    features = pd.concat(rows, axis=1).T

    return features


def _index_img(img, idx):
    return nb.Nifti1Image(img.get_fdata()[:, :, :, idx], img.affine, img.header)