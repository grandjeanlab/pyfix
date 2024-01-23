#!/usr/bin/env python

import                    logging
import                    os
import os.path     as     op
from   collections import namedtuple
from   dataclasses import dataclass, field
from   typing      import Optional, List, Type

import nibabel as nb
import numpy   as np
import pandas  as pd

from cachetools      import cached
from cachetools.keys import hashkey

from fsl.data     import featanalysis
from fsl.wrappers import fast
from file_tree    import FileTree


import pyfix.feature as feature
from pyfix import util, clean, io

IC = namedtuple('IC', ['map', 'timeseries'])

log = logging.getLogger(__name__)


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
    data: np.ndarray  # frames (time) x channels
    fs: float  # sample rate (Hz)

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
    fixdir      : str
    ica         : ICA
    input       : Type[TimeCourse] = None
    timecourses : dict             = field(default_factory=dict)
    spatialmaps : dict             = field(default_factory=dict)
    hp_fwhm     : float            = None
    species     : str              = None
    input_tree  : FileTree         = None
    fix_tree    : FileTree         = None

    @property
    def nic(self) -> int:
        return self.ica.nic

    def __post_init__(self):

        for k, v in self.timecourses.items():
            assert v.srate == self.ica.srate, \
                f'srate of {k} timecourse ({v.srate}) must equal ICA ({self.ica.srate})'

            assert v.ntime == self.ica.ntime, \
                f'number time samples of {k} timecourse ({v.ntime}) must equal ICA ({self.ica.ntime})'

        if self.input is not None:
            assert self.input.ntime == self.ica.ntime, \
                f'number of time samples in input data ({self.input.ntime}) must equal ICA ({self.ica.ntime})'

    @classmethod
    def from_input_dir(cls,
                       indir         : io.InputDirectory,
                       highpass_fwhm : float = None):

        fixdir  = indir.fixdir
        intree  = indir.input_tree
        fixtree = indir.fix_tree
        species = indir.species
        indir   = indir.indir

        os.makedirs(fixdir, exist_ok=True)

        # melodic data

        func = nb.load(intree.get('filtered_func_data'))
        tr = func.header.get_zooms()[3]

        func_brainmask = nb.load(intree.get('mask'))

        ic_timeseries = np.loadtxt(intree.get('melodic_mix'))
        ic_map = nb.load(intree.get('melodic_IC'))

        struct_fs_wmparc = intree.get('wmparc')

        struct = nb.load(intree.get('highres'))
        struct2func_xfm = intree.get('highres2example_func_mat')

        veins_exf = intree.get('veins_exf')

        # structural segmentations - may
        # exist in input dir or in fix dir
        struct_fast_seg = intree.get('highres_pveseg')
        if not op.exists(struct_fast_seg):
            struct_fast_seg = fixtree.get('fastsg')

        if not op.exists(struct_fast_seg):
            log.info('Creating structural segmentation with FSL FAST')

            fast_prefix = fixtree.placeholders['fast_basename']
            fast_prefix = op.join(op.dirname(struct_fast_seg), fast_prefix)

            fast(struct.get_filename(), fast_prefix, t=1)

        dseg = nb.load(struct_fast_seg)

        # Load subcortical mask (if it exists) and add to fast pveseg
        if op.exists(struct_fs_wmparc):
            subcort = util.create_mask(
                dseg=struct_fs_wmparc,
                labels=(10, 11, 12, 13, 49, 50, 51, 52, 26, 58),
                outname=fixtree.get('subcort'))

            dseg = nb.load(struct_fast_seg)
            dseg_d = dseg.get_fdata()
            dseg_d[subcort.get_fdata().astype(bool)] = 2
            dseg = nb.Nifti1Image(
                dseg_d.astype(int), header=dseg.header, affine=dseg.affine)
            dseg.to_filename(struct_fast_seg)

        # standard-to-func transforms

        standard = intree.get('standard')
        std2func_xfm = None

        std2func_mat = intree.get('standard2example_func_mat')
        if op.exists(std2func_mat):
            std2func_xfm = std2func_mat

        std2func_warp = intree.get('standard2example_func_warp')
        if op.exists(std2func_warp):
            std2func_xfm = std2func_warp

        # highpass settings
        design = intree.get('design')
        if op.exists(design):
            fsf = featanalysis.loadFsf(design)
            if int(fsf['temphp_yn']) == 1:
                highpass_fwhm = float(fsf['paradigm_hp'])

        # motion parameters
        mp = intree.get('motion_params')
        if not op.exists(mp):
            mp = None

        # filter, and normalise motion confounds, as required
        if mp is not None:
            mp = clean.motion_confounds(
                mp, tr=tr, temporal_fwhm=highpass_fwhm)

        # instantiate class
        d = cls(
            fixdir=fixdir,
            ica=ICA(
                maps=NiftiMap(ic_map),
                timecourses=TimeCourse(ic_timeseries, fs=1 / tr),
            ),
            spatialmaps={
                'dseg': NiftiMap(dseg, xfm=struct2func_xfm),
                'brainmask': NiftiMap(func_brainmask),
                'struct': NiftiMap(struct, xfm=struct2func_xfm),
                # 'standard': NiftiMap(standard, xfm=std2func_xfm)
            },
            hp_fwhm=highpass_fwhm,
            input=NiftiTimeCourse(func, fs=1 / tr),
            species=species,
            input_tree=intree,
            fix_tree=fixtree
        )

        if op.exists(standard):
            d.spatialmaps['standard'] = NiftiMap(standard, xfm=std2func_xfm)

        # motion parameters
        if mp is not None:
            d.timecourses['motparams'] = TimeCourse(mp, fs=1 / tr)

        # veins
        if op.exists(veins_exf):
            d.spatialmaps['veins_exf'] = NiftiMap(nb.load(veins_exf))

        return d


def extract_features(data: FixData,
                     features: List[str]):
    log.info("Extract features")

    # setup fixdir

    if not op.exists(data.fixdir):
        os.makedirs(data.fixdir)

    # extract features
    rows = []
    for ic_idx in range(data.nic):
        log.debug(f'Extracting features for IC {ic_idx}')

        row0 = [feature.FEATURE_EXTRACTORS[fe](data, ic_idx) for fe in features]
        rows.append(pd.concat(row0, verify_integrity=True))

    features = pd.concat(rows, axis=1).T

    return features
