#!/usr/bin/env python
""" FIX (FMRIB's ICA-based Xnoiseifier).

Usage:
    fix.py extract --func=FUNC --struct2func=STRUCT2FUNC
                   --ic_timeseries=ICTIME --ic_map=ICMAP
                   --motparam=MOTPARAM (--struct=STRUCT | --struct_seg=SEG)
                   [--ic_threshold=THRESHOLD] [--hp=HP]
                   [--func_brain_mask=MASK] [--standard2func_tx=TX]
                   [--overwrite]
    fix.py extract --featdir=FEATDIR
    fix.py extract --dhcpdir=DHCPDIR
    fix.py train --output=DIRNAME [--loo] FIXDIR FIXDIR...
    fix.py classify --fixdir=DIR --classifier=CLF
    fix.py apply --func=FUNC --ic_timeseries=ICTIME --rmidx=IDX
                 [--motparam=MOTPARAM]

"""
import docopt
# import json
import logging
import os
import os.path as op
# import fsl.data.image as fslimg
import util
from util import elapsedtime
from enum import Enum, unique
import numpy as np
import shutil
import nibabel as nb
import feature
import scipy

log = logging.getLogger(__name__)


@unique
class SegType(Enum):
    """Enum for supported segmentation type."""
    fsl_fast = 1
    freesurfer_wmparc = 2


class Fix(object):

    def __init__(self, workdir: str) -> None:

        # setup workdir

        self.workdir = workdir

        # setup logging

        logging.basicConfig(filename=op.join(self.workdir, 'pyfix.log'),
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s:%(name)s:%(asctime)s:%(message)s')


    @property
    def workdir(self):
        return self.__workdir

    @workdir.setter
    def workdir(self, d):
        if not op.exists(d):
            os.makedirs(d)
        self.__workdir = d

    @elapsedtime
    def extract_features(
            self,
            func: str,     # native functional space
            struct2func_rigid: str,  # native structural to native functional space Tx
            ic_timeseries: str,  # make optional, run ICA
            ic_map: str,   # make optional, run ICA
            motparam: str,      # make optional, run mcflirt if not provided
            ic_threshold: float=2.5,  # magic number taken from original fix
            hp=0,
            func_brain_mask: str=None,  # native functional space
            struct: str=None,       # native structural space
            struct_seg: str=None,  # native structrual space
            struct_seg_type: SegType=SegType.fsl_fast,
            standard2func_tx: str=None,  # mni standard to native functional space Tx -> None
            overwrite: bool=True):  # overwrite existing features if true

        feature_fname = op.join(self.workdir, 'features.csv')

        if not overwrite:
            assert not op.exists(feature_fname), (
                'feature file already exists'
            )

        log.info("Extract features")

        # parse args

        util.assert_file_exists(func, struct2func_rigid, ic_timeseries, ic_map)

        func = nb.load(func)
        ic_timeseries = np.loadtxt(ic_timeseries)
        ic_map = nb.load(ic_map)

        if func_brain_mask is None:
            func_brain = op.join(self.workdir, 'func_brain')
            util.run(['bet', func, func_brain, '-m', '-n'])
            func_brain_mask = func_brain + '_mask.nii.gz'

        func_brain_mask = nb.load(func_brain_mask)

        assert any([struct, struct_seg]), (
            'Either a struct or struct_seg is required'
        )

        struct = None if struct is None else nb.load(struct)
        struct_seg = None if struct_seg is None else nb.load(struct_seg)

        func_mean = nb.Nifti1Image(np.mean(func.get_data(), axis=3),
                                   func.affine, func.header)
        func_mean.to_filename(op.join(self.workdir, 'func_mean.nii.gz'))

        # segment structural if struct_seg not supplied and
        # warp to native func space
        func_seg = self._apply_fast(
            struct, struct_seg_type, struct_seg, func, struct2func_rigid)

        # create edge masks
        edge_masks = self._create_edge_masks(func_brain_mask)

        # create sag masks
        sag_masks = self._create_sag_masks(func, standard2func_tx, struct,
                                           struct2func_rigid)

        # extract features

        pixdim = func.header.get_zooms()
        res = func.shape
        TR = pixdim[3]

        sigma = 0.5*hp/TR
        motparam = feature.motionconfounds(motparam, sigma, TR)
        np.savetxt(op.join(self.workdir, 'motparam_filtered_extended.par'), motparam)

        func0 = np.reshape(func.get_data(), (-1, res[-1]))
        seg0 = np.ravel(func_seg.get_data())

        csf_timeseries = np.mean(func0[seg0 == 1, :], axis=0)
        gm_timeseries = np.mean(func0[seg0 == 2, :], axis=0)
        wm_timeseries = np.mean(func0[seg0 == 3, :], axis=0)

        allrows = []

        nIC = ic_map.shape[3]

        for ic_idx in range(nIC):

            log.info("Extract features for IC {}".format(ic_idx))
            # print("Extract features for IC {}".format(ic_idx))

            ic_timeseries0 = ic_timeseries[:, ic_idx]

            ic_map0 = nb.Nifti1Image(ic_map.dataobj[:, :, :, ic_idx],
                                     ic_map.affine, ic_map.header)
            ic_map0.to_filename(op.join(self.workdir, 'ic_map0.nii.gz'))

            ic_map0_abs = nb.Nifti1Image(np.abs(ic_map.dataobj[:, :, :, ic_idx]),
                                         ic_map.affine, ic_map.header)
            ic_map0_abs.to_filename(op.join(self.workdir, 'ic_map0_abs.nii.gz'))

            row = [nIC]

            # temporal features

            row += feature.arvswgn(ic_timeseries0, TR, norder=6)
            row += feature.arfull(ic_timeseries0, TR, 1)
            row += feature.arfull(ic_timeseries0, TR, 2)
            row += [scipy.stats.skew(ic_timeseries0)]
            row += [scipy.stats.kurtosis(ic_timeseries0, fisher=False)]
            row += [np.mean(ic_timeseries0)-np.median(ic_timeseries0)]
            row += feature.entropy(ic_timeseries0)
            row += feature.tsjump(ic_timeseries0)
            for i in [0.1, 0.15, 0.2, 0.25]:
                row += feature.fftcoarse(ic_timeseries0, TR, i)
            thres = [.01, .025, .05, .1, .15, .2, .25]
            row += feature.fftfiner(ic_timeseries0, TR, thres)
            row += feature.fftfinerwrtnull(ic_timeseries0, TR, thres)
            row += feature.motioncorrelation(ic_timeseries0, motparam)
            row += list(np.abs(feature.oupjk(ic_timeseries0, TR)))

            # spatial features

            row += feature.clusterdist(ic_map0_abs, pixdim[0:-1], ic_threshold)
            row += feature.negativevspositive(ic_map0, ic_map0_abs, ic_threshold)
            row += feature.zstattofuncratio(ic_map0, func_mean)
            row += feature.slicewisestats(ic_map0_abs, ic_threshold)
            row += feature.everynthvarianve(ic_map0_abs, ic_threshold)
            row += feature.masktscorrandoverlap(ic_map0_abs, ic_timeseries0, func_seg, gm_timeseries, wm_timeseries, csf_timeseries, ic_threshold)
            row += feature.smoothest(ic_map0, func_brain_mask)
            row += [row[78] * pixdim[0] * pixdim[1] * pixdim[2]]
            row += feature.maxtfce(ic_map0, ic_threshold)
            row += feature.edgemasks(ic_map0, ic_map0_abs, edge_masks, ic_threshold)
            row += feature.sagmasks(ic_map0_abs, sag_masks, ic_threshold)
            row += feature.stripe(ic_map0)
            row += res
            row += pixdim

            allrows += [np.array(row)]

        np.savetxt(feature_fname, np.array(allrows),
                   fmt='%0.8f', delimiter=',')

        log.info("Finished extracting features")

    def _create_edge_masks(self, func_brain_mask: nb.Nifti1Image) -> None:
        log.info('Creating edge masks')

        # create edge masks in native functional space
        # from the func brain mask
        mask = func_brain_mask.get_filename()
        edge_masks = []
        for i in range(1, 6):

            ero_mask = op.join(self.workdir, 'maske{0:n}'.format(i))
            util.run([
                'fslmaths',  mask, '-eroF',  ero_mask
            ])

            edge_mask = op.join(self.workdir, 'edge{0:n}'.format(i))
            util.run([
                'fslmaths', mask, '-sub', ero_mask,  edge_mask
            ])

            edge_masks += [nb.load(edge_mask+'.nii.gz')]

        return edge_masks

    def _apply_fast(
            self,
            struct: nb.Nifti1Image,
            struct_seg_type,
            struct_seg: nb.Nifti1Image,
            func: nb.Nifti1Image,
            struct2func_rigid: str) -> None:

        # create structual segmentation if it does not exist
        if struct_seg is None:
            log.info('Creating structural segmentation with FSL FAST')
            util.run([
                'fast', '-t', '1', '-o',
                op.join(self.workdir, 'fastsg'),
                struct.get_filename()
            ])
            struct_seg = nb.load(op.join(self.workdir, 'fastsg_pveseg.nii.gz'))

        # TODO: add support for freesurfer wmparc
        assert struct_seg_type is SegType.fsl_fast, (
            'Segmentation type not currently supported')

        # transform structural segmentation to native functional space
        log.info('Transform structural segmentation to native functional space')
        func_seg = op.join(self.workdir, 'func_seg.nii.gz')
        util.run([
            'flirt',
            '-in', struct_seg.get_filename(),
            '-ref', func.get_filename(),
            '-applyxfm',
            '-init', struct2func_rigid,
            '-interp', 'nearestneighbour',
            '-out', func_seg
        ])
        func_seg = nb.load(func_seg)

        return func_seg

    def _create_sag_masks(
            self,
            func: nb.Nifti1Image,
            standard2func_tx: nb.Nifti1Image,
            struct: nb.Nifti1Image,
            struct2func_rigid: str) -> None:

        FIXDIR = os.getenv('FIXPYDIR', '')
        FSLDIR = os.getenv('FSLDIR', '')

        # create standard_to_func transform if it does not exist
        if standard2func_tx is None:
            log.info('Affine registration of native structural to standard (MNI152)')
            struct2std = op.join(self.workdir, 'struct2std.mat')
            util.run([
                'flirt',
                '-in', struct.get_filename(),
                '-ref', op.join(FSLDIR, 'data/standard/MNI152_T1_1mm_brain'),
                '-omat', struct2std])

            std2struct = op.join(self.workdir, 'std2struct.mat')
            util.run([
                'convert_xfm',
                '-omat', std2struct,
                '-inverse', struct2std])

            standard2func_tx = op.join(self.workdir, 'std2func.mat')
            util.run([
                'convert_xfm',
                '-omat', standard2func_tx,
                '-concat', struct2func_rigid, std2struct])

        # transform standard space sagittal masks into native functional space

        sag_masks = np.empty((4, 3), dtype=object)

        dlabel = ['', 'dil', 'dil2']

        for r in range(sag_masks.shape[0]):
            for c in range(sag_masks.shape[1]):

                in_name = op.join(FIXDIR, 'masks', 'sag_mask{}{}'.format(r, dlabel[c]))
                out_name = op.join(self.workdir, 'std1mm2exfunc{}{}'.format(r, dlabel[c]))

                # cmd = [
                #     'flirt',
                #     '-in', in_name,
                #     '-ref', self._func.get_filename(),
                #     '-applyxfm',
                #     '-init', self._standard2func_tx,
                #     '-out', out_name,
                #     '-datatype', 'float'
                # ]

                cmd = [
                    'applywarp',
                    '-i', in_name,
                    '-r', func.get_filename(),
                    '-o', out_name,
                    '--interp=trilinear']

                if standard2func_tx.endswith('.mat'):
                    cmd += ['--premat={}'.format(standard2func_tx)]
                elif standard2func_tx.endswith('.nii.gz'):
                    cmd += ['-w', standard2func_tx]
                else:
                    raise RuntimeError('Unknown transform type!')

                util.run(cmd)

                sag_masks[r, c] = nb.load(out_name + '.nii.gz')

        # TODO: deal with veins_exf

        return list(sag_masks)

    @classmethod
    def from_melodic_dir(clss, meldir: str):

        args = {}
        args['func'] = op.join(meldir, 'filtered_func_data.nii.gz')
        args['func_brain_mask'] = op.join(meldir, 'mask.nii.gz')
        args['struct'] = op.join(meldir, 'reg/highres.nii.gz')
        args['struct2func_rigid'] = op.join(
            meldir, 'reg/highres2example_func.mat')
        util.assert_file_exists(*(args.values()))

        struct_fast_seg = op.join(meldir, 'reg/highres_pveseg.nii.gz')
        struct_fs_wmparc = op.join(meldir, 'reg/wmparc.nii.gz')

        if op.exists(struct_fast_seg):
            args['struct_seg'] = struct_fast_seg
            args['struct_seg_type'] = SegType.fsl_fast

        if op.exists(struct_fs_wmparc):
            args['struct_seg'] = struct_fs_wmparc
            args['struct_seg_type'] = SegType.freesurfer_wmparc

        std2func_mat = op.join(meldir, 'reg/standard2example_func.mat')
        std2func_warp = op.join(meldir, 'reg/standard2example_func_warp.nii.gz')

        if op.exists(std2func_mat):
            args['standard2func_tx'] = std2func_mat

        if op.exists(std2func_warp):
            args['standard2func_tx'] = std2func_warp

        args['ic_timeseries'] = op.join(
            meldir, 'filtered_func_data.ica', 'melodic_mix')
        args['ic_map'] = op.join(
            meldir, 'filtered_func_data.ica', 'melodic_IC.nii.gz')

        args['motparam'] = op.join(meldir, 'mc/prefiltered_func_data_mcf.par')

        workdir = op.join(meldir, 'fix')

        d = clss(workdir)
        d.extract_features(**args)

        return d

    @classmethod
    def from_feat_dir(clss, featdir: str):
        return clss.from_melodic_dir(featdir)

    @classmethod
    def from_dhcp_dir(clss, dhcpdir: str):

        workdir = op.join(dhcpdir, 'fix')
        d = clss(workdir)

        args = {}
        args['func'] = op.join(dhcpdir, 'ica/func_filtered.nii.gz')
        args['func_brain_mask'] = op.join(dhcpdir, 'motioncor/func_mc_brain_mask.nii.gz')
        args['struct'] = op.join(dhcpdir, 'struct_T2w.nii.gz')
        args['struct2func_rigid'] = op.join(dhcpdir, 'reg/struct_T2w_to_func_mc_rigid.mat')
        args['struct_seg'] = op.join(dhcpdir, 'masks/fast_seg.nii.gz')
        args['ic_timeseries'] = op.join(dhcpdir, 'ica/melodic_mix')
        args['ic_map'] = op.join(dhcpdir, 'ica/melodic_IC.nii.gz')
        args['motparam'] = op.join(dhcpdir, 'motioncor/func_mc.par')

        util.assert_file_exists(*(args.values()))

        args['struct_seg_type'] = SegType.fsl_fast

        d.extract_features(**args)



if __name__ == '__main__':
    args = docopt(__doc__, version='fix.py 0.1.0a')

    if args['extract']:
        if args['--featdir'] is not None:
            d = Fix.from_feat_dir(args['--featdir'])
        elif args['--dhcpdir'] is not None:
            d = Fix.from_dhcp_dir(args['--dhcpdir'])
        else:
            print('extract from files')

    if args['train']:
        print('FIX train')

    if args['classify']:
        print('FIX classify')

    if args['apply']:
        print('FIX apply')
