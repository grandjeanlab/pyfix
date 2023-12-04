#!/usr/bin/env python
"""Regress FIX labelled ICs from data."""

import os.path as op
import nibabel as nb
import numpy as np
from numpy.linalg import pinv
import scipy
import logging
from tempfile import TemporaryDirectory

from fsl.wrappers import fslmaths

from pyfix.io import load_motion_parameters

log = logging.getLogger(__name__)


def create_dummy_nifti(d, tr, fname=None):
    """
    Save data matrix [time x space] to dummy nifti1

    Args:
        d:
        tr:
        fname:

    Returns:

    """

    d = d.T

    n_v = d.shape[0]
    n_t = d.shape[1]

    if n_v < 32767:
        d = d.reshape((n_v, 1, 1, n_t), order='F')
    else:
        dim = np.ceil(n_v / 100.0).astype(int)
        d0 = np.zeros((100 * dim, n_t))

        d0[0:n_v, :] = d
        d = d0.reshape((10, 10, dim, n_t), order='F')

    img = nb.Nifti1Image(d, np.eye(4))
    img.header.set_zooms([1, 1, 1, tr])

    if fname is not None:
        img.to_filename(fname)

    return img, n_v


def _revert_dummy_nifti(fname, n=None):
    d = nb.load(fname)
    d = d.get_fdata().reshape((-1, d.shape[-1]), order='F')

    if n is not None:
        d = d[:n, :]

    return d.T


def _normalise(d, dim=0):
    d0 = d - np.mean(d, axis=dim)
    d0 = d0 / np.maximum(np.std(d0, axis=dim, ddof=1), 0.00001)
    return d0


# TODO: add support for image input/output (maybe....)
def highpass_filter(func, temporal_fwhm, tr, tmpdir=None, retain_mean=False):
    """
    Highpass filter data using fslmaths bptf
    Expects func [time x space]
    Do linear detrend if sigma==0

    """

    # convert temporal_fwhm (s) to sigma (vols)
    if temporal_fwhm is not None:
        sigma = temporal_fwhm / (2 * tr)

    log.info(f'Highpass filter: fwhm={temporal_fwhm}s, sigma={sigma} TRs')

    # if sigma == 0, do a linear detrend and return
    if sigma == 0:
        log.info('Applying linear detrend')
        return scipy.signal.detrend(func, axis=0, type='linear')

    # else, use FSLMATHS bptf for the highpass

    log.info(f'Applying highpass filter: sigma={sigma}')

    with TemporaryDirectory(dir=tmpdir) as tmp:

        dummy = op.join(tmp, 'dummy.nii.gz')
        mean = op.join(tmp, 'dummy_mean.nii.gz')

        _, N = create_dummy_nifti(func, tr=tr, fname=dummy)

        fslmaths(dummy).Tmean().run(mean)
        fslmaths(dummy).bptf(sigma, -1).run(dummy)

        if retain_mean:
            fslmaths(dummy).add(mean).run(dummy)

        func = _revert_dummy_nifti(dummy, n=N)

    return func


def _writegii(fname, func, funcSrf):
    n_t = len(funcSrf.darrays)
    gii = nb.gifti.gifti.GiftiImage(meta=funcSrf.meta)
    for i in range(n_t):
        darray = nb.gifti.gifti.GiftiDataArray()
        darray = darray.from_array(func[:, i].astype(np.float32),
                                   'NIFTI_INTENT_TIME_SERIES')
        gii.add_gifti_data_array(darray)

    nb.save(gii, fname)


def get_tr(d, tr=None):
    '''
    Get TR from image object.
    Supports nifti, gifti, and cifti.
    '''

    if isinstance(d, str):
        d = nb.load(d)

    if isinstance(d, nb.Cifti2Image):
        tr = d.header.get_axis(0).step
    elif isinstance(d, (nb.Nifti1Image, nb.Nifti2Image)):
        tr = d.header.get_zooms()[-1]
    elif isinstance(d, nb.gifti.gifti.GiftiImage):
        if tr is None:
            raise RuntimeError("TR argument is required when func is a giftii")
        tr = tr
    else:
        raise RuntimeError(f'unknown dtype {type(d)}')

    return tr


def get_fdata(d):
    '''
    Load d and return data as matrix [time x space].
    Constant columns space over time (columns) will be removed.
    Supports nifti, gifti, and cifti.
    '''

    if isinstance(d, nb.Cifti2Image):

        log.info(f'Image type: CIFTI')
        d0 = d.get_fdata(caching='unchanged')

    elif isinstance(d, (nb.Nifti1Image, nb.Nifti2Image)):

        log.info(f'Image type: NIFTI')
        log.info(f'NIFTI input shape: {d.shape}')
        d0 = d.get_fdata(caching='unchanged').reshape((-1, d.shape[-1])).T

    elif isinstance(d, nb.gifti.gifti.GiftiImage):

        log.info(f'Image type: GIFTI')
        n_t = len(d.darrays)
        n_v = d.darrays[0].data.shape[0]
        d0 = np.zeros((n_t, n_v))
        for i in range(n_t):
            d0[i, 0:n_v] = d.darrays[i].data

    else:
        raise RuntimeError(f'unknown dtype {type(d)}')

    log.debug(f'Data (pre-mask) shape: {d0.shape}')

    mask = np.ptp(d0, axis=0) > 0
    log.debug(f'Mask shape: {mask.shape}')

    d0 = d0[:, mask]

    log.info(f'Data shape: {d0.shape}')

    return d0, mask


def save_image(d, d0, mask, fname):
    """
    Save image matrix (d0) [time x space] to disk.
    Image format specfied by image object d.
    Constant columns space over time (columns) will be replaced (defined by mask).
    Supports nifti, gifti, and cifti.
    """

    # replace const columns (defined by mask)
    d0_full = np.zeros((d0.shape[0], mask.shape[0]))
    d0_full[:, mask] = d0

    if isinstance(d, nb.Cifti2Image):

        log.info(f'Output file type: CIFTI')

        bm_full = d.header.get_axis(1)

        if d0.shape[0] > 1:
            tr = get_tr(d)
            ax0 = nb.cifti2.SeriesAxis(start=0, step=tr, size=d0.shape[0])
            if not fname.endswith('.dtseries.nii'):
                fname += '.dtseries.nii'
        else:
            ax0 = nb.cifti2.ScalarAxis(name=[None])
            if not fname.endswith('.dscalar.nii'):
                fname += '.dscalar.nii'

        header = nb.cifti2.Cifti2Header.from_axes((ax0, bm_full))
        nb.Cifti2Image(d0_full, header=header).to_filename(fname)

    elif isinstance(d, (nb.Nifti1Image, nb.Nifti2Image)):

        log.info(f'Output file type: NIFTI')
        d0_full = np.reshape(d0_full.T, d.shape[:-1] + (d0.shape[0],))

        if d0_full.shape[3] == 1:
            d0_full = np.squeeze(d0_full, axis=3)

        log.info(f'NIFTI output shape: {d0_full.shape}')
        img = nb.Nifti1Image(d0_full, d.affine, d.header)
        nb.save(img, fname)

    elif isinstance(d, nb.gifti.gifti.GiftiImage):

        # TODO: add save to GIFTI
        raise NotImplementedError('Save to GIFTI not yet implemented')

    else:
        raise RuntimeError(f'unknown dtype {type(d)}')


def expand_motion_parameters(motion_parameters):
    """
    Expand motion parameters.
    The standard motion parameters (6), plus their temporal derivatives (another 6), plus
    squares of the above (another 12)

    Args:
        motion_parameters: three rotations and three translations

    Returns:
        expanded_motion_parameters
    """

    # expand motion parameters to 24
    conf = motion_parameters[['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']].values

    # add temporal derivatives
    conf0 = np.vstack((np.zeros((1, conf.shape[1])), np.diff(conf, axis=0)))
    conf0 = np.hstack((conf, conf0))
    conf0 = _normalise(conf0)

    # add squares
    conf0 = np.hstack((conf0, conf0 * conf0))
    conf0 = _normalise(conf0)

    return conf0


def motion_confounds(motion_parameters, tr, temporal_fwhm=None, tmpdir=None):
    if motion_parameters.endswith('_mcf_conf_hp.nii.gz'):

        # if motion parameters are legacy FIX (i.e. already expanded and filtered)
        # then load and return

        log.info('Looks like legacy FIX expanded and filtered motion parameter file (*_mcf_conf_hp.nii.gz)')
        conf0 = np.squeeze(nb.load(motion_parameters).get_fdata()).T

    elif motion_parameters.endswith('_mcf_conf.nii.gz'):

        # elif motion parameters are legacy FIX (i.e. already expanded but NOT filtered)
        # then load, filter (if req'd), and return

        log.info('Looks like legacy FIX expanded but NOT filtered motion parameter file (*_mcf_conf_hp.nii.gz)')
        conf0 = np.squeeze(nb.load(motion_parameters).get_fdata()).T

        if temporal_fwhm is not None:
            conf0 = highpass_filter(conf0, temporal_fwhm, tr, tmpdir=tmpdir)

    else:

        # else, load, expand, filter (if req'd), and return

        mp = load_motion_parameters(motion_parameters)
        conf0 = expand_motion_parameters(mp)

        # filter motion parameters if required
        if temporal_fwhm is not None:
            conf0 = highpass_filter(conf0, temporal_fwhm, tr, tmpdir=tmpdir)

    return conf0


# TODO: move varnorm and motparams_clean to separate functions outside apply
# TODO: remove filtering options from apply (run separately as required)
# TODO: create separate regress function with suppport for agressive and non-aggressive
def apply(func_fname,
          icamix,
          noise_idx,
          func_clean_fname,
          do_motion_regression=True,
          motion_parameters=None,
          func_tr=None,
          aggressive=False,
          varnorm_fname=None,
          motparams_clean_fname=None
          ):
    log.debug(locals())

    # if (do_filter_func or do_filter_motparams) and (temporal_fwhm is None):
    #     raise RuntimeError('temporal_fwhm required to filter func_d or motparams')

    if do_motion_regression and (motion_parameters is None):
        raise RuntimeError('motion parameters are required if do_motion_regression=True')

    if (motparams_clean_fname is not None) and (motion_parameters is None):
        raise RuntimeError('motion_parameters required to calculate cleaned-motion-parameters')

    # load funcname
    func = nb.load(func_fname)

    # load func data as [time x space] (irrespective of file format)
    func_d, func_mask = get_fdata(func)
    func_tr = get_tr(func, func_tr)

    # read and normalise ICs (i.e. melodic_mix)
    log.info(f'Load melodic_mix: {icamix}')
    ica = _normalise(np.loadtxt(icamix))

    log.debug(f'ICA shape: {ica.shape}')

    # convert noise_idx to boolean vector (for convenience)
    tmp = np.full(ica.shape[-1], False)
    tmp[noise_idx - 1] = True
    noise_idx = tmp
    log.debug(f'noise_idx shape: {noise_idx.shape}')

    # read and filter motion confounds (if req'func_d)
    if do_motion_regression:
        conf0 = motion_confounds(
            motion_parameters=motion_parameters,
            tr=func_tr,
            temporal_fwhm=None,
            tmpdir=op.dirname(func_fname),
        )
        conf = _normalise(conf0)

    # beta for ICA (good *and* bad)

    if aggressive:
        # TODO: properly test aggressive cleanup and remove this exception
        raise NotImplementedError('Aggressive cleanup not yet implemented')

        # if do_motion_regression:
        #     log.info(f'Concat motion parameters with ICA')
        #     conf0 = np.concatenate((conf0, ica[:, ddremove]), axis=1)
        #     log.info(f'Regress noise ICA components and motion parameters from functional data (aggressive)')
        # else:
        #     conf0 = ica
        #     log.info(f'Regress noise ICA components from functional data (aggressive)')
        #
        # func_d = func_d - (conf0 @ (np.linalg.pinv(conf0, 1e-6) @ func_d))  # cleanup

    else:

        if do_motion_regression:
            #  regress out motion parameters from ICA and data (aggressively)
            log.info(f'Regress motion parameters from ICA and functional data')

            ica = ica - (conf @ (pinv(conf, 1e-6) @ ica))
            func_d = func_d - (conf @ (pinv(conf, 1e-6) @ func_d))

        log.info(f'Regress noise ICA components from functional data (non-aggressive)')

        beta_ica = pinv(ica, 1e-6) @ func_d  # beta for ICA (good *and* bad)
        func_d = func_d - (ica[:, noise_idx] @ beta_ica[noise_idx, :])  # cleanup

    # write to file

    log.info(f'Save cleaned functional data to file: {func_clean_fname}')
    save_image(func, func_d, mask=func_mask, fname=func_clean_fname)

    # compute variance normalization field and save to file

    if varnorm_fname is not None:
        log.info('Compute variance normalisation field')

        signal_idx = ~noise_idx
        beta_ica = pinv(ica[:, signal_idx], 1e-6) @ func_d  # beta for ICA (good, bad already removed)
        func_d = func_d - (
                    ica[:, signal_idx] @ beta_ica)  # remove signal components to compute unstructured noise timeseries
        func_d = np.std(func_d, ddof=1, axis=0, keepdims=True)

        save_image(func, func_d, mask=func_mask, fname=varnorm_fname)


    # compute movement regressors with noise components removed

    if (not do_motion_regression) and (motparams_clean_fname is not None) and (motion_parameters is not None):

        log.info('Compute movement regressors with noise components removed')

        conf0 = motion_confounds(
            motion_parameters=motion_parameters,
            tr=func_tr,
            temporal_fwhm=None,
            tmpdir=op.dirname(func_fname),
        )
        conf = _normalise(conf0)

        if aggressive:
            # aggressively regress out noise ICA components from movement regressors
            beta_confounds = pinv(ica[:, noise_idx], 1e-6) @ conf  # beta for confounds (bad only)
            conf = conf - (ica[:, noise_idx] @ beta_confounds)  # cleanup
        else:
            # non-aggressively regress out noise ICA components from movement regressors
            beta_confounds = pinv(ica, 1e-6) @ conf  # beta for confounds (good *and* bad)
            conf = conf - (ica[:, noise_idx] @ beta_confounds[noise_idx, :])  # cleanup

        create_dummy_nifti(conf.astype(np.float32), tr=func_tr, fname=motparams_clean_fname)
