#!/usr/bin/env python
"""Regress FIX labelled ICs from data."""

import argparse
import os
import sys
import uuid
from subprocess import check_output

import nibabel as nb
import nibabel.gifti.giftiio as gio
import numpy as np


def _norm(d, dim=0):
    d = d - np.mean(d, axis=dim)
    d = d / np.maximum(np.std(d, axis=dim), 0.00001)
    return d


def run(cmd):
    if type(cmd) is list:
        _str = " ".join(cmd)
    else:
        _str = cmd
        cmd = _str.split()

    sys.stdout.flush()
    jobout = check_output(cmd)
    return jobout.decode('utf-8').strip()


def _hpfilt(func, sigma, tr):
    n_t = func.shape[1]
    n_v = func.shape[0]

    dim = np.ceil(n_v / 100.0).astype(int)
    func_0 = np.zeros((100 * dim, n_t))

    func_0[0:n_v, :] = func

    func_0 = func_0.reshape((10, 10, dim, n_t), order="F")

    # TODO: should use a sensible tempname (and location)
    tmpname = str(uuid.uuid4())
    img = nb.Nifti1Image(func_0, np.eye(4))
    nb.save(img, tmpname + ".nii.gz")

    run("fslmodhd {0} pixdim4 {1}".format(tmpname + ".nii.gz", tr))
    run("fslmaths {0} -Tmean {0}_mean".format(tmpname))
    run("fslmaths {0} -bptf {1} -1 -add {0}_mean {0}".format(tmpname,
                                                             sigma))
    img = nb.load(tmpname + ".nii.gz")
    func = img.get_data().reshape((100 * dim, img.shape[-1]), order="F")

    os.remove(tmpname + ".nii.gz")
    os.remove(tmpname + "_mean.nii.gz")

    return func[0:n_v, :]


def _readgii(funcname):
    func_srf = gio.read(funcname)
    n_t = len(func_srf.darrays)
    n_v = func_srf.darrays[0].data.shape[0]

    func = np.zeros((n_v, n_t))

    for i in range(n_t):
        func[0:n_v, i] = func_srf.darrays[i].data

    return func, func_srf


def _writegii(fname, func, funcSrf):
    n_t = len(funcSrf.darrays)
    gii = nb.gifti.gifti.GiftiImage(meta=funcSrf.meta)
    for i in range(n_t):
        darray = nb.gifti.gifti.GiftiDataArray()
        darray = darray.from_array(func[:, i].astype(np.float32),
                                   'NIFTI_INTENT_TIME_SERIES')
        gii.add_gifti_data_array(darray)

    gio.write(gii, fname)


def apply(funcname,
          icamix,
          ddremove,
          temporal_fwhm=None,
          motparams=None,
          filt_func=False,
          filt_motparams=True,
          func_tr=None,
          outname=None):

    is_gifti = funcname.endswith('.gii')
    is_nifti = funcname.endswith('.nii') or funcname.endswith('.nii.gz')

    # read funcname
    if is_gifti:
        func, func_srf = _readgii(funcname)
        if func_tr is None:
            raise RuntimeError("TR argument is required when func is a giftii")
    elif is_nifti:
        func_nii = nb.load(funcname)
        func = func_nii.get_data()
        func_tr = func_nii.header['pixdim'][4]
        dims = func.shape
        func = np.reshape(func, (-1, dims[3]), order='F')
    else:
        raise RuntimeError("Invalid filetype: {0}".format(funcname))

    # convert temporal_fwhm (s) to sigma (vols)
    if temporal_fwhm is not None:
        if func_tr is None:
            raise RuntimeError('TR is required if temporal_fwhm is provided.')
        sigma = np.round(temporal_fwhm / (2 * func_tr))

    # filter the func (if req'd)
    if sigma is not None and filt_func:
        func = _hpfilt(func, sigma, func_tr)

    # read and normalise ICs (i.e. melodic_mix)
    ica = _norm(np.loadtxt(icamix))

    # read and filter motion confounds (if req'd)
    if motparams is not None:
        conf = np.loadtxt(motparams)
        conf = conf[:, 0:6]
        conf0 = np.vstack((np.zeros((1, conf.shape[1])),
                           np.diff(conf, axis=0)))
        conf0 = np.hstack((conf, conf0))
        conf0 = _norm(conf0)
        conf0 = np.hstack((conf0, conf0 * conf0))
        conf0 = _norm(conf0)

        if sigma is not None and filt_motparams:
            conf0 = _hpfilt(conf0.T, sigma, func_tr).T
        conf = _norm(conf0)

        ica = ica - np.dot(conf, np.dot(np.linalg.pinv(conf, 1e-6), ica))
        func = func - np.dot(conf, np.dot(np.linalg.pinv(conf, 1e-6),
                                          func.T)).T

    ddremove -= 1
    # beta for ICA (good *and* bad)
    beta_ica = np.dot(np.linalg.pinv(ica, 1e-6), func.T)
    func = func - np.dot(ica[:, ddremove], beta_ica[ddremove, :]).T

    # write to file

    if is_gifti:
        if outname is None:
            outname = funcname.rstrip(".func.gii") + "_clean_pyfix.func.gii"
        _writegii(outname, func, func_srf)
        # print("write gifti")
    elif is_nifti:
        func = np.reshape(func, dims, order='F')
        img = nb.Nifti1Image(func, func_nii.affine, func_nii.header)
        if outname is None:
            outname = funcname.rstrip(".nii.gz") + "_clean_pyfix.nii.gz"
        nb.save(img, outname)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Apply FIX cleanup")

    PARSER.add_argument('--motparams', default=None,
                        help="Name of motion parameter file")
    PARSER.add_argument('--outname', default=None,
                        help="Name for cleaned func")
    PARSER.add_argument('--sigma', default=None,
                        help="Sigma (in volumes) for high-pass filter")
    PARSER.add_argument('--TR', default=None,
                        help="TR (required for giftii)")

    PARSER.add_argument('--filt-func', dest='filtfunc', action='store_true')
    PARSER.add_argument('--no-filt-func', dest='filtfunc',
                        action='store_false')
    PARSER.set_defaults(filtfunc=False)

    PARSER.add_argument('--filt-motparams', dest='filtmotparams',
                        action='store_true')
    PARSER.add_argument('--no-filt-motparams', dest='filtmotparams',
                        action='store_false')
    PARSER.set_defaults(filtmotparams=True)

    REQD = PARSER.add_argument_group('required arguments')
    REQD.add_argument('--func', help='Func file name', required=True)
    REQD.add_argument('--mix', help='Melodic mixing matric (melodic_mix)',
                      required=True)
    REQD.add_argument('--classification', help='Fix classification file',
                      required=True)

    ARGS = PARSER.parse_args()

    # CMD = 'tail -n 1 ' + ARGS.classification
    # DDREMOVE = util.shrun(CMD).lstrip('[').rstrip(']').replace(" ", "")
    # DDREMOVE = np.fromstring(DDREMOVE, dtype=int, sep=",")

    with open(ARGS.classification) as F:
        DDREMOVE = F.read().splitlines()
    DDREMOVE = DDREMOVE[-1].lstrip('[').rstrip(']').replace(" ", "")
    DDREMOVE = np.fromstring(DDREMOVE, dtype=int, sep=",")

    apply(ARGS.func, ARGS.mix, DDREMOVE, sigma=ARGS.sigma,
          motparams=ARGS.motparams, filt_func=ARGS.filtfunc,
          filt_motparams=ARGS.filtmotparams, outname=ARGS.outname,
          func_tr=ARGS.TR)
