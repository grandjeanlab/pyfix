#!/usr/bin/env python
"""Apply FIX clean to data."""

import nibabel as nb
import nibabel.gifti.giftiio as gio
import numpy as np
import os.path as op
import dhcp.util.util as util
import uuid
import os
import argparse


def norm(d, dim=0):
    d = d - np.mean(d, axis=dim)
    d = d / np.maximum(np.std(d, axis=dim), 0.00001)
    return d


def hpfilt(d, sigma, tr):
    Nt = d.shape[1]
    Nv = d.shape[0]

    dim = np.ceil(Nv/100.0).astype(int)
    d0 = np.zeros((100*dim, Nt))

    d0[0:Nv, :] = d

    d0 = d0.reshape((10, 10, dim, Nt), order="F")

    # TODO: should use a sensible tempname (and location)
    tmpname = str(uuid.uuid4())
    img = nb.Nifti1Image(d0, np.eye(4))
    nb.save(img, tmpname+".nii.gz")

    util.shrun("fslmodhd {0} pixdim4 {1}".format(tmpname+".nii.gz", tr))
    util.shrun("fslmaths {0} -Tmean {0}_mean".format(tmpname))
    util.shrun("fslmaths {0} -bptf {1} -1 -add {0}_mean {0}".format(tmpname,
                                                                    sigma))
    img = nb.load(tmpname+".nii.gz")
    d = img.get_data().reshape((100*dim, img.shape[-1]), order="F")

    os.remove(tmpname+".nii.gz")
    os.remove(tmpname+"_mean.nii.gz")

    return d[0:Nv, :]


def readgii(funcname):
    funcSrf = gio.read(funcname)
    Nt = len(funcSrf.darrays)
    Nv = funcSrf.darrays[0].data.shape[0]

    func = np.zeros((Nv, Nt))

    for i in range(Nt):
        func[0:Nv, i] = funcSrf.darrays[i].data

    return func, funcSrf


def writegii(fname, func, funcSrf):
        Nt = len(funcSrf.darrays)
        gii = nb.gifti.gifti.GiftiImage(meta=funcSrf.meta)
        for i in range(Nt):
            darray = nb.gifti.gifti.GiftiDataArray()
            darray = darray.from_array(func[:, i].astype(np.float32),
                                       'NIFTI_INTENT_TIME_SERIES')
            gii.add_gifti_data_array(darray)

        gio.write(gii, fname)


def main(funcname, icamix, DDremove, sigma=None, motparams=None,
         filtfunc=False, filtmotparams=True, outname=None):

    isgifti = funcname.endswith('.gii')
    isnifti = funcname.endswith('.nii') or funcname.endswith('.nii.gz')

    # read funcname
    if isgifti:
        # TODO: add and test gifti support
        # func, funcSrf = readgii(funcname)
        raise RuntimeError("gifti is not currently supported")
    elif isnifti:
        funcnii = nb.load(funcname)
        func = funcnii.get_data()
        TR = funcnii.header['pixdim'][4]
        dims = func.shape
        func = np.reshape(func, (-1, dims[3]), order='F')
    else:
        raise RuntimeError("Invalid filetype: {0}".format(funcname))

    # filter the func (if req'd)
    if sigma is not None and filtfunc:
        func = hpfilt(func, sigma, TR)

    # read and normalise ICs (i.e. melodic_mix)
    ICA = norm(np.loadtxt(icamix))

    # read and filter motion confounds (if req'd)
    if motparams is not None:
        conf = np.loadtxt(motparams)
        conf = conf[:, 0:6]
        conf0 = np.vstack((np.zeros((1, conf.shape[1])),
                          np.diff(conf, axis=0)))
        conf0 = np.hstack((conf, conf0))
        conf0 = norm(conf0)
        conf0 = np.hstack((conf0, conf0*conf0))
        conf0 = norm(conf0)

        if sigma is not None and filtmotparams:
            conf0 = hpfilt(conf0.T, sigma, TR).T
        conf = norm(conf0)

        ICA = ICA - np.dot(conf, np.dot(np.linalg.pinv(conf, 1e-6), ICA))
        func = func - np.dot(conf, np.dot(np.linalg.pinv(conf, 1e-6),
                             func.T)).T

    DDremove -= 1
    # beta for ICA (good *and* bad)
    betaICA = np.dot(np.linalg.pinv(ICA, 1e-6), func.T)
    func = func - np.dot(ICA[:, DDremove], betaICA[DDremove, :]).T

    # write to file

    if isgifti:
        # if outname is None:
        #     outname = funcname.rstrip(".gii") + "_clean_py.gii"
        # writegii(outname, func, funcSrf)
        print("write gifti")
    elif isnifti:
        func = np.reshape(func, dims, order='F')
        img = nb.Nifti1Image(func, funcnii.affine, funcnii.header)
        if outname is None:
            outname = funcname.rstrip(".nii.gz") + "_clean_py.nii.gz"
        nb.save(img, outname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply FIX cleanup")

    parser.add_argument('--motparams', default=None,
                        help="Name of motion parameter file")
    parser.add_argument('--outname', default=None,
                        help="Name for cleaned func")
    parser.add_argument('--sigma', default=None,
                        help="Sigma (in volumes) for high-pass filter")

    parser.add_argument('--filt-func', dest='filtfunc', action='store_true')
    parser.add_argument('--no-filt-func', dest='filtfunc',
                        action='store_false')
    parser.set_defaults(filtfunc=False)

    parser.add_argument('--filt-motparams', dest='filtmotparams',
                        action='store_true')
    parser.add_argument('--no-filt-motparams', dest='filtmotparams',
                        action='store_false')
    parser.set_defaults(filtmotparams=True)

    reqd = parser.add_argument_group('required arguments')
    reqd.add_argument('--func', help='Func file name', required=True)
    reqd.add_argument('--mix', help='Melodic mixing matric (melodic_mix)',
                      required=True)
    reqd.add_argument('--classification', help='Fix classification file',
                      required=True)

    args = parser.parse_args()

    cmd = 'tail -n 1 ' + args.classification
    DDremove = util.shrun(cmd).lstrip('[').rstrip(']').replace(" ", "")
    DDremove = np.fromstring(DDremove, dtype=int, sep=",")

    main(args.func, args.mix, DDremove, sigma=args.sigma,
         motparams=args.motparams, filtfunc=args.filtfunc,
         filtmotparams=args.filtmotparams, outname=args.outname)
