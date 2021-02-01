#!/usr/bin/env python
import io
import os
import os.path as op
import uuid

import nibabel as nb
import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model
from spectrum import aryule
import logging
from  pyfix import util, legacy
from pyfix.util import SegType
import functools
import itertools
from tempfile import TemporaryDirectory
from functools import lru_cache
from fsl.wrappers import bet, fslmaths, flirt, applywarp, applyxfm, concatxfm, invxfm, convertwarp
from fsl.utils.filetree import FileTree
from typing import List
from cachetools import cached
from cachetools.keys import hashkey


DEFAULTS = FileTree.read(util.get_resource('pyfix.tree'), partial_fill=False)

# TODO: add typing
# TODO: rationalise naming and document
# TODO: optimise
# TODO: tidy code.  much of this is ported from legacy-fix matlab code and is very untidy/inefficient

log = logging.getLogger(__name__)

LEGACY_FEATURE_LABELS = legacy.LEGACY_FEATURE_LABELS

FEATURE_EXTRACTORS = {}

def feature_extractor(func):

    name = func.__name__

    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)

        if np.isscalar(output):
            output = pd.Series({name: output})
        elif isinstance(output, list):
            output = pd.Series({f'{name}:{i}': output[i] for i in range(len(output))})
        elif isinstance(output, dict):
            output = pd.Series(output)
        elif isinstance(output, pd.Series):
            pass
        else:
            raise RuntimeError(f'Unsupported output type ({type(output)}) for feature extractor ({name})')

        return output

    log.debug('Registering function: {name}', name)
    FEATURE_EXTRACTORS[name] = wrapper

    return wrapper


# def read_legacy_features(fname):
#     return pd.read_csv(fname, header=None, names=LEGACY_FEATURE_LABELS)
#
#
# def save_legacy_features(d):
#     d[LEGACY_FEATURE_LABELS].to_csv(index=None)


@feature_extractor
def nic(data, ic_idx):
    return data.nic


@feature_extractor
def dim(data, ic_idx):
    return list(data.input.data.shape)


@feature_extractor
def pixdim(data, ic_idx):
    return list(data.input.data.header.get_zooms())


@feature_extractor
def arvswgn(data, ic_idx, norder=6):
    ts = data.ica.get_timecourse(ic_idx)
    mvec = np.zeros(norder)
    for i in range(norder):
        mvec[i] = aryule(ts, i + 1)[1]
    return list(np.polyfit(np.arange(norder), mvec, 1))


@feature_extractor
def arfull(data, ic_idx, norder=(1, 2)):

    ts = data.ica.get_timecourse(ic_idx)

    def _arfull(ts, norder=1):

        AR, V, k = aryule(ts, norder)
        return list(np.concatenate(([V], AR)))

    output = [_arfull(ts, n) for n in norder]
    return list(itertools.chain.from_iterable(output))


@feature_extractor
def entropy(data, ic_idx):

    ts = data.ica.get_timecourse(ic_idx)

    bins = np.linspace(-2, 2, int(np.floor(np.sqrt(ts.shape[0]))))
    X = np.histogram(ts, bins=bins)[0]
    X = X / np.sum(X)
    X = -np.sum(X * np.log(X + np.finfo(float).eps))

    S = ts / (np.std(ts) + np.finfo(float).eps)
    S = np.sum((np.mean(-np.exp(-S ** 2 / 2)) + 0.71) ** 2)

    return [X, S]


@feature_extractor
def tsjump(data, ic_idx):

    ts = data.ica.get_timecourse(ic_idx)

    # Some jump characteristics of the time series
    diff_ts = np.diff(ts)

    eps = np.finfo(float).eps

    out = []
    out += (np.max(np.abs(diff_ts / (eps + np.std(ts)))),)
    out += (np.max(np.abs(diff_ts / (eps + np.std(diff_ts)))),)
    out += (np.sum((diff_ts / (eps + np.std(ts))) ** 2) / ts.shape[0],)

    # from Russell Poldrack's 08; Feature 5

    mx = np.amax(np.abs(diff_ts))
    imx = np.argmax(np.abs(diff_ts))

    sel = np.arange(imx - 2, imx + 2)

    x1 = sel >= 0
    x2 = sel < diff_ts.shape[0]
    sel = sel[x1 & x2]

    diff_ts = np.delete(diff_ts, sel)

    out += (mx / (eps + np.mean(np.abs(diff_ts))),)
    out += (mx / (eps + np.sum(np.abs(diff_ts))),)

    # from Russell Poldrack's 08; Feature 6
    ts = ts - np.mean(ts)
    out += (np.corrcoef([ts[:-1], ts[1:]])[0, 1],)

    return out


@feature_extractor
def skewness(data, ic_idx):
    return scipy.stats.skew(data.ica.get_timecourse(ic_idx))


@feature_extractor
def kurtosis(data, ic_idx):
    return scipy.stats.kurtosis(data.ica.get_timecourse(ic_idx), fisher=False)


@feature_extractor
def mean_median(data, ic_idx):
    ts = data.ica.get_timecourse(ic_idx)
    return np.mean(ts) - np.median(ts)


def nextpow2(x):
    return np.ceil(np.log2(x)).astype('int')


@feature_extractor
def fftcoarse(data, ic_idx, freq_threshold=[0.1, 0.15, 0.2, 0.25]):

    def _fftcoarse(ts, TR, freq_threshold):
        # In this function, given the time series and its sampling rate (e.g,
        # TR), we calculate the ratio of its high-frequency energy to low-frequency
        # energy

        L = ts.shape[0]
        nFFT = 2 ** nextpow2(L)

        Y = np.abs(np.fft.fft(ts, nFFT) / L)
        f = ((1 / TR) / 2) * np.linspace(0, 1, int(nFFT / 2))
        Y = Y[0:f.shape[0]]

        eps = np.finfo(float).eps

        return [np.sum(Y[f >= freq_threshold]) / (eps + np.sum(Y[f < freq_threshold]))]

    ts = data.ica.get_timecourse(ic_idx)
    TR = data.ica.tr

    output = [_fftcoarse(ts, TR, f) for f in freq_threshold]
    return list(itertools.chain.from_iterable(output))


@feature_extractor
def fftfiner(data, ic_idx, thres=[.01, .025, .05, .1, .15, .2, .25]):
    # In this function, given the time series and its sampling rate (e.g,
    # TR), we calculate the ratio of its high-frequency energy to low-frequency
    # energy

    ts = data.ica.get_timecourse(ic_idx)
    TR = data.ica.tr

    L = ts.shape[0]  # Length of signal
    nFFT = 2 ** nextpow2(L)  # Next power of 2 from length of S

    Y = np.abs(np.fft.fft(ts, nFFT) / L)
    f = ((1 / TR) / 2) * np.linspace(0, 1, int(nFFT / 2))
    Y = Y[0:f.shape[0]]

    out = np.zeros(len(thres))

    for i in range(len(thres) - 1):
        x1 = f > thres[i]
        x2 = f <= thres[i + 1]
        out[i] = np.sum(Y[x1 & x2])

    out[i + 1] = np.sum(Y[f > thres[i + 1]])

    eps = np.finfo(float).eps
    out = out / (eps + np.sum(out))

    return list(out)


@feature_extractor
def fftfinerwrtnull(data, ic_idx, thres=[.01, .025, .05, .1, .15, .2, .25]):
    # Concolve single-gamma HRF w/WGN and compare its spectrum w/actual RSN
    # do it for real

    ts = data.ica.get_timecourse(ic_idx)
    TR = data.ica.tr

    L = ts.shape[0]  # Length of signal
    nFFT = 2 ** nextpow2(L)  # Next power of 2 from length of S

    Y = np.abs(np.fft.fft(ts, nFFT) / L)
    f = ((1 / TR) / 2) * np.linspace(0, 1, int(nFFT / 2))
    Y = Y[:f.shape[0]]

    out = np.zeros(len(thres))

    for i in range(len(thres) - 1):
        x1 = f > thres[i]
        x2 = f <= thres[i + 1]
        out[i] = np.sum(Y[x1 & x2])

    out[i + 1] = np.sum(Y[f > thres[i + 1]])

    featuresReal = out.copy()
    # do it under the H0
    delay = 6 / TR
    sigma = delay / 2

    hrf = scipy.stats.gamma.pdf(np.arange(0, ts.shape[0]), (delay / sigma) ** 2, scale=delay / (sigma ** 2))

    features_null = np.zeros(len(thres))

    for i in np.arange(100):
        y = np.random.randn(L)
        y = np.convolve(y, hrf)
        y = y[:L]
        y = y / np.std(y, ddof=1)
        yFft = np.abs(np.fft.fft(y, nFFT) / L)
        yFft = yFft[:f.shape[0]]

        out = np.zeros(len(thres))

        for j in range(len(thres) - 1):
            x1 = f > thres[j]
            x2 = f <= thres[j + 1]
            out[j] = np.sum(yFft[x1 & x2])

        out[j + 1] = np.sum(yFft[f > thres[j + 1]])
        features_null += out

    features_null = features_null / 100

    eps = np.finfo(np.float64).eps
    out = (featuresReal - features_null) ** 2 / (eps + (featuresReal ** 2))

    out = np.concatenate((out, [np.sum(out)]))

    return list(out)


@cached(cache={}, key=lambda conf, sigma, tr: hashkey(conf.tostring(), sigma, tr))
def motionconfounds(conf, sigma, tr):

    log.info('Filter and normalise motion confounds')

    conf = np.array(conf)

    conf0 = np.vstack((np.zeros((1, conf.shape[1])),
                       np.diff(conf, axis=0)))
    conf0 = np.hstack((conf, conf0))
    conf0 = norm(conf0)
    conf0 = np.hstack((conf0, conf0 * conf0))
    conf0 = norm(conf0)

    if sigma is not None:
        conf0 = hpfilt(conf0.T, sigma, tr).T
    conf = norm(conf0)

    return conf


def hpfilt(func, sigma, tr):
    if sigma == 0:
        func = scipy.signal.detrend(func)
    else:
        n_t = func.shape[1]
        n_v = func.shape[0]

        dim = np.ceil(n_v / 100.0).astype(int)
        func_0 = np.zeros((100 * dim, n_t))

        func_0[0:n_v, :] = func

        func_0 = func_0.reshape((10, 10, dim, n_t), order="F")

        with TemporaryDirectory() as tmp:
            tmpname = op.join(tmp, 'func')
            img = nb.Nifti1Image(func_0, np.eye(4))
            nb.save(img, tmpname + ".nii.gz")

            util.run(['fslmodhd', tmpname + '.nii.gz', 'pixdim4', str(tr)])
            util.run(['fslmaths', tmpname, '-Tmean', tmpname + '_mean'])
            util.run(['fslmaths', tmpname, '-bptf', str(sigma), '-1', '-add',
                      tmpname + '_mean', tmpname])
            img = nb.load(tmpname + ".nii.gz")
            func = img.get_data().reshape((100 * dim, img.shape[-1]), order="F")

        # os.remove(tmpname + ".nii.gz")
        # os.remove(tmpname + "_mean.nii.gz")

        func = func[0:n_v, :]

    return func


def norm(d, dim=0):
    d = d - np.mean(d, axis=dim)
    d = d / np.maximum(np.std(d, axis=dim), 0.00001)
    return d


@feature_extractor
def motioncorrelation(data, ic_idx):
    # calculate the time series' correlation w/motion

    ts = data.ica.get_timecourse(ic_idx)
    TR = data.ica.tr

    motparam = np.stack([data.timecourses[k].data for k in ['RotX', 'RotY', 'RotZ', 'X', 'Y', 'Z']], axis=1)

    hp = data.hp_fwhm

    sigma = 0.5*hp/TR

    TSS = motionconfounds(motparam, sigma, TR)

    out = []
    C1 = np.corrcoef(ts.T, TSS.T)
    C1 = np.abs(C1[0, 1:])
    out += [np.max(C1[:6])]
    out += [np.max(C1[6:])]
    out += [np.max(out)]

    eps = np.finfo(float).eps
    regModel = np.dot(np.linalg.pinv(np.concatenate((np.ones((ts.shape[0], 1)), TSS), axis=1)), ts) / (eps + np.std(ts))
    regModel = np.sort(np.abs(regModel[1:]))

    n = int(np.floor(regModel.shape[0] / 2))
    out += [regModel[-2], regModel[-1], np.mean(regModel[n - 1:])]

    return out


def functionoupmle(S, deltat):
    # % The Ornstein Uhlenbeck process is widely used for modelling a
    # % mean-reverting process. The process S is modelled as
    # % dS = \lambda (\mu-S)dT + \sigma dW
    # % Where
    # % W is a Brownian- Motion, so dW~N(0, dt),
    # % \lambda meaures the speed of mean reversion
    # % \mu is the 'long run mean', to which the process tends to revert.
    # % \sigma, as usual, is a measure of the process volatility

    n = np.max(S.shape) - 1
    Sx = np.sum(S[:-1])
    Sy = np.sum(S[1:])
    Sxx = np.sum(S[:-1] ** 2)
    Sxy = np.sum(S[:-1] * S[1:])
    Syy = np.sum(S[1:] ** 2)

    mu = (Sy * Sxx - Sx * Sxy) / (n * (Sxx - Sxy) - (Sx ** 2 - Sx * Sy))

    _log_val = (Sxy - mu * Sx - mu * Sy + n * mu ** 2) / (Sxx - 2 * mu * Sx + n * mu ** 2)
    if _log_val < 0:
        _log_val = complex(_log_val)

    _log_val = np.log(_log_val)

    # if isinstance(_log_val, complex):
    #     _log_val = _log_val.real

    lmbda = -(1 / deltat) * _log_val

    alpha = np.exp(-lmbda * deltat)
    alpha2 = np.exp(-2 * lmbda * deltat)
    sigmahat2 = (1 / n) * (
                Syy - 2 * alpha * Sxy + alpha2 * Sxx - 2 * mu * (1 - alpha) * (Sy - alpha * Sx) + n * mu ** 2 * (
                    1 - alpha) ** 2)

    sigma = np.sqrt(sigmahat2 * 2 * lmbda / (1 - alpha2))

    return mu, sigma, lmbda


@feature_extractor
def oupjk(data, ic_idx):
    ts = data.ica.get_timecourse(ic_idx)
    TR = data.ica.tr
    return list(np.abs(_oupjk(ts, TR)))


def _oupjk(S, deltat):
    # % The Ornstein Uhlenbeck process is widely used for modelling a
    # % mean-reverting process. The process S is modelled as
    # % dS = \lambda (\mu-S)dT + \sigma dW
    # % Where
    # % W is a Brownian- Motion, so dW~N(0, dt),
    # % \lambda meaures the speed of mean reversion
    # % \mu is the 'long run mean', to which the process tends to revert.
    # % \sigma, as usual, is a measure of the process volatility

    # % Since the basic ML has a bias (resulting in frequent estimates of
    # % lambda which are much too high), we perform a 'jackknife' operation
    # % to reduce the bias.

    # % Regressions prefer row vectors to column vectors, so rearrange if
    # % necessary.

    # if S.shape[1] > S.shape[0]:
    #     S = S.T

    m = 2  # number of partitions
    partlength = int(np.floor(np.max(S.shape) / m))
    Spart = np.zeros((m, partlength))
    for i in np.arange(m):
        Spart[i, :] = S[partlength * i:partlength * (i + 1)]

    [muT, sigmaT, lambdaT] = functionoupmle(S, deltat)

    # % Calculate the individual partitions.
    mupart = [None] * m #np.zeros(m)
    sigmapart = [None] * m #np.zeros(m)
    lambdapart = [None] * m #np.zeros(m)
    for i in np.arange(m):
        mupart[i], sigmapart[i], lambdapart[i] = functionoupmle(np.ravel(Spart[i, :]), deltat)

    # for i=1:1:m
    #     [mupart(i), sigmapart(i), lambdapart(i)] = functionoupmle(Spart(i,:), deltat);
    # end
    # % Now the jacknife calculation.
    lmbda = (m / (m - 1)) * lambdaT - (np.nansum(lambdapart)) / (m ** 2 - m)
    # % mu and sigma are not biased, so there's no real need for the jackknife.
    # % But we do it anyway for demonstration purposes.
    # mu = (m/(m-1))*muT - (np.sum(mupart))/(m**2-m)
    sigma = (m / (m - 1)) * sigmaT - (np.nansum(sigmapart)) / (m ** 2 - m)
    # features =  [sigma lambda];

    return [sigma, lmbda]


# @cached(cache={}, key=lambda map0, pixdim, fname: hashkey(map0.tostring(), pixdim, fname))
# def save_map(map0, pixdim, fname):
#
#     img = nb.Nifti1Image(np.array(map0), affine=np.eye(4))
#
#     print(f'save: {fname}')
#
#     pxd = img.header['pixdim']
#     pxd[1:1+len(pixdim)] = pixdim
#
#     img.header['pixdim'] = pxd
#     img.to_filename(fname)
#
#     return fname

@feature_extractor
def clusterdist(data, ic_idx, ic_threshold=2.5):
    # % this feature tries to quantify the number of clusters in a component

    xSiz = data.ica.maps.pixdim[:-1]
    fname = _get_abs_ic_filename(data, ic_idx)

    cmd = ['cluster', '-i', fname, '-t', str(ic_threshold)]

    clst = np.loadtxt(io.StringIO(util.run(cmd)), skiprows=1, delimiter='\t')

    if clst.shape[0] > 0:
        X = clst[:, 1]
        X = X[X > 4]
        if X.shape[0] < 3:
            X = np.concatenate((X, np.zeros(3 - X.shape[0])))
        Y = np.sort(X)[::-1] * xSiz[0] * xSiz[1] * xSiz[2]
        feat = [X.shape[0], np.mean(X) - np.median(X), np.max(X),
                np.var(X, ddof=1), scipy.stats.skew(X),
                scipy.stats.kurtosis(X, fisher=False), Y[0], Y[1], Y[2]]
    else:
        feat = [0, 0, 0, 0, 0, 1.5, 0, 0, 0]

    return feat


@cached(cache={}, key=lambda data, ic_idx: hashkey('_get_abs_ic_filename', ic_idx))
def _get_abs_ic_filename(data, ic_idx):

    fname = op.join(data.fixdir, f'icmap{ic_idx}_abs.nii.gz')

    # print(f'save _get_abs_image_filename: {fname}')
    icmap = data.ica.get_spatialmap(ic_idx)
    nb.Nifti1Image(
        np.abs(icmap.get_fdata()),
        affine=icmap.affine,
        header=icmap.header,
    ).to_filename(fname)

    return fname


@cached(cache={}, key=lambda data, ic_idx: hashkey('_get_ic_filename', id(data), ic_idx))
def _get_ic_filename(data, ic_idx):

    filename = op.join(data.fixdir, f'icmap{ic_idx}.nii.gz')

    # print(f'save _get_image_filename: {filename}')
    data.ica.get_spatialmap(ic_idx).to_filename(filename)

    return filename


@cached(cache={}, key=lambda d: hashkey('_get_data_tmean_filename', id(d)))
def _get_data_tmean_filename(d):

    filename = op.join(d.fixdir, f'input_data_tmean.nii.gz')

    # print(f'save _get_mean_data_filename: {filename}')

    d0 = d.input.data

    nb.Nifti1Image(
        np.mean(d0.get_fdata(), axis=-1),
        affine=d0.affine,
        header=d0.header,
    ).to_filename(filename)

    return filename


@feature_extractor
def negativevspositive(data, ic_idx, ic_threshold=2.5):

    icmap = _get_ic_filename(data, ic_idx)
    icmap_abs = _get_abs_ic_filename(data, ic_idx)

    out = []

    out += [float(util.run(['fslstats', icmap, '-E']))]
    out += [float(util.run(['fslstats', icmap_abs, '-E']))]

    a = float(util.run(['fslstats', icmap, '-M']))
    aS = float(util.run(['fslstats', icmap, '-S']))

    b = float(util.run(['fslstats', icmap_abs, '-M']))
    bS = float(util.run(['fslstats', icmap_abs, '-S']))

    eps = np.finfo(float).eps

    out += [np.abs(a) / (eps + np.abs(b))]
    out += [np.abs(a * bS) / np.abs(eps + aS * b)]

    # % diff (for abs of pos and neg voxels)
    img = data.ica.get_spatialmap(ic_idx).get_fdata()
    out += [np.abs(np.sum(img > 0) - np.sum(img < 0)) / (eps + np.sum(img > 0))]
    out += [np.abs(np.sum(img > ic_threshold) - np.sum(img < -ic_threshold)) / (eps + np.sum(img > ic_threshold))]

    return out


# @cached(cache={}, key=lambda d0: hashkey(d0.tostring()))
# def _calc_tmean(d0):
#     log.info('Calculate Tmean func')
#     print('calc tmean')
#     return np.mean(d0, axis=-1)


# TODO: This is SUPER SLOW!!! investigate why. probably the gaussian smoothing.
@feature_extractor
def zstattofuncratio(data, ic_idx):
    # % compare high zstat voxels against original EPI mean image intensities
    # % (in some artefacts the mean FMRI image is dark)
    # % Similar to above - compare zstat against original EPI *edge* image
    # % and/or the original PCA residual image
    # % global icThreshold;

    # defaults = DEFAULTS.update(workdir=data['fixdir'])

    icmap = _get_ic_filename(data, ic_idx)

    # icmap = data.get_ic(ic_idx).map
    # icmap = save_map(icmap, data.map_pixdim, op.join(data.fixdir, f'icmap{ic_idx}.nii.gz'))

    # func_mean = _calc_tmean(data.data)
    # func_mean = save_map(func_mean, data.map_pixdim, op.join(data.fixdir, f'func_mean.nii.gz'))

    func_mean = _get_data_tmean_filename(data)

    workdir = data.fixdir

    img_ratio = op.join(workdir, 'dummyd')
    util.run(['fslmaths', icmap, '-s', '5', '-abs', '-div', func_mean, img_ratio])
    img_ratio = nb.load(img_ratio + '.nii.gz').get_data()

    img_mult = op.join(workdir, 'dummym')
    util.run(['fslmaths', icmap, '-s', '5', '-abs', '-mul', func_mean, img_mult])
    img_mult = nb.load(img_mult + '.nii.gz').get_data()

    a = np.logical_not(np.isnan(img_mult)) & np.logical_not(np.isinf(img_mult))
    dmp = img_mult[(img_mult > 0) & a]

    a = np.logical_not(np.isnan(img_ratio)) & np.logical_not(np.isinf(img_ratio))
    drp = img_ratio[(img_ratio > 0) & a]

    return list(np.concatenate((np.percentile(dmp, [99, 95]), np.percentile(drp, [99, 95]))))


@feature_extractor
def slicewisestats(data, ic_idx, ic_threshold=2.5):
    # % all-of-a-single-slice and none-of-others or every odd (or even) slices -
    # % effects; the above also tend to show up as VERY sparse or oddly
    # % nonGaussian in time domain

    icmap = data.ica.get_spatialmap(ic_idx).get_fdata()
    icmap_abs = np.abs(icmap)

    out = []
    eps = np.finfo(float).eps

    sumSlc = np.zeros(icmap_abs.shape[2])
    sumTot = np.sum(icmap_abs ** 2) + eps

    for i in range(icmap_abs.shape[2]):
        sumSlc[i] = np.sum(icmap_abs[:, :, i] ** 2) / sumTot

    out += [np.sum(sumSlc > 0.15)]
    out += [np.max(sumSlc)]

    sumSlc = np.zeros(icmap_abs.shape[2])
    sumTot = np.sum(icmap_abs > ic_threshold) + eps

    for i in range(icmap_abs.shape[2]):
        sumSlc[i] = np.sum(icmap_abs[:, :, i] > ic_threshold) / sumTot

    out += [np.sum(sumSlc > 0.15)]
    out += [np.max(sumSlc)]

    return out


@feature_extractor
def everynthvariance(data, ic_idx, ic_threshold=2.5):
    # % all-of-a-single-slice and none-of-others or every odd (or even) slices -
    # % effects; the above also tend to show up as VERY sparse or oddly
    # % nonGaussian in time domain (this is like F4 in Podrack 08)

    icmap = data.ica.get_spatialmap(ic_idx).get_fdata()
    icmap_abs = np.abs(icmap)

    out = []
    eps = np.finfo(float).eps

    N = icmap_abs.shape[2]
    varSlc = np.zeros(N)

    for i in range(N):
        varSlc[i] = np.sum(icmap_abs[:, :, i] ** 2)

    out += [np.abs(np.sum(varSlc[0::2]) - np.sum(varSlc[1::2])) / (eps + np.sum(varSlc))]

    ##

    AA = np.concatenate((np.arange(0, N, 4), np.arange(1, N, 4)))
    BB = np.concatenate((np.arange(2, N, 4), np.arange(3, N, 4)))

    out += [np.abs(np.sum(varSlc[AA]) - np.sum(varSlc[BB])) / (eps + np.sum(varSlc))]

    ##

    varSlc = np.zeros(N)
    icmap_abs = icmap_abs * (icmap_abs > ic_threshold)

    for i in range(N):
        varSlc[i] = np.var(icmap_abs[:, :, i])

    out += [np.abs(np.sum(varSlc[0::2]) - np.sum(varSlc[1::2])) / (eps + np.sum(varSlc))]

    ##

    out += [np.abs(np.sum(varSlc[AA]) - np.sum(varSlc[BB])) / (eps + np.sum(varSlc))]

    return out


@lru_cache()
def _extract_tissue_timeseries(func, dseg):
    ts_csf = np.mean(func.get_fdata()[dseg.get_fdata() == 1], axis=0)
    ts_gm = np.mean(func.get_fdata()[dseg.get_fdata() == 2], axis=0)
    ts_wm = np.mean(func.get_fdata()[dseg.get_fdata() == 3], axis=0)

    return ts_csf, ts_gm, ts_wm


# @cached(cache={}, key=lambda , idx: hashkey(idx))
# def _calc_func_mean(func, func_mean_name):
#     log.info('Calculate Tmean func')
#     fslmaths(func).Tmean().run(func_mean_name)
#     return nb.load(func_mean_name)


# @lru_cache()
# def _transform_seg(struct_dseg, ref, xfm, out_name):
#     log.info('Transform structural segmentation to native functional space')
#     _apply_xfm(
#         src=struct_dseg,
#         ref=ref,
#         xfm=xfm,
#         interp='nearestneighbour',
#         out_name=out_name
#     )
#     return nb.load(out_name)


@lru_cache()
def _apply_xfm(src, ref, xfm, out_name, interp='spline'):

    if xfm.endswith('.mat'):
        log.info(f'Apply transform: affine')
        applyxfm(
            src=src,
            ref=ref,
            mat=xfm,
            interp='nearestneighbour' if interp == 'nn' else interp,
            out=out_name,
        )
    elif xfm.endswith('.nii.gz'):
        log.info(f'Apply transform: warp')
        applywarp(
            src=src,
            ref=ref,
            warp=xfm,
            interp='nn' if interp == 'nearestneighbour' else interp,
            out=out_name,
        )
    else:
        raise RuntimeError('Unknown transform type!')

    return nb.load(out_name)


@cached(cache={}, key=lambda data: hashkey('_get_dseg_filename', id(data)))
def _get_dseg_filename(data):

    dseg = op.join(data.fixdir, f'dseg.nii.gz')

    # print(f'save _get_dseg_filename: {dseg}')
    data.spatialmaps['dseg'].data.to_filename(dseg)

    return dseg


@cached(cache={}, key=lambda data: hashkey('_get_brainmask_filename', id(data)))
def _get_brainmask_filename(data):

    d = op.join(data.fixdir, f'brainmask.nii.gz')

    # print(f'save _get_brainmask_filename: {d}')
    data.spatialmaps['brainmask'].data.to_filename(d)

    return d

@cached(cache={}, key=lambda data, map_key: hashkey(map_key, id(data)))
def _get_map_with_tempname(data, map_key):

    fname = op.join(data.fixdir, f'{map_key}.nii.gz')

    # print(f'_get_map_filename: {map_key} {fname}')
    d = data.spatialmaps[map_key].data
    d.to_filename(fname)

    return fname


@feature_extractor
def masktscorrandoverlap(data, ic_idx, ic_threshold=2.5):
# def masktscorrandoverlap(icmap_abs, ts, seg, ts_gm, ts_wm, ts_csf, ic_threshold):
    # % In this function, given the time series, the mean CSF time-series is
    # % estimated and then its correlation with the given ts is calc'd
    # % Next, the overlap of spatial map w/each and everyone of these masks is
    # % estimated

    icmap = data.ica.get_spatialmap(ic_idx).get_fdata() #ndarray
    icmap_abs = np.abs(icmap) #ndarray
    ts = data.ica.get_timecourse(ic_idx) #ndarray

    struct_dseg = _get_dseg_filename(data)
    struct2func_xfm = data.spatialmaps['dseg'].xfm
    fixdir = data.fixdir

    func_mean = _get_data_tmean_filename(data)

    seg = _apply_xfm(
        src=struct_dseg,
        ref=func_mean,
        xfm=struct2func_xfm,
        interp='nearestneighbour',
        out_name=f'{fixdir}/dseg-to-input.nii.gz'
    )

    func = data.input.data #nifti
    ts_csf, ts_gm, ts_wm = _extract_tissue_timeseries(func, seg)

    out = []
    eps = np.finfo(float).eps

    clf = linear_model.LinearRegression()

    ts = ts / scipy.std(ts)
    N = ts.shape[0]

    X = np.concatenate((
        np.zeros((N, 1)),
        (ts_wm / (eps + np.std(ts_wm)))[:, np.newaxis],
        (ts_gm / (eps + np.std(ts_gm)))[:, np.newaxis],
        (ts_csf / (eps + np.std(ts_csf)))[:, np.newaxis]), axis=1)

    clf.fit(X, ts)
    out += list(np.abs(clf.coef_[1:]))

    seg = seg.get_data()

    denom = eps + np.sum(icmap_abs)
    for i in range(1, 4):
        out += [np.sum(icmap_abs * (seg == i)) / denom]

    icmap_abs_th = icmap_abs > ic_threshold
    denom = eps + np.sum(icmap_abs_th)
    for i in range(1, 4):
        out += [np.sum(icmap_abs_th * (seg == i)) / denom]

    wch = icmap_abs > ic_threshold
    denom = eps + np.sum(icmap_abs[wch])
    for i in range(1, 4):
        out += [np.sum(icmap_abs[wch] * (seg[wch] == i)) / denom]

    return out


@feature_extractor
def smoothest(data, ic_idx):
    # % closely related: spatial smoothness (resel size etc) of zstat map
    # % spatial entropy (histogram) / parameters from histogram mixture-model fit
    # % smoothness

    icmap = _get_ic_filename(data, ic_idx)
    func_brain_mask = _get_brainmask_filename(data)

    smth = util.run(['smoothest', '-z', icmap, '-m', func_brain_mask])
    smth = smth.split('\n')
    for l in smth:
        if l.startswith('RESELS'):
            out = float(l.split(' ')[1])
    return [out]


@feature_extractor
def spatialoverlap(data, ic_idx):

    # func = data['func']
    # struct_dseg = data['struct_dseg']
    # struct2func_xfm = data['struct2func_xfm']
    # fixdir = data['fixdir']

    fixdir = data.fixdir

    struct_dseg = _get_dseg_filename(data)
    struct2func_xfm = data.spatialmaps['dseg'].xfm


    # func_mean = _calc_func_mean(func, defaults.get('func_mean', make_dir=True))
    func_mean = _get_data_tmean_filename(data)

    seg = _apply_xfm(
        src=struct_dseg,
        ref=func_mean,
        xfm=struct2func_xfm,
        interp='nearestneighbour',
        out_name=f'{fixdir}/dseg-to-input.nii.gz'
    )

    # icmap_abs = data['ic_map_abs'].get_fdata()

    icmap_abs = np.abs(data.ica.get_spatialmap(ic_idx).get_fdata())

    seg = seg.get_fdata()
    pixdim = data.ica.maps.pixdim

    eps = np.finfo(float).eps
    denom = eps + np.sum(icmap_abs)
    out = np.sum(icmap_abs * (seg == 3)) / denom

    return out * pixdim[0] * pixdim[1] * pixdim[2]


@feature_extractor
def maxtfce(data, ic_idx, ic_threshold=2.5):
    # % test for "clusterness" / "compactness" - e.g. feed through TFCE and
    # % take the max (maybe pre-TFCE normalise e.g. by max(abs(z)))

    # ic_threshold = data['ic_threshold']
    icmap = _get_ic_filename(data, ic_idx)

    std = util.run(['fslstats', icmap, '-S'])

    out = []

    icmap2 = op.join(op.dirname(icmap), 'dummy2')
    util.run(['fslmaths', icmap, '-abs', '-div', std, '-tfce', '2', '0.5', '6', icmap2])
    out += [float(util.run(['fslstats', icmap2, '-P', '100']))]

    util.run(['fslmaths', icmap, '-abs', '-tfce', '2', '0.5', '6', icmap2])
    out += [float(util.run(['fslstats', icmap2, '-P', '100']))]

    util.run(['fslmaths', icmap, '-abs', '-thr', str(ic_threshold), '-tfce', '2', '0.5', '6', icmap2])
    out += [float(util.run(['fslstats', icmap2, '-P', '100']))]

    return out


@lru_cache(maxsize=None)
def _create_edge_masks(func_brain_mask, workdir):
    # create edge masks if not already created
    # load edge masks

    log.info('Creating edge masks')

    # create edge masks in native functional space
    # from the func brain mask
    mask = func_brain_mask
    ero_mask = [mask] + [op.join(workdir, f'maske{i+1}') for i in range(5)]
    edge_mask = [op.join(workdir, f'edge{i+1}') for i in range(5)]

    for i in range(5):
        util.run(['fslmaths', ero_mask[i], '-eroF', ero_mask[i+1]])
        util.run(['fslmaths', mask, '-sub', ero_mask[i+1], edge_mask[i]])

    return [nb.load(f'{m}.nii.gz') for m in edge_mask]


@feature_extractor
def edgemasks(data, ic_idx, ic_threshold=2.5):
    # % compare high zstat voxels against original EPI mean image intensities
    # % (in some artefacts the mean FMRI image is dark)
    # % Similar to above - compare zstat against original EPI *edge* image
    # % and/or the original PCA residual image

    # ic_threshold = data['ic_threshold']
    icmap = data.ica.get_spatialmap(ic_idx)
    # icmap_abs = data['ic_map_abs']
    # func_brainmask = data['func_brainmask']
    func_brainmask = _get_brainmask_filename(data)
    workdir = data.fixdir

    edge_masks = _create_edge_masks(func_brainmask, workdir)

    out = []

    eps = np.finfo(float).eps

    icmap_abs = np.abs(icmap.get_fdata())
    icmap = np.abs(icmap.get_data()) > ic_threshold


    for msk in edge_masks:
        msk = msk.get_data()
        whatPercentOnEdgeB = np.sum(icmap * msk) / (eps + np.sum(icmap))
        whatPercentOfEdge = np.sum(icmap * msk) / (eps + np.sum(msk))

        wch = icmap_abs > ic_threshold
        whatPercentOnEdgeC = np.sum(icmap_abs[wch] * msk[wch]) / (eps + np.sum(icmap_abs[wch]))

        out += [whatPercentOnEdgeB, whatPercentOnEdgeC, whatPercentOfEdge]

    return out


@cached(
    cache={},
    key=lambda func, standard2func_tx, struct, struct2func_xfm, workdir: hashkey('_create_sag_masks')
)
def _create_sag_masks(
        func: nb.Nifti1Image,
        standard2func_tx: nb.Nifti1Image,
        struct: nb.Nifti1Image,
        struct2func_xfm: str,
        workdir,
) -> List[str]:

    # FIXDIR = os.getenv('FIXPYDIR', '')
    FSLDIR = os.getenv('FSLDIR', '')

    # create standard_to_func transform if it does not exist
    if standard2func_tx is None:
        log.info('Affine registration of native structural to standard (MNI152)')

        struct2std = op.join(workdir, 'struct2std.mat')
        flirt(
            src=struct,
            ref=op.join(FSLDIR, 'data/standard/MNI152_T1_1mm_brain'),
            omat=struct2std,
        )

        std2struct = op.join(workdir, 'std2struct.mat')
        invxfm(struct2std, std2struct)

        if struct2func_xfm.endswith('.mat'):

            standard2func_tx = op.join(workdir, 'std2func.mat')
            concatxfm(
                std2struct,
                struct2func_xfm,
                standard2func_tx,
            )

        elif struct2func_xfm.endswith('.nii.gz'):

            standard2func_tx = op.join(workdir, 'std2func_warp.nii.gz')
            convertwarp(
                ref=func,
                premat=std2struct,
                warp1=struct2func_xfm,
                out=standard2func_tx,
            )

        else:
            raise RuntimeError('Unknown transform type!')


    # transform standard space sagittal masks into native functional space

    sag_masks = np.empty((4, 3), dtype=object)

    dlabel = ['', 'dil', 'dil2']

    for r in range(sag_masks.shape[0]):
        for c in range(sag_masks.shape[1]):

            in_name = op.join(util.get_resource_path(), 'masks', 'sag_mask{}{}'.format(r, dlabel[c]))
            out_name = op.join(workdir, 'std1mm2exfunc{}{}'.format(r, dlabel[c]))

            sag_masks[r, c] = _apply_xfm(
                src=in_name,
                ref=func,
                xfm=standard2func_tx,
                interp='spline',
                out_name=out_name+'.nii.gz'
            )

        # TODO: deal with veins_exf

    return list(sag_masks)


@feature_extractor
def sagmasks(data, ic_idx, ic_threshold=2.5):
    # % look at intersections of ICA spatial map with major vein mask images

    # ic_threshold = data['ic_threshold']
    # icmap_abs = data['ic_map_abs']
    # func = data['func']
    # standard2func_tx = data['standard2func_xfm']
    # struct = data['struct']
    # struct2func_xfm = data['struct2func_xfm']
    # workdir = data['workdir']

    workdir = data.fixdir
    icmap_abs = nb.load(_get_abs_ic_filename(data, ic_idx))  # nifti
    func_mean = nb.load(_get_data_tmean_filename(data))  # nifti
    struct = _get_map_with_tempname(data, 'struct')   # nifti
    struct2func_xfm = data.spatialmaps['struct'].xfm  # fname
    standard2func_tx = data.spatialmaps['standard'].xfm if 'standard' in data.spatialmaps else None # fname

    # icmap = data.ica.get_spatialmap(ic_idx).get_fdata()  # ndarray
    # icmap_abs = np.abs(icmap)  # ndarray
    # ts = data.ica.get_timecourse(ic_idx)  # ndarray
    #
    # struct_dseg = _get_dseg_filename(data)
    # struct2func_xfm = data.spatialmaps['dseg'].xfm
    # fixdir = data.fixdir

    # func_mean = _get_data_tmean_filename(data)

    if not op.exists(workdir):
        os.makedirs(workdir)

    if struct is None:
        raise RuntimeError('struct is required for feature extractor: sagmasks')

    # func_mean = _calc_func_mean(func, op.join(workdir, 'func_mean.nii.gz'))

    out = []

    icmap_abs = icmap_abs.get_data()

    icmap_abs2 = icmap_abs > ic_threshold
    wch = icmap_abs > ic_threshold

    sag_masks = _create_sag_masks(
        func=func_mean,
        standard2func_tx=standard2func_tx,
        struct=struct,
        struct2func_xfm=struct2func_xfm,
        workdir=workdir
    )

    sag_masks = np.asarray(sag_masks)

    for i in range(sag_masks.shape[0]):
        for j in range(sag_masks.shape[1]):
            smask = sag_masks[i, j].get_data()

            whatPercentOnEdgeB = np.sum(icmap_abs2 * smask) / np.maximum(1, np.sum(icmap_abs2))
            whatPercentOfEdge = np.sum(icmap_abs2 * smask) / np.maximum(1, np.sum(smask))
            whatPercentOnEdgeC = np.sum(icmap_abs[wch] * smask[wch]) / np.maximum(1, np.sum(icmap_abs[wch]))

            grot1 = np.concatenate((smask.reshape(1, -1), icmap_abs2.reshape(1, -1), icmap_abs.reshape(1, -1)), axis=0)
            grot1 = np.corrcoef(grot1)
            grot1[np.isnan(grot1)] = 0

            grot2 = np.concatenate((smask[wch].reshape(1, -1), icmap_abs[wch].reshape(1, -1)), axis=0)
            grot2 = np.corrcoef(grot2)
            grot2[np.isnan(grot2)] = 0

            out += [whatPercentOnEdgeB, whatPercentOnEdgeC, whatPercentOfEdge, grot1[0, 1], grot1[0, 2], grot2[0, 1]]

    return out

@feature_extractor
def stripe(data, ic_idx):

    icmap = nb.load(_get_ic_filename(data, ic_idx))

    dm1 = op.join(op.dirname(icmap.get_filename()), 'dummy01')
    util.run(['fslmaths', icmap.get_filename(), '-s', '2', '-abs', dm1])
    util.run(['fslmaths', icmap.get_filename(), '-abs', '-s', '2', '-sub', dm1, '-abs', dm1])
    p95 = float(util.run(['fslstats', dm1, '-P', '95']))

    util.run(['imrm', dm1])

    return [p95]
