#!/usr/bin/env python
from spectrum import aryule
import numpy as np
import nibabel as nb
import scipy
import util
import uuid
import os
import io
import os.path as op
from sklearn import linear_model


def arvswgn(ts, TR, norder=6):
    mvec = np.zeros(norder)
    for i in range(norder):
        mvec[i] = aryule(ts, i+1)[1]

    return list(np.polyfit(np.arange(norder), mvec, 1))


def arfull(ts, TR, norder=1):
    AR, V, k = aryule(ts, norder)
    return list(np.concatenate(([V], AR)))


def entropy(ts):
    bins = np.linspace(-2, 2, int(np.floor(np.sqrt(ts.shape[0]))))
    X = np.histogram(ts, bins=bins)[0]
    X = X / np.sum(X)
    X = -np.sum(X * np.log(X + np.finfo(float).eps))

    S = ts / (np.std(ts)+np.finfo(float).eps)
    S = np.sum((np.mean(-np.exp(-S**2/2))+0.71)**2)

    return [X, S]


def tsjump(ts):
    # Some jump characteristics of the time series
    diff_ts = np.diff(ts)

    eps = np.finfo(float).eps

    out = []
    out += (np.max(np.abs(diff_ts/(eps+np.std(ts)))),)
    out += (np.max(np.abs(diff_ts/(eps+np.std(diff_ts)))),)
    out += (np.sum((diff_ts/(eps+np.std(ts)))**2)/ts.shape[0],)

    # from Russell Poldrack's 08; Feature 5

    mx = np.amax(np.abs(diff_ts))
    imx = np.argmax(np.abs(diff_ts))

    sel = np.arange(imx-2, imx+2)

    x1 = sel >= 0
    x2 = sel < diff_ts.shape[0]
    sel = sel[x1 & x2]

    diff_ts = np.delete(diff_ts, sel)

    out += (mx/(eps+np.mean(np.abs(diff_ts))),)
    out += (mx/(eps+np.sum(np.abs(diff_ts))),)

    # from Russell Poldrack's 08; Feature 6
    ts = ts-np.mean(ts)
    out += (np.corrcoef([ts[:-1], ts[1:]])[0, 1],)

    return out


def nextpow2(x):
    return np.ceil(np.log2(x)).astype('int')


def fftcoarse(ts, TR, freq_threshold):
    # In this function, given the time series and its sampling rate (e.g,
    # TR), we calculate the ratio of its high-frequency energy to low-frequency
    # energy

    L = ts.shape[0]
    nFFT = 2**nextpow2(L)

    Y = np.abs(np.fft.fft(ts, nFFT)/L)
    f = ((1/TR)/2)*np.linspace(0, 1, int(nFFT/2))
    Y = Y[0:f.shape[0]]

    eps = np.finfo(float).eps

    return [np.sum(Y[f >= freq_threshold])/(eps+np.sum(Y[f < freq_threshold]))]


def fftfiner(ts, TR, thres):
    # In this function, given the time series and its sampling rate (e.g,
    # TR), we calculate the ratio of its high-frequency energy to low-frequency
    # energy

    L = ts.shape[0]  # Length of signal
    nFFT = 2**nextpow2(L)  # Next power of 2 from length of S

    Y = np.abs(np.fft.fft(ts, nFFT)/L)
    f = ((1/TR)/2)*np.linspace(0, 1, int(nFFT/2))
    Y = Y[0:f.shape[0]]

    out = np.zeros(len(thres))

    for i in range(len(thres)-1):
        x1 = f > thres[i]
        x2 = f <= thres[i+1]
        out[i] = np.sum(Y[x1 & x2])

    out[i+1] = np.sum(Y[f > thres[i+1]])

    eps = np.finfo(float).eps
    out = out / (eps + np.sum(out))

    return list(out)


def fftfinerwrtnull(ts, TR, thres):
    # Concolve single-gamma HRF w/WGN and compare its spectrum w/actual RSN
    # do it for real

    L = ts.shape[0]  # Length of signal
    nFFT = 2**nextpow2(L)  # Next power of 2 from length of S

    Y = np.abs(np.fft.fft(ts, nFFT)/L)
    f = ((1/TR)/2)*np.linspace(0, 1, int(nFFT/2))
    Y = Y[:f.shape[0]]

    out = np.zeros(len(thres))

    for i in range(len(thres)-1):
        x1 = f > thres[i]
        x2 = f <= thres[i+1]
        out[i] = np.sum(Y[x1 & x2])

    out[i+1] = np.sum(Y[f > thres[i+1]])

    featuresReal = out.copy()
    # do it under the H0
    delay = 6/TR
    sigma = delay/2

    hrf = scipy.stats.gamma.pdf(np.arange(0, ts.shape[0]), (delay/sigma)**2, delay/(sigma**2))

    features_null = np.zeros(len(thres))

    for i in np.arange(100):
        y = np.random.normal(0, 1, L)
        y = np.convolve(y, hrf)
        y = y[:L]
        y = y / np.std(y)
        yFft = np.abs(np.fft.fft(y, nFFT)/L)
        yFft = yFft[:f.shape[0]]

        out = np.zeros(len(thres))

        for j in range(len(thres)-1):
            x1 = f > thres[j]
            x2 = f <= thres[j+1]
            out[j] = np.sum(yFft[x1 & x2])

        out[j+1] = np.sum(yFft[f > thres[j+1]])
        features_null += out

    features_null = features_null / 100

    eps = np.finfo(np.float64).eps
    out = (featuresReal-features_null)**2/(eps+(featuresReal**2))

    out = np.concatenate((out, [np.sum(out)]))

    return list(out)


def motionconfounds(motparams, sigma, tr):
    conf = np.loadtxt(motparams)
    conf = conf[:, 0:6]
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

        # TODO: should use a sensible tempname (and location)
        tmpname = str(uuid.uuid4())
        img = nb.Nifti1Image(func_0, np.eye(4))
        nb.save(img, tmpname + ".nii.gz")

        util.run(['fslmodhd',  tmpname+'.nii.gz', 'pixdim4', str(tr)])
        util.run(['fslmaths', tmpname, '-Tmean', tmpname+'_mean'])
        util.run(['fslmaths', tmpname, '-bptf', str(sigma), '-1', '-add',
                  tmpname+'_mean', tmpname])
        img = nb.load(tmpname + ".nii.gz")
        func = img.get_data().reshape((100 * dim, img.shape[-1]), order="F")

        os.remove(tmpname + ".nii.gz")
        os.remove(tmpname + "_mean.nii.gz")

        func = func[0:n_v, :]

    return func


def norm(d, dim=0):
    d = d - np.mean(d, axis=dim)
    d = d / np.maximum(np.std(d, axis=dim), 0.00001)
    return d


def motioncorrelation(ts, motparam):
    # calculate the time series' correlation w/motion

    # hp = 0
    # if design is not None:
    #     if int(design['temphp_yn']):
    #         hp = float(design['paradigm_hp'])

    # sigma = 0.5*hp/TR
    # TSS = motionconfounds(motparam, sigma, TR)

    TSS = motparam

    out = []
    C1 = np.corrcoef(ts.T, TSS.T)
    C1 = np.abs(C1[0, 1:])
    out += [np.max(C1[:6])]
    out += [np.max(C1[6:])]
    out += [np.max(out)]

    eps = np.finfo(float).eps
    regModel = np.dot(np.linalg.pinv(np.concatenate((np.ones((ts.shape[0], 1)), TSS), axis=1)), ts) / (eps+np.std(ts))
    regModel = np.sort(np.abs(regModel[1:]))

    n = int(np.floor(regModel.shape[0]/2))
    out += [regModel[-2], regModel[-1], np.mean(regModel[n-1:])]

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

    n = np.max(S.shape)-1
    Sx = np.sum(S[:-1])
    Sy = np.sum(S[1:])
    Sxx = np.sum(S[:-1]**2)
    Sxy = np.sum(S[:-1]*S[1:])
    Syy = np.sum(S[1:]**2)

    mu = (Sy*Sxx - Sx*Sxy) / (n*(Sxx - Sxy) - (Sx**2 - Sx*Sy))
    lmbda = -(1/deltat)*np.log((Sxy - mu*Sx - mu*Sy + n*mu**2) / (Sxx-2*mu*Sx+n*mu**2))

    alpha = np.exp(-lmbda*deltat)
    alpha2 = np.exp(-2*lmbda*deltat)
    sigmahat2 = (1/n)*(Syy - 2*alpha*Sxy + alpha2*Sxx - 2*mu*(1-alpha)*(Sy-alpha*Sx) + n*mu**2*(1-alpha)**2)

    sigma = np.sqrt(sigmahat2*2*lmbda/(1-alpha2))

    return mu, sigma, lmbda


def oupjk(S, deltat):
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

    m = 2  # number of partitions
    partlength = int(np.floor(np.max(S.shape)/m))
    Spart = np.zeros((m, partlength))
    for i in np.arange(m):
        Spart[i, :] = S[partlength*i:partlength*(i+1)]

    [muT, sigmaT, lambdaT] = functionoupmle(S, deltat)

    # % Calculate the individual partitions.
    mupart = np.zeros(m)
    sigmapart = np.zeros(m)
    lambdapart = np.zeros(m)
    for i in np.arange(m):
        mupart[i], sigmapart[i], lambdapart[i] = functionoupmle(np.ravel(Spart[i,:]), deltat)
    # for i=1:1:m
    #     [mupart(i), sigmapart(i), lambdapart(i)] = functionoupmle(Spart(i,:), deltat);
    # end
    # % Now the jacknife calculation.
    lmbda = (m/(m-1))*lambdaT - (np.nansum(lambdapart))/(m**2-m)
    # % mu and sigma are not biased, so there's no real need for the jackknife.
    # % But we do it anyway for demonstration purposes.
    # mu = (m/(m-1))*muT - (np.sum(mupart))/(m**2-m)
    sigma = (m/(m-1))*sigmaT - (np.nansum(sigmapart))/(m**2-m)
    # features =  [sigma lambda];

    return [sigma, lmbda]


def clusterdist(icmap, xSiz, icThreshold):
    # % this feature tries to quantify the number of clusters in a component

    cmd = ['cluster', '-i', icmap.get_filename(), '-t', str(icThreshold)]
    clst = np.loadtxt(io.StringIO(util.run(cmd)), skiprows=1, delimiter='\t')

    if clst.shape[0] > 0:
        X = clst[:, 1]
        X = X[X > 4]
        if X.shape[0] < 3:
            X = np.concatenate((X, np.zeros(3-X.shape[0])))
        Y = np.sort(X)[::-1] * xSiz[0] * xSiz[1] * xSiz[2]
        feat = [X.shape[0], np.mean(X)-np.median(X), np.max(X),
                np.var(X, ddof=1), scipy.stats.skew(X),
                scipy.stats.kurtosis(X, fisher=False), Y[0], Y[1], Y[2]]
    else:
        feat = [0, 0, 0, 0, 0, 1.5, 0, 0, 0]

    return feat


def negativevspositive(icmap, icmap_abs, icThreshold):
    out = []

    out += [float(util.run(['fslstats', icmap.get_filename(), '-E']))]
    out += [float(util.run(['fslstats', icmap_abs.get_filename(), '-E']))]

    a = float(util.run(['fslstats', icmap.get_filename(), '-M']))
    aS = float(util.run(['fslstats', icmap.get_filename(), '-S']))

    b = float(util.run(['fslstats', icmap_abs.get_filename(), '-M']))
    bS = float(util.run(['fslstats', icmap_abs.get_filename(), '-S']))

    eps = np.finfo(float).eps

    out += [np.abs(a)/(eps+np.abs(b))]
    out += [np.abs(a*bS)/np.abs(eps+aS*b)]

    # % diff (for abs of pos and neg voxels)
    img = icmap.get_data()
    out += [np.abs(np.sum(img > 0)-np.sum(img < 0))/(eps+np.sum(img > 0))]
    out += [np.abs(np.sum(img > icThreshold)-np.sum(img < -icThreshold))/(eps+np.sum(img > icThreshold))]

    return out


def zstattofuncratio(icmap, func_mean):
    # % compare high zstat voxels against original EPI mean image intensities
    # % (in some artefacts the mean FMRI image is dark)
    # % Similar to above - compare zstat against original EPI *edge* image
    # % and/or the original PCA residual image
    # % global icThreshold;

    workdir = op.dirname(icmap.get_filename())

    img_ratio = op.join(workdir, 'dummyd')
    util.run(['fslmaths', icmap.get_filename(), '-s', '5', '-abs', '-div', func_mean.get_filename(), img_ratio])
    img_ratio = nb.load(img_ratio+'.nii.gz').get_data()

    img_mult = op.join(workdir, 'dummym')
    util.run(['fslmaths', icmap.get_filename(), '-s', '5', '-abs', '-mul', func_mean.get_filename(), img_mult])
    img_mult = nb.load(img_mult+'.nii.gz').get_data()

    a = np.logical_not(np.isnan(img_mult)) & np.logical_not(np.isinf(img_mult))
    dmp = img_mult[(img_mult > 0) & a]

    a = np.logical_not(np.isnan(img_ratio)) & np.logical_not(np.isinf(img_ratio))
    drp = img_ratio[(img_ratio > 0) & a]

    return list(np.concatenate((np.nanpercentile(dmp, [99, 95]), np.nanpercentile(drp, [99, 95]))))


def slicewisestats(icmap_abs, ic_threshold):
    # % all-of-a-single-slice and none-of-others or every odd (or even) slices -
    # % effects; the above also tend to show up as VERY sparse or oddly
    # % nonGaussian in time domain

    out = []
    eps = np.finfo(float).eps

    icmap_abs = icmap_abs.get_data()

    sumSlc = np.zeros(icmap_abs.shape[2])
    sumTot = np.sum(icmap_abs**2)+eps

    for i in range(icmap_abs.shape[2]):
        sumSlc[i] = np.sum(icmap_abs[:, :, i]**2)/sumTot

    out += [np.sum(sumSlc > 0.15)]
    out += [np.max(sumSlc)]

    sumSlc = np.zeros(icmap_abs.shape[2])
    sumTot = np.sum(icmap_abs > ic_threshold)+eps

    for i in range(icmap_abs.shape[2]):
        sumSlc[i] = np.sum(icmap_abs[:, :, i] > ic_threshold)/sumTot

    out += [np.sum(sumSlc > 0.15)]
    out += [np.max(sumSlc)]

    return out


def everynthvarianve(icmap_abs, icThreshold):
    # % all-of-a-single-slice and none-of-others or every odd (or even) slices -
    # % effects; the above also tend to show up as VERY sparse or oddly
    # % nonGaussian in time domain (this is like F4 in Podrack 08)

    out = []
    eps = np.finfo(float).eps

    icmap_abs = icmap_abs.get_data()

    N = icmap_abs.shape[2]
    varSlc = np.zeros(N)

    for i in range(N):
        varSlc[i] = np.sum(icmap_abs[:,:,i]**2)

    out += [np.abs(np.sum(varSlc[0::2])-np.sum(varSlc[1::2]))/(eps+np.sum(varSlc))]

    ##

    AA = np.concatenate((np.arange(0, N, 4), np.arange(1, N, 4)))
    BB = np.concatenate((np.arange(2, N, 4), np.arange(3, N, 4)))

    out += [np.abs(np.sum(varSlc[AA])-np.sum(varSlc[BB]))/(eps+np.sum(varSlc))]

    ##

    varSlc = np.zeros(N)
    icmap_abs = icmap_abs * (icmap_abs > icThreshold)

    for i in range(N):
        varSlc[i] = np.var(icmap_abs[:,:,i])

    out += [np.abs(np.sum(varSlc[0::2])-np.sum(varSlc[1::2]))/(eps+np.sum(varSlc))]

    ##

    out += [np.abs(np.sum(varSlc[AA])-np.sum(varSlc[BB]))/(eps+np.sum(varSlc))]

    return out


def masktscorrandoverlap(icmap_abs, ts, seg, ts_gm, ts_wm, ts_csf, ic_threshold):
    # % In this function, given the time series, the mean CSF time-series is
    # % estimated and then its correlation with the given ts is calc'd
    # % Next, the overlap of spatial map w/each and everyone of these masks is
    # % estimated

    out = []
    eps = np.finfo(float).eps

    clf = linear_model.LinearRegression()

    ts = ts / scipy.std(ts)
    N = ts.shape[0]

    X = np.concatenate((
        np.zeros((N, 1)),
        (ts_wm/(eps+np.std(ts_wm)))[:, np.newaxis],
        (ts_gm/(eps+np.std(ts_gm)))[:, np.newaxis],
        (ts_csf/(eps+np.std(ts_csf)))[:, np.newaxis]), axis=1)

    clf.fit(X, ts)
    out += list(np.abs(clf.coef_[1:]))

    seg = seg.get_data()
    icmap_abs = icmap_abs.get_data()

    denom = eps+np.sum(icmap_abs)
    for i in range(1, 4):
        out += [np.sum(icmap_abs*(seg == i))/denom]

    icmap_abs_th = icmap_abs > ic_threshold
    denom = eps+np.sum(icmap_abs_th)
    for i in range(1, 4):
        out += [np.sum(icmap_abs_th*(seg == i))/denom]

    wch = icmap_abs > ic_threshold
    denom = eps+np.sum(icmap_abs[wch])
    for i in range(1, 4):
        out += [np.sum(icmap_abs[wch]*(seg[wch] == i))/denom]

    return out


def smoothest(icmap, func_brain_mask):
    # % closely related: spatial smoothness (resel size etc) of zstat map
    # % spatial entropy (histogram) / parameters from histogram mixture-model fit
    # % smoothness

    smth = util.run(['smoothest', '-z', icmap.get_filename(), '-m', func_brain_mask.get_filename()])
    smth = smth.split('\n')
    for l in smth:
        if l.startswith('RESELS'):
            out = float(l.split(' ')[1])
    return [out]


def maxtfce(icmap, ic_threshold):
    # % test for "clusterness" / "compactness" - e.g. feed through TFCE and
    # % take the max (maybe pre-TFCE normalise e.g. by max(abs(z)))
    std = util.run(['fslstats', icmap.get_filename(), '-S'])

    out = []

    icmap2 = op.join(op.dirname(icmap.get_filename()), 'dummy2')
    util.run(['fslmaths', icmap.get_filename(), '-abs', '-div', std, '-tfce', '2', '0.5', '6', icmap2])
    out += [float(util.run(['fslstats', icmap2, '-P', '100']))]

    util.run(['fslmaths', icmap.get_filename(), '-abs', '-tfce', '2', '0.5', '6', icmap2])
    out += [float(util.run(['fslstats', icmap2, '-P', '100']))]

    util.run(['fslmaths', icmap.get_filename(), '-abs', '-thr', str(ic_threshold), '-tfce', '2', '0.5', '6', icmap2])
    out += [float(util.run(['fslstats', icmap2, '-P', '100']))]

    return out


def edgemasks(icmap, icmap_abs, edge_masks, ic_threshold):
    # % compare high zstat voxels against original EPI mean image intensities
    # % (in some artefacts the mean FMRI image is dark)
    # % Similar to above - compare zstat against original EPI *edge* image
    # % and/or the original PCA residual image

    out = []

    eps = np.finfo(float).eps

    icmap = np.abs(icmap.get_data()) > ic_threshold
    icmap_abs = icmap_abs.get_data()

    for msk in edge_masks:
        msk = msk.get_data()
        whatPercentOnEdgeB = np.sum(icmap*msk)/(eps+np.sum(icmap))
        whatPercentOfEdge = np.sum(icmap*msk)/(eps+np.sum(msk))

        wch = icmap_abs > ic_threshold
        whatPercentOnEdgeC = np.sum(icmap_abs[wch]*msk[wch])/(eps+np.sum(icmap_abs[wch]))

        out += [whatPercentOnEdgeB, whatPercentOnEdgeC, whatPercentOfEdge]

    return out


def sagmasks(icmap_abs, sag_masks, ic_threshold):
    # % look at intersections of ICA spatial map with major vein mask images

    out = []

    icmap_abs = icmap_abs.get_data()

    icmap_abs2 = icmap_abs > ic_threshold
    wch = icmap_abs > ic_threshold

    sag_masks = np.asarray(sag_masks)

    for i in range(sag_masks.shape[0]):
        for j in range(sag_masks.shape[1]):
            smask = sag_masks[i, j].get_data()

            whatPercentOnEdgeB = np.sum(icmap_abs2*smask) / np.maximum(1, np.sum(icmap_abs2))
            whatPercentOfEdge = np.sum(icmap_abs2*smask) / np.maximum(1, np.sum(smask))
            whatPercentOnEdgeC = np.sum(icmap_abs[wch]*smask[wch])/np.maximum(1, np.sum(icmap_abs[wch]))

            grot1 = np.concatenate((smask.reshape(1, -1), icmap_abs2.reshape(1, -1), icmap_abs.reshape(1, -1)), axis=0)
            grot1 = np.corrcoef(grot1)
            grot1[np.isnan(grot1)] = 0

            grot2 = np.concatenate((smask[wch].reshape(1, -1), icmap_abs[wch].reshape(1, -1)), axis=0)
            grot2 = np.corrcoef(grot2)
            grot2[np.isnan(grot2)] = 0

            out += [whatPercentOnEdgeB, whatPercentOnEdgeC, whatPercentOfEdge, grot1[0, 1], grot1[0, 2], grot2[0, 1]]

    return out


def stripe(icmap):

    dm1 = op.join(op.dirname(icmap.get_filename()), 'dummy01')
    util.run(['fslmaths', icmap.get_filename(), '-s', '2', '-abs', dm1])
    util.run(['fslmaths', icmap.get_filename(), '-abs', '-s', '2', '-sub', dm1, '-abs', dm1])
    p95 = float(util.run(['fslstats', dm1, '-P', '95']))

    util.run(['imrm', dm1])

    return [p95]
