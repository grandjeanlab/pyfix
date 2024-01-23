#!/usr/bin/env python

from   typing   import List
import                 csv
import                 io
import                 logging
import os.path  as     op
import pandas   as     pd
import numpy    as     np
from   fsl.data import fixlabels
from   pyfix    import (__version__,
                        feature)


log = logging.getLogger(__name__)


# List of feature labels produced by the old
# R/MATLAB-based FIX, and their corresponding
# pyfix label (as defined in pyfix/features.py)
LEGACY_FEATURE_LABELS = {
    'number.of.ics'                      : 'nic',
    'ar.drop.slope'                      : 'arvswgn:0',
    'ar.drop.offset'                     : 'arvswgn:1',
    'ar1.noise.var'                      : 'arfull:0',
    'ar1.param'                          : 'arfull:1',
    'ar2.noise.var'                      : 'arfull:2',
    'ar2.param1'                         : 'arfull:3',
    'ar2.param2'                         : 'arfull:4',
    'ts.skewness'                        : 'skewness',
    'ts.kurtosis'                        : 'kurtosis',
    'ts.mean.min.median'                 : 'mean_median',
    'ts.entropy1'                        : 'entropy:0',
    'ts.entropy2'                        : 'entropy:1',
    'ts.jump.max.abs.diff'               : 'tsjump:0',
    'ts.jump.max.abs.diff.std'           : 'tsjump:1',
    'ts.jump.max.abs.diff.std.len'       : 'tsjump:2',
    'ts.jump.max.abs.diff.std.localmean' : 'tsjump:3',
    'ts.jump.max.abs.diff.std.localsum'  : 'tsjump:4',
    'ts.xcorr'                           : 'tsjump:5',
    'fft.coarse.10'                      : 'fftcoarse:0',
    'fft.coarse.15'                      : 'fftcoarse:1',
    'fft.coarse.20'                      : 'fftcoarse:2',
    'fft.coarse.25'                      : 'fftcoarse:3',
    'fft.finer.01'                       : 'fftfiner:0',
    'fft.finer.025'                      : 'fftfiner:1',
    'fft.finer.05'                       : 'fftfiner:2',
    'fft.finer.10'                       : 'fftfiner:3',
    'fft.finer.15'                       : 'fftfiner:4',
    'fft.finer.20'                       : 'fftfiner:5',
    'fft.finer.25'                       : 'fftfiner:6',
    'fft.H0r.01'                         : 'fftfinerwrtnull:0',
    'fft.H0r.025'                        : 'fftfinerwrtnull:1',
    'fft.H0r.05'                         : 'fftfinerwrtnull:2',
    'fft.H0r.10'                         : 'fftfinerwrtnull:3',
    'fft.H0r.15'                         : 'fftfinerwrtnull:4',
    'fft.H0r.20'                         : 'fftfinerwrtnull:5',
    'fft.H0r.25'                         : 'fftfinerwrtnull:6',
    'fft.H0r'                            : 'fftfinerwrtnull:7',
    'mcorr.max'                          : 'motioncorrelation:0',
    'mcorr.max.grad'                     : 'motioncorrelation:1',
    'mcorr.max.max'                      : 'motioncorrelation:2',
    'mcorr.reg.2nd.max'                  : 'motioncorrelation:3',
    'mcorr.reg.max'                      : 'motioncorrelation:4',
    'mcorr.reg.mean'                     : 'motioncorrelation:5',
    'ts.oup.sigma'                       : 'oupjk:0',
    'ts.oup.lambda'                      : 'oupjk:1',
    'cluster.num'                        : 'clusterdist:0',
    'cluster.mean.min.median'            : 'clusterdist:1',
    'cluster.max'                        : 'clusterdist:2',
    'cluster.var'                        : 'clusterdist:3',
    'cluster.skew'                       : 'clusterdist:4',
    'cluster.kurt'                       : 'clusterdist:5',
    'cluster.max1'                       : 'clusterdist:6',
    'cluster.max2'                       : 'clusterdist:7',
    'cluster.max3'                       : 'clusterdist:8',
    'negpos.entropy'                     : 'negativevspositive:0',
    'negpos.abs.entropy'                 : 'negativevspositive:1',
    'negpos.mean.ratio'                  : 'negativevspositive:2',
    'negpos.stdmean.ratio'               : 'negativevspositive:3',
    'negpos.ratio'                       : 'negativevspositive:4',
    'negpos.ratio.thresh'                : 'negativevspositive:5',
    'zxfunc.99prctl'                     : 'zstattofuncratio:0',
    'zxfunc.95prctl'                     : 'zstattofuncratio:1',
    'z2func.99prctl'                     : 'zstattofuncratio:2',
    'z2func.95prctl'                     : 'zstattofuncratio:3',
    'num.slice.bt.15prcnt'               : 'slicewisestats:0',
    'max.slice.prcnt'                    : 'slicewisestats:1',
    'thr.num.slice.bt.15prcnt'           : 'slicewisestats:2',
    'thr.max.slice.prcnt'                : 'slicewisestats:3',
    'every.2nd.single.var'               : 'everynthvariance:0',
    'every.2nd.pair.var'                 : 'everynthvariance:1',
    'thr.every.2nd.single.var'           : 'everynthvariance:2',
    'thr.every.2nd.pair.var'             : 'everynthvariance:3',
    'wm.ts.coef'                         : 'masktscorrandoverlap:0',
    'gm.ts.coef'                         : 'masktscorrandoverlap:1',
    'csf.ts.coef'                        : 'masktscorrandoverlap:2',
    'prcnt.on.csf'                       : 'masktscorrandoverlap:3',
    'prcnt.on.gm'                        : 'masktscorrandoverlap:4',
    'prcnt.on.wm'                        : 'masktscorrandoverlap:5',
    'thr.prcnt.on.csf'                   : 'masktscorrandoverlap:6',
    'thr.prcnt.on.gm'                    : 'masktscorrandoverlap:7',
    'thr.prcnt.on.wm'                    : 'masktscorrandoverlap:8',
    'thr.both.prcnt.on.csf'              : 'masktscorrandoverlap:9',
    'thr.both.prcnt.on.gm'               : 'masktscorrandoverlap:10',
    'thr.both.prcnt.on.wm'               : 'masktscorrandoverlap:11',
    'smoothness'                         : 'smoothest:0',
    'smoothness.mm'                      : 'spatialoverlap',
    'std.then.tfce.max'                  : 'maxtfce:0',
    'tfce.max'                           : 'maxtfce:1',
    'thr.then.tfce'                      : 'maxtfce:2',
    'percent.on.edge.bin1'               : 'edgemasks:0',
    'percent.on.edge.norm1'              : 'edgemasks:1',
    'percent.of.edge1'                   : 'edgemasks:2',
    'percent.on.edge.bin2'               : 'edgemasks:3',
    'percent.on.edge.norm2'              : 'edgemasks:4',
    'percent.of.edge2'                   : 'edgemasks:5',
    'percent.on.edge.bin3'               : 'edgemasks:6',
    'percent.on.edge.norm3'              : 'edgemasks:7',
    'percent.of.edge3'                   : 'edgemasks:8',
    'percent.on.edge.bin4'               : 'edgemasks:9',
    'percent.on.edge.norm4'              : 'edgemasks:10',
    'percent.of.edge4'                   : 'edgemasks:11',
    'percent.on.edge.bin5'               : 'edgemasks:12',
    'percent.on.edge.norm5'              : 'edgemasks:13',
    'percent.of.edge5'                   : 'edgemasks:14',
    'percent.on.mask01.bin'              : 'sagmasks:0',
    'percent.on.mask01.norm'             : 'sagmasks:1',
    'percent.of.mask01'                  : 'sagmasks:2',
    'cc01a'                              : 'sagmasks:3',
    'cc01b'                              : 'sagmasks:4',
    'cc01c'                              : 'sagmasks:5',
    'percent.on.mask02.bin'              : 'sagmasks:6',
    'percent.on.mask02.norm'             : 'sagmasks:7',
    'percent.of.mask02'                  : 'sagmasks:8',
    'cc02a'                              : 'sagmasks:9',
    'cc02b'                              : 'sagmasks:10',
    'cc02c'                              : 'sagmasks:11',
    'percent.on.mask03.bin'              : 'sagmasks:12',
    'percent.on.mask03.norm'             : 'sagmasks:13',
    'percent.of.mask03'                  : 'sagmasks:14',
    'cc03a'                              : 'sagmasks:15',
    'cc03b'                              : 'sagmasks:16',
    'cc03c'                              : 'sagmasks:17',
    'percent.on.mask11.bin'              : 'sagmasks:18',
    'percent.on.mask11.norm'             : 'sagmasks:19',
    'percent.of.mask11'                  : 'sagmasks:20',
    'cc11a'                              : 'sagmasks:21',
    'cc11b'                              : 'sagmasks:22',
    'cc11c'                              : 'sagmasks:23',
    'percent.on.mask12.bin'              : 'sagmasks:24',
    'percent.on.mask12.norm'             : 'sagmasks:25',
    'percent.of.mask12'                  : 'sagmasks:26',
    'cc12a'                              : 'sagmasks:27',
    'cc12b'                              : 'sagmasks:28',
    'cc12c'                              : 'sagmasks:29',
    'percent.on.mask13.bin'              : 'sagmasks:30',
    'percent.on.mask13.norm'             : 'sagmasks:31',
    'percent.of.mask13'                  : 'sagmasks:32',
    'cc13a'                              : 'sagmasks:33',
    'cc13b'                              : 'sagmasks:34',
    'cc13c'                              : 'sagmasks:35',
    'percent.on.mask21.bin'              : 'sagmasks:36',
    'percent.on.mask21.norm'             : 'sagmasks:37',
    'percent.of.mask21'                  : 'sagmasks:38',
    'cc21a'                              : 'sagmasks:39',
    'cc21b'                              : 'sagmasks:40',
    'cc21c'                              : 'sagmasks:41',
    'percent.on.mask22.bin'              : 'sagmasks:42',
    'percent.on.mask22.norm'             : 'sagmasks:43',
    'percent.of.mask22'                  : 'sagmasks:44',
    'cc22a'                              : 'sagmasks:45',
    'cc22b'                              : 'sagmasks:46',
    'cc22c'                              : 'sagmasks:47',
    'percent.on.mask23.bin'              : 'sagmasks:48',
    'percent.on.mask23.norm'             : 'sagmasks:49',
    'percent.of.mask23'                  : 'sagmasks:50',
    'cc23a'                              : 'sagmasks:51',
    'cc23b'                              : 'sagmasks:52',
    'cc23c'                              : 'sagmasks:53',
    'percent.on.mask31.bin'              : 'sagmasks:54',
    'percent.on.mask31.norm'             : 'sagmasks:55',
    'percent.of.mask31'                  : 'sagmasks:56',
    'cc31a'                              : 'sagmasks:57',
    'cc31b'                              : 'sagmasks:58',
    'cc31c'                              : 'sagmasks:59',
    'percent.on.mask32.bin'              : 'sagmasks:60',
    'percent.on.mask32.norm'             : 'sagmasks:61',
    'percent.of.mask32'                  : 'sagmasks:62',
    'cc32a'                              : 'sagmasks:63',
    'cc32b'                              : 'sagmasks:64',
    'cc32c'                              : 'sagmasks:65',
    'percent.on.mask33.bin'              : 'sagmasks:66',
    'percent.on.mask33.norm'             : 'sagmasks:67',
    'percent.of.mask33'                  : 'sagmasks:68',
    'cc33a'                              : 'sagmasks:69',
    'cc33b'                              : 'sagmasks:70',
    'cc33c'                              : 'sagmasks:71',
    'stripe.patt'                        : 'stripe:0',
    'xlen'                               : 'dim:0',
    'ylen'                               : 'dim:1',
    'zlen'                               : 'dim:2',
    'tlen'                               : 'dim:3',
    'xres'                               : 'pixdim:0',
    'yres'                               : 'pixdim:1',
    'zres'                               : 'pixdim:2',
    'tres'                               : 'pixdim:3'
}


def read_features(path: str, columns : List[str] = None):
    """Read legacy FIX features from file.

    Args:
        path (str):     Path to feature file
        columns (list): List of column names - defaults to
                        FEATURE_LABELS

    Returns:
        features table (pd.DataFrame):
    """

    log.debug(f'Loading feature file {path}')

    with open(path, 'rt') as f:
        data = f.read()

    # do we have column names?
    header = csv.Sniffer().has_header(data)
    ncols  = len(data.split('\n')[0].split(','))

    # If columns have been provided, we assume
    # that the user knows what they're doing.
    if columns is not None:
        header = 0 if header else None

        log.debug('Expecting a custom set of {len(columns)} columns')

    else:
        # Otherwise, if the file has no header,
        # we assume it has FEATURE_LABELS columns
        if not header:
            columns = feature.FEATURE_LABELS
            header  = None
            log.debug(f'Expecting all {len(columns)} '
                      'pyfix {__version__} columns')

        # otherwise the file contains column labels
        else:
            header = 0
            log.debug(f'Expecting a subset of pyfix '
                      '{__version__} columns')

    df = pd.read_csv(io.StringIO(data), header=header, names=columns)

    if columns is None:
        columns = df.columns

    if len(columns) != ncols:
        msg = (f'Feature file {path} contains an unexpected '
               'number of columns! Expected {len(columns)} '
               'columns, but the file contains {ncols} columns')
        log.error(msg)
        raise RuntimeError(msg)

    # Convert legacy FIX labels into pyfix labels
    df = df.rename(columns=LEGACY_FEATURE_LABELS)

    for column in df.columns:
        if column not in feature.FEATURE_LABELS:
            log.warning(f'Unrecognised feature column in {path}: {column}')

    for column in feature.FEATURE_LABELS:
        if column not in df.columns:
            log.warning(f'Missing feature column in {path}: {column}')

    return df


def save_features(path : str, features : pd.DataFrame):
    """
    Save legacy FIX features to file.
    """

    features[feature.FEATURE_LABELS].to_csv(
        path, index=None, header=None)


def prob_threshold(pb, threshold, col=1):
    """
    Threshold probablistic classifications.

    probability[:, col] > thres

    Lower (conservative) threshold will result in more outputs being included
    """
    return pb[:, col] > threshold
