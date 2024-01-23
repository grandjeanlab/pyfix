#!/usr/bin/env python

from   typing   import List
import                 csv
import                 io
import                 logging
import os.path  as     op
import pandas   as     pd
import numpy    as     np
from   fsl.data import fixlabels


log = logging.getLogger(__name__)


# Feature extractors which return multiple values will
# have an index appended to their name, e.g.:
#
#     @feature_extractor
#     def myfeature():
#         return [1,2,3]
#
# will result in myfeature:0, myfeature:1, myfeature:2
LEGACY_FEATURE_LABELS = [
    'nic',
    *[f'arvswgn:{i}' for i in np.arange(2)],
    *[f'arfull:{i}' for i in np.arange(5)],
    'skewness',
    'kurtosis',
    'mean_median',
    *[f'entropy:{i}' for i in np.arange(2)],
    *[f'tsjump:{i}' for i in np.arange(6)],
    *[f'fftcoarse:{i}' for i in np.arange(4)],
    *[f'fftfiner:{i}' for i in np.arange(7)],
    *[f'fftfinerwrtnull:{i}' for i in np.arange(8)],
    *[f'motioncorrelation:{i}' for i in np.arange(6)],
    *[f'oupjk:{i}' for i in np.arange(2)],
    *[f'clusterdist:{i}' for i in np.arange(9)],
    *[f'negativevspositive:{i}' for i in np.arange(6)],
    *[f'zstattofuncratio:{i}' for i in np.arange(4)],
    *[f'slicewisestats:{i}' for i in np.arange(4)],
    *[f'everynthvariance:{i}' for i in np.arange(4)],
    *[f'masktscorrandoverlap:{i}' for i in np.arange(12)],
    'smoothest:0',
    'spatialoverlap',
    *[f'maxtfce:{i}' for i in np.arange(3)],
    *[f'edgemasks:{i}' for i in np.arange(15)],
    *[f'sagmasks:{i}' for i in np.arange(72)],
    'stripe:0',
    *[f'dim:{i}' for i in np.arange(4)],
    *[f'pixdim:{i}' for i in np.arange(4)],
]

LEGACY_FEATURE_EXTRACTORS = list(np.unique([f.split(':')[0] for f in LEGACY_FEATURE_LABELS]))


def read_features(path: str, columns : List[str] = None):
    """Read legacy FIX features from file.

    Args:
        path (str):     Path to feature file
        columns (list): List of column names - defaults to
                        LEGACY_FEATURE_LABELS

    Returns:
        features table (pd.DataFrame):
    """

    with open(path, 'rt') as f:
        data = f.read()

    # do we have column names?
    header = csv.Sniffer().has_header(data)

    # If columns have been provided, we assume
    # that the user knows what they're doing.
    if columns is not None:
        header = 0 if header else None

    else:
        # Otherwise, if the file has no header,
        # we assume it has FEATURE_LABELS columns
        if not header:
            columns = LEGACY_FEATURE_LABELS
            header  = None

        # otherwise the file contains column labels
        else:
            header = 0

    df = pd.read_csv(io.StringIO(data), header=header, names=columns)

    for column in df.columns:
        if column not in LEGACY_FEATURE_LABELS:
            log.warning(f'Unrecognised feature column in {path}: {column}')
    for column in LEGACY_FEATURE_LABELS:
        if column not in df.columns:
            log.warning(f'Missing feature column in {path}: {column}')

    return df


def save_features(path : str, features : pd.DataFrame):
    """
    Save legacy FIX features to file.
    """

    features[LEGACY_FEATURE_LABELS].to_csv(
        path, index=None, header=None)


def generate_file_table(*args):
    """Generate a table of files from legacy FIX folders.

    Args:
        *args:

    Returns:

    """

    d = pd.DataFrame(
        {
            'features': [op.join(f, 'fix/features.csv') for f in args],
            'labels': [op.join(f, 'hand_labels_noise.txt') for f in args],
            'subid': [i for i, _ in enumerate(args)]
        }
    )

    return d


def generate_label_filename(meldir, modelfile, threshold):
    model     = op.splitext(op.basename(modelfile))[0]
    threshold = int(threshold)
    return op.join(meldir, f'fix4melview_{model}_thr{threshold}.txt')


def read_labels(filename: str, nics : int = None):
    """Read legacy label files, melview and hand_labels_noise.txt.

    If nics is None, a sequence of one-indexed noise IC labels is returned.
    Otherwise, a sequence with a value for each IC is returned, containing
    1 for noise ICs, and 0 for signal ICs.
    """

    indices = fixlabels.loadLabelFile(filename, returnIndices=True)[2]
    indices = np.array(indices, dtype=int)

    if nics is None:
        return indices
    else:
        labels = np.zeros(nics)
        labels[indices - 1] = 1

    return labels


def prob_threshold(pb, threshold, col=1):
    """
    Threshold probablistic classifications.

    probability[:, col] > thres

    Lower (conservative) threshold will result in more outputs being included
    """
    return pb[:, col] > threshold
