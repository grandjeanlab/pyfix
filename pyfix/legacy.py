#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import os.path as op
import logging

log = logging.getLogger(__name__)

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


def read_features(filename: str):
    """Read legacy FIX features from file.

    Args:
        filename (str):

    Returns:
        features table (pd.DataFrame):
    
    """
    return pd.read_csv(filename, header=None, names=LEGACY_FEATURE_LABELS)


def save_features(path, features):
    """
    Save legacy FIX features to file.
    """

    filename = op.join(path, 'features.csv')

    if op.exists(filename):
        log.warning(f'Features file already exists: {filename}')

    workdir = op.expanduser(op.dirname(filename))
    if not op.exists(workdir):
        os.makedirs(workdir)

    log.info(f'Saving legacy fix features to: {filename}')
    features[LEGACY_FEATURE_LABELS].to_csv(filename, index=None, header=None)


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

def read_labels(filename: str):
    """Read legacy label files, melview and hand_labels_noise.txt"""

    with open(filename, 'r') as f:
        labels = f.readlines()[-1]

    labels = np.fromstring(labels.strip('[]\n'), sep=',', dtype=int)

    return labels
