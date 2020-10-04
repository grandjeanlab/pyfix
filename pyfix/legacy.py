#!/usr/bin/env python
import pandas as pd
import numpy as np
import os.path as op

FEATURE_LABELS = [
    'nic',
    *[f'arvswgn:{i}' for i in np.arange(2)],
    *[f'arfull:{i}' for i in np.arange(5)],
    'skewness',
    'kurtosis',
    'mean-median',
    *[f'entropy:{i}' for i in np.arange(2)],
    *[f'tsjump:{i}' for i in np.arange(6)],
    *[f'fftcoarse:{i}' for i in np.arange(4)],
    *[f'fftfiner:{i}' for i in np.arange(7)],
    *[f'fftfinerwrtnull:{i}' for i in np.arange(8)],
    *[f'motioncorrelation:{i}' for i in np.arange(6)],
    *[f'abs_oupjk:{i}' for i in np.arange(2)],
    *[f'clusterdist:{i}' for i in np.arange(9)],
    *[f'negativevspositive:{i}' for i in np.arange(6)],
    *[f'zstattofuncratio:{i}' for i in np.arange(4)],
    *[f'slicewisestats:{i}' for i in np.arange(4)],
    *[f'everynthvariance:{i}' for i in np.arange(4)],
    *[f'masktscorrandoverlap:{i}' for i in np.arange(12)],
    'smoothest',
    'spatialoverlap',
    *[f'maxtfce:{i}' for i in np.arange(3)],
    *[f'edgemasks:{i}' for i in np.arange(15)],
    *[f'sagmasks:{i}' for i in np.arange(72)],
    'stripe',
    *[f'dim:{i}' for i in np.arange(4)],
    *[f'pixdim:{i}' for i in np.arange(4)],
]


def read_features(filename: str):
    """Read legacy FIX features from file.

    Args:
        filename (str):

    Returns:
        features table (pd.DataFrame):
    
    """
    return pd.read_csv(filename, header=None, names=FEATURE_LABELS)


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
