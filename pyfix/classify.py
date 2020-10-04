#!/usr/bin/env python
"""Apply trained FIX model to classify ICs."""
import numpy as np

from pyfix import legacy


def classify(features, clf):
    table = legacy.read_features(features)

    drop_cols = [
        'nic',
        *[f'pixdim:{i}' for i in np.arange(4)],
        *[f'dim:{i}' for i in np.arange(4)],
    ]

    X = table.drop(columns=drop_cols).values

    Y = clf.predict(X)

    # Y = np.where(Y == 1)[0] + 1

    return Y


