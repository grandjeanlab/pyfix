#!/usr/bin/env python
# from fsl.data import fixlabels
import logging
# import os.path as op
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm as svm
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

from pyfix import io
from pyfix import legacy

log = logging.getLogger(__name__)


def classify(feature_table, clf):
    # TODO: these cols should probably get dropped prior to classify
    drop_cols = [
        'nic',
        *[f'pixdim:{i}' for i in np.arange(4)],
        *[f'dim:{i}' for i in np.arange(4)],
    ]

    X = feature_table.drop(columns=drop_cols, errors='ignore').values
    y_pred = clf.predict(X)

    try:
        y_pred_pb = clf.predict_proba(X)
    except:
        y_pred_pb = None

    return y_pred, y_pred_pb


def collate_features(table: Union[str, pd.DataFrame], legacy_features=False, group_by=None):
    if isinstance(table, str):
        table = io.read_file_table(table)

    assert {'features', 'labels'}.issubset(set(table.columns)), \
        'file table must contains columns called features and labels'

    if group_by is not None:
        group_vals = table[group_by]
    else:
        group_vals = table.index.values

    N = table.shape[0]
    train_data = [None] * N
    for idx, (f0, l0, g0) in enumerate(zip(table.features, table.labels, group_vals)):

        if legacy_features:
            train_data[idx] = legacy.read_features(f0)
        else:
            train_data[idx] = io.read_features(f0)

        train_data[idx]['group'] = g0

        l1 = np.zeros(train_data[idx].shape[0])
        for line in open(l0, "r"):
            ln = line
        ln = ln.strip().strip("[]")
        l1[np.fromstring(ln, dtype=int, sep=",") - 1] = 1

        train_data[idx]['label'] = l1

    train_data = pd.concat(train_data).astype({'label': bool})

    # invert train_data so that signal==True and noise==False
    train_data['label'] = ~train_data['label']

    return train_data


def train(table: pd.DataFrame,
          clf=None,
          loo=False):
    # TODO: these cols should probably get dropped prior to train
    drop_cols = [
        'nic',
        *[f'pixdim:{i}' for i in np.arange(4)],
        *[f'dim:{i}' for i in np.arange(4)],
        'label',
        'group',
    ]

    # TODO: let user define the ylabel and group cols
    X = table.drop(columns=drop_cols, errors='ignore').values
    y = table['label'].astype(int).values
    group = table['group'].values

    # y = preprocessing.OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    # print(y)

    if clf is None:
        clf = svm.SVC(probability=True, C=1, gamma='scale', class_weight="balanced")
        # clf = ensemble.RandomForestClassifier()

    y_pred, y_pred_pb, test_idx = None, None, None

    if loo:
        y_pred, y_pred_pb, test_idx = _cv_loo(clf, X, y, group, n_jobs=-2)

    return clf.fit(X, y), {'actual': y, 'predicted': y_pred, 'predicted_pb': y_pred_pb, 'idx': test_idx}


def _cv_loo(clf, X, y, groups, n_jobs=1):
    loo = LeaveOneGroupOut()

    pipe = pipeline.make_pipeline(preprocessing.StandardScaler(), clf)

    par = Parallel(n_jobs=n_jobs)
    blocks = par(delayed(__fit_and_predict)(clone(pipe), X, y, train_idx, test_idx)
                 for train_idx, test_idx in loo.split(X, y, groups))

    y_pred = np.concatenate([y0 for y0, _ in blocks])
    y_pred_pb = np.concatenate([ypb for _, ypb in blocks])

    test_idx = [tst for _, tst in loo.split(X, y, groups)]
    test_idx_cat = np.concatenate(test_idx)

    y_pred = y_pred[np.argsort(test_idx_cat)]
    y_pred_pb = y_pred_pb[np.argsort(test_idx_cat), :]

    return y_pred, y_pred_pb, test_idx


def __fit_and_predict(estimator, X, y, train_idx, test_idx):
    X_trn, X_tst, y_trn, y_tst = X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]
    estimator.fit(X_trn, y_trn)

    y_pred = estimator.predict(X_tst)

    try:
        y_pred_pb = estimator.predict_proba(X_tst)
    except:
        y_pred_pb = None

    # cm = metrics.confusion_matrix(y_tst, y_pred)

    return y_pred, y_pred_pb
