#!/usr/bin/env python

import base64
from tempfile import TemporaryFile

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
import importlib
from typing import List
from typing import Union

import jinja2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn import pipeline, metrics
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.base import clone
# from sklearn.inspection import _weights_scorer
from sklearn.metrics import check_scoring
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import check_random_state

from pyfix import io, legacy
from pyfix.util import weak_import

log = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, m=None, labels=None):
    if m is None:
        m = {
            'confusion_matrix': lambda y, y_pred: metrics.confusion_matrix(y, y_pred, labels=labels),
            'balanced_accuracy_score': metrics.balanced_accuracy_score,
            'matthews_corrcoef': metrics.matthews_corrcoef,
        }

    return {k: f(y_true, y_pred) for k, f in m.items()}


def calculate_loo_metrics(y, y_pred_pb, test_idx, threshold=None, labels=None):

    if threshold is None:
        threshold = 0.5

    if np.isscalar(threshold):
        threshold = [threshold]

    if labels is None:
        labels = np.arange(y_pred_pb.shape[1])

    records = []

    for thr0 in threshold:

        thr0 = np.around(thr0, 3)

        y_pred = legacy.prob_threshold(y_pred_pb, threshold=thr0)

        for fold, idx in enumerate(test_idx):
            m = calculate_metrics(y[idx], y_pred[idx], labels=labels)
            m['fold'] = fold
            m['threshold'] = thr0

            m['tpr'], m['tnr'], m['(3*tpr+tnr)/4'] = legacy.fix_metric(m['confusion_matrix'])

            records += [m]

    df = pd.DataFrame(records)

    return df


def classify(feature_table, model):
    # TODO: these cols should probably get dropped prior to classify
    drop_cols = [
        'nic',
        *[f'pixdim:{i}' for i in np.arange(4)],
        *[f'dim:{i}' for i in np.arange(4)],
    ]

    X = feature_table.drop(columns=drop_cols, errors='ignore').values
    y_pred = model.predict(X)

    try:
        y_pred_pb = model.predict_proba(X)
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

        labels = legacy.read_labels(l0, train_data[idx].shape[0])

        train_data[idx]['label'] = labels

    train_data = pd.concat(train_data).astype({'label': bool})

    # invert train_data so that signal==True and noise==False
    train_data['label'] = ~train_data['label']

    return train_data


def create_model(model=None):
    """Create the xgboost or scikit learn classifier that is used for training
    and classification. By default a XGBoostClassifier is created,

    The classifier type can be specified via the "PYFIX_MODEL" environment
    variable, e.g.

        PYFIX_MODEL="sklearn.ensemble.RandomForestClassifier"

    Initialisation parameters can be set via "PYFIX_MODEL_<parameter>"
    environment variables, e.g:

        PYFIX_MODEL_n_estimators=100
    """
    if model is None:
        model = os.environ.get('PYFIX_MODEL', None)

    if model is None:
        model = 'xgboost.XGBClassifier'

    if model.startswith('xgboost') and weak_import('xgboost') is None:
        model = 'sklearn.ensemble.RandomForestClassifier'
        log.warning(f'xgboost is not installed - falling back to {model}')

    params = {k : v for k, v in os.environ.items() if k.startswith('PYFIX_MODEL_')}
    params = {k.lstrip('PYFIX_NODEL_') : v for k, v in params.items()}

    log.info(f'Using {model}')

    mod, cls = model.rsplit('.', 1)
    mod      = importlib.import_module(mod)
    cls      = getattr(mod, cls)

    return cls(**params)


def train(table: pd.DataFrame,
          model=None,
          loo=False,
          n_jobs=None):
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

    model = create_model(model)

    feature_labels = list(table.drop(columns=drop_cols, errors='ignore'))

    output = (model.fit(X, y),)

    if loo:
        y_pred, y_pred_pb, test_idx, trained_models = _cv_loo(model, X, y, group, n_jobs=n_jobs)
        output += ((y, y_pred_pb, test_idx),)

    return output


def _cv_loo(model, X, y, groups, n_jobs=1):
    loo = LeaveOneGroupOut()

    pipe = pipeline.make_pipeline(preprocessing.StandardScaler(), model)

    par = Parallel(n_jobs=n_jobs)
    blocks = par(delayed(__fit_and_predict)(clone(pipe), X, y, train_idx, test_idx)
                 for train_idx, test_idx in loo.split(X, y, groups))

    trained_models = [m for _, _, m in blocks]

    y_pred = np.concatenate([y0 for y0, _, _ in blocks])
    y_pred_pb = np.concatenate([ypb for _, ypb, _ in blocks])

    test_idx = [tst for _, tst in loo.split(X, y, groups)]
    test_idx_cat = np.concatenate(test_idx)

    y_pred = y_pred[np.argsort(test_idx_cat)]

    y_pred_pb = y_pred_pb[np.argsort(test_idx_cat), :]

    return y_pred, y_pred_pb, test_idx, trained_models


def __fit_and_predict(model, X, y, train_idx, test_idx):
    X_trn, X_tst, y_trn, y_tst = X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]

    # fit model

    model.fit(X_trn, y_trn)

    # predict class, and class probabilities (where possible)

    y_pred = model.predict(X_tst)

    try:
        y_pred_pb = model.predict_proba(X_tst)
    except:
        y_pred_pb = None

    return y_pred, y_pred_pb, deepcopy(model)
