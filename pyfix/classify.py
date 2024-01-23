#!/usr/bin/env python

from tempfile import TemporaryDirectory
import logging
import multiprocessing as mp
import os.path as op
import importlib
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

from pyfix import io, legacy, report
from pyfix.util import weak_import

xgboost = weak_import('xgboost')
optuna  = weak_import('optuna')

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

            m['tpr'], m['tnr'], m['(3*tpr+tnr)/4'] = report.fix_metric(m['confusion_matrix'])

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

    X      = feature_table.drop(columns=drop_cols, errors='ignore').values
    pipe   = create_pipeline(model, X)
    y_pred = pipe.predict(X)

    # not all classifiers support predict_proba
    try:                   y_pred_pb = pipe.predict_proba(X)
    except AttributeError: y_pred_pb = None

    return y_pred, y_pred_pb


def collate_features(
        table   : Union[str, pd.DataFrame],
        columns : List[str] = None
):
    if isinstance(table, str):
        table = io.read_file_table(table)

    assert {'features', 'labels'}.issubset(set(table.columns)), \
        'file table must contains columns called features and labels'

    group_vals = table.index.values

    N = table.shape[0]
    train_data = [None] * N
    for idx, (f0, l0, g0) in enumerate(zip(table.features, table.labels, group_vals)):

        train_data[idx] = legacy.read_features(f0, columns)
        labels          = io.read_labels(l0, train_data[idx].shape[0])

        train_data[idx]['group'] = g0
        train_data[idx]['label'] = labels

    train_data = pd.concat(train_data).astype({'label': bool})

    # invert train_data so that signal==True and noise==False
    train_data['label'] = ~train_data['label']

    columns = list(train_data.columns)
    columns.remove('label')
    columns.remove('group')

    return train_data, columns


def create_model(model_class=None, n_jobs=None, **params):
    """Create the xgboost or scikit learn classifier that is used for training
    and classification. By default a XGBoostClassifier is created,
    """


    if model_class is None:
        model_class = 'xgboost.XGBClassifier'
        pass

    if model_class.startswith('xgboost') and xgboost is None:
        model_class = 'sklearn.ensemble.RandomForestClassifier'
        log.warning('xgboost is not installed - '
                    f'falling back to {model_class}')

    # default parameters
    if len(params) == 0:
        if model_class == 'sklearn.ensemble.RandomForestClassifier':
            params = {'n_estimators' : 50}
        elif model_class == 'xgboost.XGBClassifier':
            params = {'n_estimators' : 50}

    params['n_jobs'] = n_jobs

    mod, cls = model_class.rsplit('.', 1)
    mod      = importlib.import_module(mod)
    cls      = getattr(mod, cls)
    model    = cls(**params)

    return model


def create_pipeline(model, data=None):
    """Creates a sklearn pipeline containing some standard preprocessing
    estimators.  which are run prior to the given classifier model. If data is
    provided, fit(data) is called on the preprocessing estimators.
    """
    # Empirical testing has determined that
    # classification performance is better
    # without any scaling of the input data.
    # If this changes in the future, we can
    # add preprocessing estimators and build
    # a sklearn pipeline within this function.
    return model


def prepare_data_for_training(table):

    drop_cols = [
        'nic',
        *[f'pixdim:{i}' for i in np.arange(4)],
        *[f'dim:{i}' for i in np.arange(4)],
        'label',
        'group',
    ]

    X      = table.drop(columns=drop_cols, errors='ignore').values
    y      = table['label'].astype(int).values
    groups = table['group'].values

    return X, y, groups


def train(table: pd.DataFrame, model_class=None, loo=False, n_jobs=None):

    X, y, groups = prepare_data_for_training(table)
    model        = create_model(model_class, n_jobs=n_jobs)
    pipe         = create_pipeline(model)

    pipe.fit(X, y)

    if loo:
        loo = _cv_loo(model, X, y, groups, n_jobs=n_jobs)
    else:
        loo = None

    return model, loo


def tuned_train(table: pd.DataFrame,
                model_class=None,
                loo=False,
                n_jobs=None,
                n_trials=100):

    X, y, groups = prepare_data_for_training(table)

    def objective(trial):

        # We parallelise sklearn/xgboost model fitting
        # and run optuna trials (below) single-threaded
        model = create_model(
            n_jobs=n_jobs,
            **_tune_get_model_hyper_parameters(model_class, trial))

        thres = trial.suggest_categorical('pyfix_threshold',
                                          (1, 2, 5, 10, 20, 30, 40, 50))
        y_pred, y_pred_pb, test_idx = _cv_loo(model, X, y, groups)[:3]
        y_pred = legacy.prob_threshold(y_pred_pb, threshold=thres / 100)

        score = 0

        for idx in test_idx:
            cm = metrics.confusion_matrix(y[idx], y_pred[idx], labels=[0, 1])
            score += 1 - report.fix_metric(cm)[2]

        return score

    with TemporaryDirectory() as td:

        logfile = op.join(td, 'study.log')
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(logfile))
        study = optuna.create_study(study_name='pyfix', storage=storage)

        study.enqueue_trial({'pyfix_threshold' : 20})

        study.optimize(objective, n_trials=n_trials)

        params    = dict(study.best_trial.params)
        threshold = params.pop('pyfix_threshold')

    # Re-train model using parameters from best trial
    model = create_model(**params)
    pipe  = create_pipeline(model)
    pipe.fit(X, y)

    if loo:
        loo = _cv_loo(pipe, X, y, groups, n_jobs=n_jobs)
    else:
        loo = None

    return model, loo, threshold


def _tune_get_model_hyper_parameters(model_class, trial):
    """Used by tuned_train. Returns a dict of initialisation paramaters
    for the classifier, generated by optuna.
    """

    # XGBClassifier hyper-parameters - Copied from
    # https://github.com/optuna/optuna-examples/blob/\
    #   71e00a0b16103ca91b5e3380ebc158bca6582741/xgboost/xgboost_simple.py
    if xgboost is not None:
        params = {
            # defines booster, gblinear for linear functions.
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            # L2 regularization weight.
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            # L1 regularization weight.
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            # sampling according to each tree.
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        }

        if params['booster'] in ['gbtree', 'dart']:
            # maximum depth of the tree, signifies complexity of the tree.
            params['max_depth'] = trial.suggest_int('max_depth', 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            params['min_child_weight'] = trial.suggest_int('min_child_weight', 2, 10)
            params['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            params['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

        if params['booster'] == 'dart':
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
            params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)

    # sklearn RandomForestClassifier params
    else:
        params = {
            'n_estimators'      : [10, 50, 100, 150, 200, 250],
            'max_features'      : ['auto', 'sqrt', 'log2'],
            'max_depth'         : [None, 10, 20, 30, 40, 50],
            'min_samples_split' : [2, 5, 10, 15, 20],
            'min_samples_leaf'  : [1, 2, 5, 10, 15]
        }
        params = {k : trial.suggest_categorical(k, v)
                  for k, v in params.items()}

    return params


def _cv_loo(model, X, y, groups, n_jobs=None):

    loo     = LeaveOneGroupOut()
    results = []

    folds   = list(loo.split(X, y, groups))
    for train, test in folds:
        results.append(__fit_and_predict(clone(model), X, y, train, test))

    y_pred    = np.concatenate([y0  for y0, _  in results])
    y_pred_pb = np.concatenate([ypb for _, ypb in results])

    test_idx     = [tst for _, tst in folds]
    test_idx_cat = np.concatenate(test_idx)

    y_pred    = y_pred[   np.argsort(test_idx_cat)]
    y_pred_pb = y_pred_pb[np.argsort(test_idx_cat), :]

    return y_pred, y_pred_pb, test_idx


def __fit_and_predict(model, X, y, train_idx, test_idx):
    X_trn, X_tst, y_trn, y_tst = X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]

    pipe = create_pipeline(model)

    # fit model
    pipe.fit(X_trn, y_trn)

    # predict class
    y_pred = pipe.predict(X_tst)

    # and class probabilities (where possible -
    # not all classifiers support predict_proba)
    try:                   y_pred_pb = pipe.predict_proba(X_tst)
    except AttributeError: y_pred_pb = None

    return y_pred, y_pred_pb
