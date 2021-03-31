#!/usr/bin/env python
# from fsl.data import fixlabels
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import pipeline, metrics
from sklearn import preprocessing
from sklearn import svm as svm
from sklearn.base import clone
# from sklearn.inspection import _weights_scorer
from sklearn.metrics import check_scoring
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import check_random_state

from pyfix import io, legacy

log = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, m=None):
    if m is None:
        m = [
            metrics.confusion_matrix,
            metrics.balanced_accuracy_score,
            metrics.matthews_corrcoef,
            metrics.roc_auc_score,
        ]

    return {f.__name__: f(y_true, y_pred) for f in m}


def loo_training_metrics(y, y_pred_pb, test_idx, threshold=None):
    if threshold is None:
        threshold = 0.5

    if np.isscalar(threshold):
        threshold = [threshold]

    records = []

    for thr0 in threshold:

        y_pred = legacy.prob_threshold(y_pred_pb, threshold=thr0)

        for fold, idx in enumerate(test_idx):
            m = calculate_metrics(y[idx], y_pred[idx])
            m['fold'] = fold
            m['threshold'] = thr0

            m['tpr'], m['tnr'], m['(3*tpr+tnr)/4'] = legacy.fix_metric(m['confusion_matrix'])

            records += [m]

    df = pd.DataFrame(records)

    # def df_to_html(df0):
    #     html_str = df0.to_html(
    #         classes="table table-hover table-bordered",
    #         float_format=lambda x: f'{x:0.4f}'
    #     )
    #     return html_str
    #
    # env = jinja2.Environment(
    #     loader=jinja2.PackageLoader('pyfix', 'resources'),
    #     autoescape=jinja2.select_autoescape(['html', 'xml']),
    #     extensions=['jinja2.ext.do']
    # )
    #
    # template = env.get_template('training_report_template.html')
    # html = template.render(
    #     dataframe=df,
    #     df_to_html=df_to_html,
    # )
    #
    # with open(outname, 'w') as outfile:
    #     outfile.write(html)

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


@dataclass(frozen=True)
class LooResults:
    actual: np.ndarray
    predicted: np.ndarray
    predicted_pb: np.ndarray
    features: np.ndarray
    feature_labels: List[str]
    idx: List[np.array]
    models: list

    def __post_init__(self):
        assert isinstance(self.actual, np.ndarray), 'LooResults.actual must be an np.ndarray'
        assert isinstance(self.predicted, np.ndarray), 'LooResults.predicted must be an np.ndarray'
        assert isinstance(self.predicted_pb, np.ndarray), 'LooResults.predicted_pb must be an np.ndarray'
        assert isinstance(self.features, np.ndarray), 'LooResults.features must be an np.ndarray'

        # TODO: check the contents of these lists
        assert isinstance(self.idx, list), 'LooResults.idx must be list of lists of indices'
        assert isinstance(self.models, list), 'LooResults.idx must be list of models'
        assert isinstance(self.feature_labels, list), 'LooResults.idx must be list of models'

        assert np.all([d0.shape[0] == self.n_samples for d0 in [self.predicted, self.predicted_pb, self.features]]), \
            f'LooResults.{{actual,predicted,predicted_pb,features}} first dimension size must be = {self.n_samples}'

        assert np.all([len(d0) == self.n_folds for d0 in [self.models]]), \
            f'LooResults.{{idx,models}} must be length = {self.n_folds}'

        assert len(self.feature_labels) == self.n_features, \
            f'LooResults.feature_labels must be length = {self.n_features}'

    @property
    def n_samples(self):
        return self.features.shape[0]

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_folds(self):
        return len(self.idx)

    def __repr__(self):
        str0 = (
            f"""LooResults:
    n_folds: {self.n_folds}
    n_samples: {self.n_samples}
    n_features: {self.n_features}
    actual: {self.actual.shape}
    predicted: {self.predicted.shape}
    predicted_pb: {self.predicted_pb.shape}
    features: {self.features.shape}
    feature_labels: [{len(self.feature_labels)}]
    idx: [{len(self.idx)}]
    models: {'None' if self.models is None else '[' + str(len(self.models)) + ']'}\n""")

        return str0


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

    feature_labels = list(table.drop(columns=drop_cols, errors='ignore'))

    if model is None:
        model = svm.SVC(probability=True, C=1, gamma='scale', class_weight="balanced")

    output = (model.fit(X, y),)

    if loo:
        y_pred, y_pred_pb, test_idx, trained_models = _cv_loo(model, X, y, group, n_jobs=n_jobs)
        output += (LooResults(y, y_pred, y_pred_pb, X, feature_labels, test_idx, trained_models),)

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


def loo_permutation_importance(loo_result, scoring='balanced_accuracy', n_repeats_per_fold=1, feature_groups=None,
                               n_jobs=None):
    n_samples = loo_result.n_samples

    X = loo_result.features
    y = loo_result.actual

    imp_tst = []
    imp_trn = []

    for idx_tst, model0 in zip(loo_result.idx, loo_result.models):
        idx_trn = np.array(list(set(np.arange(n_samples)).difference(set(list(idx_tst)))))

        imp_tst += [
            permutation_importance(model0, X[idx_tst, :], y[idx_tst], scoring=scoring, n_repeats=n_repeats_per_fold,
                                   feature_groups=feature_groups, n_jobs=n_jobs)]
        imp_trn += [
            permutation_importance(model0, X[idx_trn, :], y[idx_trn], scoring=scoring, n_repeats=n_repeats_per_fold,
                                   feature_groups=feature_groups, n_jobs=n_jobs)]

    imp_tst = np.concatenate(imp_tst)
    imp_trn = np.concatenate(imp_trn)

    return imp_trn, imp_tst


# adapted from scikit-learn
def permutation_importance(model, X, y, scoring=None, n_repeats=5, random_state=None, sample_weight=None,
                           feature_groups=None, n_jobs=None):
    random_state = check_random_state(random_state)
    scorer = check_scoring(model, scoring=scoring)

    if sample_weight is not None:
        baseline_score = scorer(model, X, y, sample_weight)
    else:
        baseline_score = scorer(model, X, y)

    n_feat = X.shape[1]

    if feature_groups is not None:
        features_idx = feature_groups.values()
    else:
        features_idx = np.arange(n_feat)

    scores = Parallel(n_jobs=n_jobs)(delayed(_calculate_permutation_scores)(
        model, X, y, sample_weight, feat, random_state, n_repeats, scorer
    ) for feat in features_idx)

    scores = np.stack(scores, axis=1)
    importance = baseline_score - scores

    return importance


# adapted from scikit-learn
def _calculate_permutation_scores(model, X, y, sample_weight, col_idx,
                                  random_state, n_repeats, scorer):
    """
    Calculate score when `col_idx` is permuted.

    Adapted from scikit-learn:
    https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/inspection/_permutation_importance.py#L19

    This version will accept multiple columns for permutation.
    This allows evaluation of the importance of groups of features.
    """

    col_idx = np.array(col_idx, ndmin=1)

    X_copy = X.copy()
    scores = np.zeros(n_repeats)
    shuffling_idx = np.arange(X.shape[0])
    for n_round in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_copy, "iloc"):
            X_copy = X_copy.values

        X_copy[:, col_idx] = X_copy[:, col_idx][shuffling_idx, :]

        if sample_weight is not None:
            feature_score = scorer(model, X_copy, y, sample_weight)
        else:
            feature_score = scorer(model, X_copy, y)

        scores[n_round] = feature_score

    return scores
