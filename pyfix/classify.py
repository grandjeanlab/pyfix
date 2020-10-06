#!/usr/bin/env python
import pandas as pd
import numpy as np
# import os.path as op
from typing import Union
from pyfix import io
from pyfix import legacy
# from fsl.data import fixlabels
import logging

from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics

from joblib import Parallel, delayed

from sklearn import svm as svm, ensemble, neighbors, tree, neural_network as nn

from sklearn.base import clone
from sklearn.model_selection import cross_validate as cv, LeaveOneGroupOut
# from sklearn.base import BaseEstimator, ClassifierMixin

log = logging.getLogger(__name__)

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


def collate_features(table: Union[str, pd.DataFrame]):
    if isinstance(table, str):
        table = io.read_file_table(table)

    N = table.shape[0]

    train_data = [None] * N

    for idx, (f0, l0, g0) in enumerate(zip(table.features, table.labels, table.subid)):

        train_data[idx] = legacy.read_features(f0)
        train_data[idx]['group'] = g0

        l1 = np.zeros(train_data[idx].shape[0])
        for line in open(l0, "r"):
            ln = line
        ln = ln.strip().strip("[]")
        l1[np.fromstring(ln, dtype=int, sep=",") - 1] = 1

        train_data[idx]['label'] = l1

    train_data = pd.concat(train_data).astype({'label': bool})

    return train_data


def train(table: pd.DataFrame,
          clf=None,
          loo=False):

    drop_cols = [
        'nic',
        *[f'pixdim:{i}' for i in np.arange(4)],
        *[f'dim:{i}' for i in np.arange(4)],
        'label',
        'group',
    ]

    X = table.drop(columns=drop_cols).values
    y = table['label'].astype(int).values
    group = table['group'].values

    if clf is None:
        clf = svm.SVC(probability=False, class_weight="balanced", gamma='scale')

    if loo:
        Y, CM = _cvloo(clf, X, y, group, n_jobs=-2)

        TPR = np.array([__TPR(m) for m in CM])
        TNR = np.array([__TNR(m) for m in CM])
        M = ((3 * TPR) + TNR) / 4

        CM0 = np.sum(np.stack(CM, axis=2), axis=2)
        TPR0, TNR0 = __metric(CM0)
        M0 = ((3 * TPR0) + TNR0) / 4

        print("")
        print("\tTPR\tTNR\t(3*TPR+TNR)/4")
        print("Mean\t{0:0.2f}\t{1:0.2f}\t{2:0.2f}".format(np.mean(TPR) * 100,
                                                          np.mean(TNR) * 100,
                                                          np.mean(M) * 100))
        print("Median\t{0:0.2f}\t{1:0.2f}\t{2:0.2f}".format(np.median(TPR) * 100,
                                                            np.median(TNR) * 100,
                                                            np.median(M) * 100))
        print("Pooled\t{0:0.2f}\t{1:0.2f}\t{2:0.2f}".format(TPR0 * 100, TNR0 * 100,
                                                            M0 * 100))
        print("")
        print("Pooled confusion matrix:")
        print(CM0)

    return clf.fit(X, y)


def _cvloo(clf, X, y, l, n_jobs=1):
    # loo = cv.LeaveOneLabelOut(l)
    loo = LeaveOneGroupOut()

    pipe = pipeline.make_pipeline(preprocessing.StandardScaler(), clf)

    print("TPR\tTNR")

    par = Parallel(n_jobs=n_jobs)
    blocks = par(delayed(__fit_and_predict)(clone(pipe), X, y, l, train, test)
                 for train, test in loo.split(X, y, l))

    Y = np.concatenate([y for y, _, _ in blocks])
    CM = [cm for _, cm, _ in blocks]
    test = np.concatenate([tst for _, _, tst in blocks])

    Y = Y[test]

    return Y, CM


def __fit_and_predict(estimator, X, y, l, train, test):
    X_trn, X_tst, y_trn, y_tst = X[train, :], X[test, :], y[train], y[test]
    estimator.fit(X_trn, y_trn)
    Y = estimator.predict(X_tst)
    CM = metrics.confusion_matrix(y_tst, Y)
    print("{0:0.2f}\t{1:0.2f}".format(__TPR(CM) * 100,
                                      __TNR(CM) * 100))

    return Y, CM, test


def __metric(cm):
    return __TPR(cm), __TNR(cm)


def __TNR(cm):
    return cm[1, 1] / np.sum(cm[1, :])


def __TPR(cm):
    return cm[0, 0] / np.sum(cm[0, :])




# class FixClassifier(BaseEstimator, ClassifierMixin):
#     """FMRIB's ICA-based Xnoiseifier."""
#
#     def __init__(self, n_jobs=1):
#         self.n_jobs = n_jobs
#
#     def fit(self, X, y, labels=None):
#         # if labels is None:
#         #     # loo = cv.LeaveOneOut()
#         #     loo = cv.StratifiedKFold(y, n_folds=5)
#         # else:
#         #     loo = cv.LeaveOneLabelOut(labels)
#         loo = LeaveOneGroupOut()
#
#         self.classes_, indices = np.unique(y, return_inverse=True)
#
#         # SVM - RBF
#         self.base1 = svm.SVC(probability=True, class_weight="balanced",
#                              kernel='rbf')
#         Y1 = cv.cross_val_predict(self.base1, X, y, cv=loo, verbose=1,
#                                   n_jobs=self.n_jobs)
#         self.base1.fit(X, y)
#
#         # SVM - Linear
#         self.base2 = svm.SVC(probability=True, class_weight="balanced",
#                              kernel='linear')
#         Y2 = cv.cross_val_predict(self.base2, X, y, cv=loo, verbose=1,
#                                   n_jobs=self.n_jobs)
#         self.base2.fit(X, y)
#
#         # SVM - polynomial
#         self.base3 = svm.SVC(probability=True, class_weight="balanced",
#                              kernel='poly')
#         Y3 = cv.cross_val_predict(self.base3, X, y, cv=loo, verbose=1,
#                                   n_jobs=self.n_jobs)
#         self.base3.fit(X, y)
#
#         # k-NN
#         self.base4 = neighbors.KNeighborsClassifier()
#         Y4 = cv.cross_val_predict(self.base4, X, y, cv=loo, verbose=1,
#                                   n_jobs=self.n_jobs)
#         self.base4.fit(X, y)
#
#         # decision tree
#         self.base5 = tree.DecisionTreeClassifier()
#         Y5 = cv.cross_val_predict(self.base5, X, y, cv=loo, verbose=1,
#                                   n_jobs=self.n_jobs)
#         self.base5.fit(X, y)
#
#         # decision fusion
#         # self.fusion = svm.SVC(probability=False, class_weight="balanced",
#         #                       kernel='rbf')
#         self.fusion = ensemble.RandomForestClassifier(class_weight="balanced")
#
#         Xf = np.vstack((Y1, Y2, Y3, Y4, Y5)).T
#         self.fusion.fit(Xf, y)
#
#         # self.majority_ = np.argmax(np.bincount(indices))
#         return self
#
#     def predict(self, X):
#         D1 = self.base1.predict(X)
#         D2 = self.base2.predict(X)
#         D3 = self.base3.predict(X)
#         D4 = self.base4.predict(X)
#         D5 = self.base5.predict(X)
#
#         Xf = np.vstack((D1, D2, D3, D4, D5)).T
#         D = self.fusion.predict(Xf)
#
#         # return self.classes_[np.argmax(D, axis=1)]
#         return D
