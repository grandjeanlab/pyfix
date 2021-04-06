#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import os.path as op
import logging
from sklearn import metrics
from typing import Tuple

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


def read_features(path: str):
    """Read legacy FIX features from file.

    Args:
        path (str):

    Returns:
        features table (pd.DataFrame):
    
    """

    filename = path

    if not filename.endswith('features.csv'):
        filename = filename + 'features.csv'

    # filename = op.join(path, 'features.csv')

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


def prob_threshold(pb, threshold, col=1):
    """
    Threshold probablistic classifications.

    probability[:, col] > thres

    Lower (conservative) threshold will result in more outputs being included
    """
    return pb[:, col] > threshold


def report_loo(y, y_pred_pb, test_idx, threshold=(1, 2, 5, 10, 20, 30, 40, 50), labels=[0, 1]):

    threshold = np.array(threshold) / 100
    records = []

    for thr0 in threshold:

        y_pred = prob_threshold(y_pred_pb, threshold=thr0)

        for fold, idx in enumerate(test_idx):
            cm = metrics.confusion_matrix(y[idx], y_pred[idx], labels=labels)
            tpr, tnr, weighted = fix_metric(cm)
            records += [(fold, thr0*100, tpr, tnr, weighted)]

    df = pd.DataFrame(records, columns=('Fold', 'Threshold', 'TPR', 'TNR', '(3*TPR+TNR)/4'))

    df_str = df.pivot(index='Fold', columns='Threshold').to_string()
    df_str += '\n\nMean\n'
    df_str += df.groupby('Threshold').mean().T.drop(index='Fold').to_string()
    df_str += '\n\nMedian\n'
    df_str += df.groupby('Threshold').median().T.drop(index='Fold').to_string()

    df_str += '\n\n\n---\nTPR==True Positive (signal) rate\nTPR==True Negative (noise) rate\n'

    return df_str


def fix_metric(cm: np.array) -> Tuple[float, float, float]:
    """
    Calculate:
    1. TNR from CM (negative==noise)
    2. TPR from CM (positive==signal)
    3. weighted TPR and TNR metric [ (3*TPR+TNR)/4 ]


    Args:
        cm: confusion matrix [true x predicted] (sklearn definition)

    Returns:
        TPR: true positive (signal) rate
        TNR: true negative (noise) rate
        weighted: weighted TPR and TNR metric [ (3*TPR+TNR)/4 ]

    """

    TPR = __tpr(cm)
    TNR = __tnr(cm)

    return TPR, TNR, (3*TPR+TNR)/4


def __tnr(cm):
    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        tnr = 0
    else:
        tnr = tn / (tn + fp)

    return tnr


def __tpr(cm):
    tn, fp, fn, tp = cm.ravel()

    if (tp + fn) == 0:
        tpr = 0
    else:
        tpr = tp / (tp + fn)

    return tpr


# class FixLegacyClassifier(BaseEstimator, ClassifierMixin):
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
#         loo = LeaveOneOut()
#
#
#         self.classes_, indices = np.unique(y, return_inverse=True)
#
#         # SVM - RBF
#         self.base1 = svm.SVC(probability=True, class_weight="balanced",
#                              kernel='rbf', gamma='scale')
#         Y1 = cross_val_predict(self.base1, X, y, cv=loo, verbose=1,
#                                   n_jobs=self.n_jobs)
#         self.base1.fit(X, y)
#
#         # SVM - Linear
#         self.base2 = svm.SVC(probability=True, class_weight="balanced",
#                              kernel='linear')
#         Y2 = cross_val_predict(self.base2, X, y, cv=loo, verbose=1,
#                                   n_jobs=self.n_jobs)
#         self.base2.fit(X, y)
#
#         # SVM - polynomial
#         self.base3 = svm.SVC(probability=True, class_weight="balanced",
#                              kernel='poly')
#         Y3 = cross_val_predict(self.base3, X, y, cv=loo, verbose=1,
#                                   n_jobs=self.n_jobs)
#         self.base3.fit(X, y)
#
#         # k-NN
#         self.base4 = neighbors.KNeighborsClassifier()
#         Y4 = cross_val_predict(self.base4, X, y, cv=loo, verbose=1,
#                                   n_jobs=self.n_jobs)
#         self.base4.fit(X, y)
#
#         # decision tree
#         self.base5 = tree.DecisionTreeClassifier()
#         Y5 = cross_val_predict(self.base5, X, y, cv=loo, verbose=1,
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


