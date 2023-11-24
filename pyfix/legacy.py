#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import os.path as op
import logging
from sklearn import metrics
from typing import Tuple

from fsl.data.image import Image
import fsl.utils.path as fslpath
from fsl.data import fixlabels

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


def report_evaluate(meldirs, model, thresholds):

    basedir = fslpath.commonBase([op.abspath(d) for d in meldirs])
    columns = ['Dataset', 'Threshold', 'TPR', 'TNR', '(3*TPR+TNR)/4']
    df      = pd.DataFrame(columns=columns)

    row = 0

    for meldir in meldirs:
        dataset     = op.relpath(meldir, basedir)
        nics        = Image(op.join(meldir, 'filtered_func_data.ica', 'melodic_IC')).shape[3]
        hand_labels = read_labels(op.join(meldir, 'hand_labels_noise.txt'), nics)
        hand_labels = hand_labels == 0

        for threshold in thresholds:
            model_labels = generate_label_filename(meldir, model, threshold)
            model_labels = read_labels(model_labels, nics)
            model_labels = model_labels == 0

            cm = metrics.confusion_matrix(hand_labels, model_labels, labels=[0, 1])
            tpr, tnr, weighted = fix_metric(cm)

            df.loc[row, columns] = (dataset, threshold, tpr, tnr, weighted)
            row += 1

    all_results = df.pivot(index='Dataset', columns='Threshold', values=['TPR', 'TNR'])
    all_results = all_results.sort_index(axis='columns', level='Threshold')
    all_results.insert(0, 'Dataset', all_results.index)
    all_results.set_index(np.arange(len(all_results)), inplace=True)
    all_results.index.name = 'ID'

    fmt_func    = lambda f: f'{f:0.2f}'
    report_str  = all_results.to_string(float_format=fmt_func)

    group = df.drop('Dataset', axis=1).groupby('Threshold')

    report_str += '\n\nMean\n'
    report_str += group.mean().T.to_string(float_format=fmt_func)
    report_str += '\n\nMedian\n'
    report_str += group.median().T.to_string(float_format=fmt_func)
    report_str += '\n\n\n---\nTPR==True Positive (signal) rate\nTPR==True Negative (noise) rate\n'

    return report_str


def report_loo(y, y_pred_pb, test_idx, threshold=(1, 2, 5, 10, 20, 30, 40, 50), labels=[0, 1]):

    threshold = np.array(threshold) / 100
    records = []

    for thr0 in threshold:

        y_pred = prob_threshold(y_pred_pb, threshold=thr0)

        for fold, idx in enumerate(test_idx):
            cm = metrics.confusion_matrix(y[idx], y_pred[idx], labels=labels)
            tpr, tnr, weighted = fix_metric(cm)
            records += [(fold, thr0*100, tpr, tnr, weighted)]

    df       = pd.DataFrame(records, columns=('Fold', 'Threshold', 'TPR', 'TNR', '(3*TPR+TNR)/4'))
    fmt_func = lambda f: f'{f:0.2f}'

    df_str = df.pivot(index='Fold', columns='Threshold').to_string(float_format=fmt_func)
    df_str += '\n\nMean\n'
    df_str += df.groupby('Threshold').mean().T.drop(index='Fold').to_string(float_format=fmt_func)
    df_str += '\n\nMedian\n'
    df_str += df.groupby('Threshold').median().T.drop(index='Fold').to_string(float_format=fmt_func)

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
