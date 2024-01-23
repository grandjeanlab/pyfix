#!/usr/bin/env python


import os.path as     op
from   typing  import Tuple

import pandas  as     pd
import numpy   as     np
from   sklearn import metrics

import fsl.utils.path as fslpath

from pyfix import io, legacy


def report_evaluate(meldirs, model, thresholds):

    basedir = fslpath.commonBase([op.abspath(d) for d in meldirs])
    columns = ['Dataset', 'Threshold', 'TPR', 'TNR', '(3*TPR+TNR)/4']
    df      = pd.DataFrame(columns=columns)

    row = 0

    for meldir in meldirs:
        indir       = io.InputDirectory.from_input_dir(meldir)
        dataset     = op.relpath(indir.indir, basedir)
        nics        = indir.nics
        hand_labels = io.read_labels(indir.input_tree.get('hand_labels'), nics)
        hand_labels = hand_labels == 0

        for threshold in thresholds:
            model_labels = io.generate_label_filename(meldir, model, threshold)
            model_labels = io.read_labels(model_labels, nics)
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
    records   = []

    for thr0 in threshold:

        y_pred = legacy.prob_threshold(y_pred_pb, threshold=thr0)

        for fold, idx in enumerate(test_idx):
            cm = metrics.confusion_matrix(y[idx], y_pred[idx], labels=labels)
            tpr, tnr, weighted = fix_metric(cm)
            records += [(fold, thr0*100, tpr, tnr, weighted)]

    df       = pd.DataFrame(records, columns=('Fold', 'Threshold', 'TPR', 'TNR', '(3*TPR+TNR)/4'))
    fmt_func = lambda f: f'{f:0.2f}'

    df_str  = df.pivot(index='Fold', columns='Threshold').to_string(float_format=fmt_func)
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
