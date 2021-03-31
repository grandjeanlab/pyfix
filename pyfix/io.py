#!/usr/bin/env python
import numpy as np
import pandas as pd
import joblib
import nibabel as nb
import os.path as op
import os
import logging

log = logging.getLogger(__name__)


# def read_ic_maps(filename):
#
#     if filename.endswith('.nii') or filename.endswith('.nii.gz'):
#         d = nb.load(filename)
#     else:
#         raise RuntimeError(f'unknown filetype: {filename}')
#
#     return d
#
#
# def read_ic_timeseries(filename):
#
#     if op


def read_file_table(filename):
    """Read a table of files for training.

    Args:
        filename:

    Returns:

    """
    return pd.read_csv(filename, index_col=None)


def save_file_table(filename, tbl):
    """
    Save pyFIX features to a file
    """

    if not filename.endswith('.file_table'):
        filename = filename + '.file_table'

    if op.exists(filename):
        log.warning(f'File table already exists: {filename}')

    workdir = op.expanduser(op.dirname(filename))
    if workdir == '':
        workdir = './'
    if not op.exists(workdir):
        os.makedirs(workdir)

    log.info(f'Saving file table to: {filename}')
    tbl.to_csv(filename, index=None)


def generate_file_table(*args,
                        feature_name='pyfix/features.pyfix_features',
                        label_name='hand_labels_noise.txt'):
    """
    Helper function to generate a table of file.

    Args:
        label_name:
        feature_name:
        *args:

    Returns:

    """

    d = pd.DataFrame(
        {
            'features': [op.join(f, feature_name) for f in args],
            'labels': [op.join(f, label_name) for f in args],
            'subid': [i for i, _ in enumerate(args)]
        }
    )

    return d


def read_features(filename: str):
    """Read FIX features from file.

    Args:
        filename (str):

    Returns:
        features table (pd.DataFrame):

    """
    if not filename.endswith('.pyfix_features'):
        filename = filename + '.pyfix_features'

    return pd.read_csv(filename)


def save_features(filename, features):
    """
    Save pyFIX features to a file
    """

    if not filename.endswith('.pyfix_features'):
        filename = filename + '.pyfix_features'

    if op.exists(filename):
        log.warning(f'Features file already exists: {filename}')

    workdir = op.expanduser(op.dirname(filename))
    workdir = './' if workdir == '' else workdir

    if not op.exists(workdir):
        os.makedirs(workdir)

    log.info(f'Saving pyFIX features to: {filename}')
    features.to_csv(filename, index=None)



def load_model(fname):
    """
    Load pyFIX model.

    Args:
        fname:

    Returns:

    """
    return joblib.load(fname)


def save_model(fname, mdl):
    """
    Save pyFIX model.

    Args:
        fname:
        mdl:

    Returns:

    """

    if not fname.endswith('.pyfix_model'):
        fname = fname + '.pyfix_model'

    if op.exists(fname):
        log.warning(f'Model file already exists: {fname}')

    workdir = op.expanduser(op.dirname(fname))
    workdir = './' if workdir == '' else workdir

    if not op.exists(workdir):
        os.makedirs(workdir)

    log.info(f'Saving trained model to: {fname}')
    joblib.dump(mdl, fname)


# @pd.api.extensions.register_dataframe_accessor("filetable")
# class FileTableAccessor:
#     def __init__(self, pandas_obj):
#         self._validate(pandas_obj)
#         self._obj = pandas_obj
#
#     @staticmethod
#     def _validate(obj):
#         # verify there is a column latitude and a column longitude
#         for attribute in ['features', 'labels', 'subid']:
#             if attribute not in obj.columns:
#                 raise AttributeError(f"FileTable must have {attribute}.")
#
#
# @pd.api.extensions.register_dataframe_accessor("feature")
# class FeatureAccessor:
#     def __init__(self, pandas_obj):
#         self._validate(pandas_obj)
#         self._obj = pandas_obj
#
#     @staticmethod
#     def _validate(obj):
#         # verify there is a column latitude and a column longitude
#         for attribute in ['group', 'label']:
#             if attribute not in obj.columns:
#                 raise AttributeError(f"FeatureTable must have {attribute}.")


def load_motion_parameters(fname):
    log.info(f'Load motion parameters: {fname}')

    if fname.endswith('.par'):

        log.info('Looks like a mcflirt motion parameter file (*.par)')
        mp = np.loadtxt(fname)
        mp = pd.DataFrame(mp, columns=['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z'])

    elif fname.endswith('_motion.tsv'):

        log.info('Looks like BIDS motion parameter file (*_motion.tsv)')
        mp = pd.read_csv(fname, delimiter='\t', index_col=None)

    elif fname.endswith('.eddy_parameters '):

        log.info('Looks like EDDY motion parameter file (*.eddy_parameters)')
        mp = np.loadtxt(fname)
        mp = pd.DataFrame(mp, columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'])

    else:
        raise RuntimeError('Unknown motion parameter filetype')

    return mp