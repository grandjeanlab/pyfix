#!/usr/bin/env python
import pandas as pd
import joblib

def read_file_table(filename):
    """Read a table of files for training.

    Args:
        filename:

    Returns:

    """
    return pd.read_csv(filename, index_col=None)


def read_features(filename: str):
    """Read FIX features from file.

    Args:
        filename (str):

    Returns:
        features table (pd.DataFrame):

    """
    return pd.read_csv(filename)


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

    if not fname.endswith('.pyfix-model'):
        fname = fname + '.pyfix-model'

    joblib.dump(mdl, fname)