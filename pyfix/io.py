#!/usr/bin/env python
import pandas as pd

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