#!/usr/bin/env python

import logging
import os.path as op

from typing import Tuple

import nibabel as nb
import numpy as np


def weak_import(name):
    """Try to import a module, returning None if the import fails. """
    try:
        return __import__(name)
    except ImportError:
        return None


def assert_file_exists(*args):
    """Raise an exception if the specified file/folder/s do not exist."""
    for f in args:
        if not op.exists(f):
            raise FileNotFoundError(f)


def get_resource_path() -> str:
    rpath = op.realpath(
        op.join(op.dirname(op.abspath(__file__)), "resources"))
    return rpath


def get_resource(name) -> str:
    r = op.join(get_resource_path(), name)
    assert_file_exists(r)
    return r


def create_mask(dseg: str,
                labels: Tuple[int],
                outname: str = None):

    dseg = nb.load(str(dseg))
    d = dseg.get_fdata()

    d0 = np.zeros(d.shape)

    for i in labels:
        d0[d == i] = 1

    nii = nb.Nifti1Image(d0, dseg.affine, dseg.header)

    if outname is not None:
        nii.to_filename(str(outname))

    return nii
