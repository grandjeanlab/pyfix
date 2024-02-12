#!/usr/bin/env python

import contextlib
import glob
import json
import logging
import os
import os.path as op

from typing import List, Tuple

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


@contextlib.contextmanager
def indir(path):
    """Context manager which changes into the given directory, yields,
    then changes back.
    """
    cwd = os.getcwd()

    if not op.exists(path):
        os.makedirs(path, exist_ok=True)

    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def get_resource_path() -> str:
    rpath = op.realpath(
        op.join(op.dirname(op.abspath(__file__)), "resources"))
    return rpath


def get_resource(name) -> str:
    r = op.join(get_resource_path(), name)
    assert_file_exists(r)
    return r


def list_models() -> List[str]:
    modeldir   = op.join(get_resource_path(), 'models')
    modelfiles = glob.glob(op.join(modeldir, '*.pyfix_model'))
    modelfiles = [op.splitext(op.basename(f))[0] for f in modelfiles]
    return sorted(modelfiles)


def get_mask(name : str, species : str = None) -> str:
    if species in (None, 'mni152_1mm'):
        species = ''
    return op.join(get_resource_path(), 'masks', species, name)


def get_builtin_model(name : str) -> dict:

    modelpath = op.join(get_resource_path(), 'models')
    info      = op.join(modelpath, 'info.json')

    with open(info, 'rt') as f:
        info = json.loads(f.read())

    info         = info[name]
    info['file'] = op.join(modelpath, info['file'])

    return info


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
