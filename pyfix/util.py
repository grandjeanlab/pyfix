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
    """Return a list of the names of all built-in pyFIX models. """
    modeldir   = op.join(get_resource_path(), 'models')
    info       = op.join(modeldir, 'info.json')

    with open(info, 'rt') as f:
        info = json.loads(f.read())

    return sorted(info.keys())


def get_mask(name : str, species : str = None) -> str:
    if species in (None, 'mni152_1mm'):
        species = ''
    return op.join(get_resource_path(), 'masks', species, name)


def get_builtin_model(name : str) -> dict:
    """Retrieve information about a built-in pyFIX model, given its
    name as listed in info.json, or its model file name prefix.
    """

    modelpath = op.join(get_resource_path(), 'models')
    info      = op.join(modelpath, 'info.json')

    with open(info, 'rt') as f:
        info = json.loads(f.read())

    # name is not listed in info.json - see if
    # it is the prefix of any model file names
    if name not in info:
        for model_name, model_info in info.items():
            fname = model_info['file']
            fname = op.splitext(op.basename(fname))[0]
            if name == fname:
                name = model_name
                break
        else:
            raise KeyError(f'No built in model with name {name}')

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
