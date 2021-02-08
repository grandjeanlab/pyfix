#!/usr/bin/env python
import sys
from enum import Enum, unique
from subprocess import check_output
from typing import List, Tuple
import logging
import time
import os.path as op
import nibabel as nb
import numpy as np

log = logging.getLogger(__name__)


def run(cmd: List[str]) -> str:
    log.debug(" ".join(cmd))
    # sys.stdout.flush()
    jobout = check_output(cmd)
    return jobout.decode('utf-8').strip()


def elapsedtime(func):
    def wrapper(*args, **kwargs):
        start = time.time()

        try:
            value = func(*args, **kwargs)
        finally:
            log.info('Elapsed: {}'.format(time.time() - start))

        return value
    return wrapper


def assert_file_exists(*args):
    """Raise an exception if the specified file/folder/s do not exist."""
    for f in args:
        if not op.exists(f):
            raise FileNotFoundError(f)


@unique
class SegType(Enum):
    """Enum for supported segmentation type."""
    fsl_fast = 1
    freesurfer_wmparc = 2


def get_resource_path() -> str:
    rpath = op.realpath(
        op.join(op.dirname(op.abspath(__file__)), "./resources"))
    return rpath


def get_resource(name) -> str:
    r = op.join(get_resource_path(), name)
    assert_file_exists(r)
    return r


def create_mask(dseg: str,
                labels: Tuple[int],
                outname: str = None):
    """

    """
    dseg = nb.load(str(dseg))
    d = dseg.get_data()

    d0 = np.zeros(d.shape)

    for i in labels:
        d0[d == i] = 1

    nii = nb.Nifti1Image(d0, dseg.affine, dseg.header)

    if outname is not None:
        nii.to_filename(str(outname))

    return nii