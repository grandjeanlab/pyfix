#!/usr/bin/env python
import sys
from enum import Enum, unique
from subprocess import check_output
from typing import List
import logging
import time
import os.path as op

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