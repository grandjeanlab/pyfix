#!/usr/bin/env python
import sys
from subprocess import check_output
from typing import List
import logging
import time
import os.path as op

log = logging.getLogger(__name__)


def run(cmd: List[str]) -> str:
    log.info(" ".join(cmd))
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
        assert op.exists(f), "file/folder does not exist: {0}".format(f)

