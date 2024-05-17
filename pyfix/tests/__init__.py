#!/usr/bin/env python

import               collections
import               contextlib
import functools  as ft
import               io
import               os
import os.path    as op
import               shlex
import               sys
import               zipfile
from unittest import mock

from pyfix import fix, feature
from pyfix.util import indir


DATADIR              = op.join(op.dirname(__file__), 'testdata')
DATA_ARCHIVE         = op.join(DATADIR, 'rest.ica.zip')
FIXDIR_ARCHIVE       = op.join(DATADIR, 'rest.ica.fixdir.zip')
TRAINDATA_ARCHIVE    = op.join(DATADIR, 'training_data.zip')
TRAINED_MODEL        = op.join(DATADIR, 'training.pyfix_model')
TRAINING_RDATA_MODEL = op.join(DATADIR, 'training.RData')
CLEANED_DATA_ARCHIVE = op.join(DATADIR, 'cleaned_data.zip')


class CaptureStdout:
    """Context manager which captures stdout and stderr. """

    def __init__(self):
        self.reset()

    def reset(self):
        self.__mock_stdout      = io.StringIO('')
        self.__mock_stderr      = io.StringIO('')
        self.__mock_stdout.mode = 'w'
        self.__mock_stderr.mode = 'w'
        return self

    def __enter__(self):
        self.__real_stdout = sys.stdout
        self.__real_stderr = sys.stderr
        sys.stdout         = self.__mock_stdout
        sys.stderr         = self.__mock_stderr
        return self


    def __exit__(self, *args, **kwargs):
        sys.stdout = self.__real_stdout
        sys.stderr = self.__real_stderr
        return False

    @property
    def stdout(self):
        self.__mock_stdout.seek(0)
        return self.__mock_stdout.read()

    @property
    def stderr(self):
        self.__mock_stderr.seek(0)
        return self.__mock_stderr.read()


def link_directory(src, dest, exclude=None):
    """Replicate a directory structure using symbolic links for all files. """

    if exclude is None:
        exclude = []

    src  = op.abspath(src)
    dest = op.abspath(dest)

    for dirpath, _, filenames in os.walk(src):

        reldirpath = op.relpath(dirpath, src)

        if reldirpath in exclude:
            continue

        os.makedirs(op.join(dest, reldirpath), exist_ok=True)

        for filename in filenames:
            filename = op.join(reldirpath, filename)

            if filename in exclude:
                continue

            os.symlink(op.join(src, filename), op.join(dest, filename))


def unzip(infile, destdir=None):
    with zipfile.ZipFile(infile) as f:
        f.extractall(destdir)


def main(cmd=''):
    return fix.main(shlex.split(cmd))


@contextlib.contextmanager
def mock_features(feature_hook=None):

    def mock_feature_func(nvals, data, ic_idx):

        if feature_hook is not None:
            feature_hook(data, ic_idx)

        if nvals == 0: return 0
        else:          return [0] * nvals

    feature_columns = collections.defaultdict(list)
    for label in feature.FEATURE_LABELS:
        prefix = label.split(':')[0]
        feature_columns[prefix].append(label)

    def num_values_for_feature(cols):
        if len(cols) == 1 and len(cols[0].split(':')) == 1:
            return 0
        return len(cols)


    funcs = {}
    for label, columns in feature_columns.items():

        nvals = num_values_for_feature(columns)

        mff = ft.partial(mock_feature_func, nvals)
        mff = feature.feature_extractor(mff, name=label, register=False)
        funcs[label] = mff

    with mock.patch('pyfix.feature.FEATURE_EXTRACTORS', funcs):
        yield
