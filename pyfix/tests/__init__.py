#!/usr/bin/env python

import            io
import            os
import os.path as op
import            shlex
import            sys
import            zipfile


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
