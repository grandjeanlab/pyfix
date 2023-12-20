#!/usr/bin/env python
#
# test_normal_usage.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import              glob
import itertools as it
import              os
import os.path   as op
import              shlex
import              shutil

import numpy   as np
import pandas  as pd

from fsl.data.fixlabels import loadLabelFile
from fsl.utils.tempdir  import tempdir

from pyfix              import (fix,
                                legacy)
from pyfix.tests        import  unzip


DATADIR           = op.join(op.dirname(__file__), 'testdata')
DATA_ARCHIVE      = op.join(DATADIR, 'rest.ica.zip')
FIXDIR_ARCHIVE    = op.join(DATADIR, 'rest.ica.fixdir.zip')
TRAINDATA_ARCHIVE = op.join(DATADIR, 'training_data.zip')
TRAINED_MODEL     = op.join(DATADIR, 'training.pyfix_model')



def test_print_help():
    fix.main([])
    fix.main(['-h'])


def test_feature_extraction():
    """Test fix -f rest.ica"""
    with tempdir():
        unzip(DATA_ARCHIVE)
        unzip(FIXDIR_ARCHIVE)

        fix.main(shlex.split('-f rest.ica'))

        got = legacy.read_features(op.join('rest.ica', 'fix', 'features.csv'))
        exp = legacy.read_features(op.join(            'fix', 'features.csv'))

        for col in legacy.LEGACY_FEATURE_LABELS:
            gotcol = got[col]
            expcol = exp[col]
            if np.all(np.isclose(gotcol, expcol)):
                continue

            exprange = expcol.max() - expcol.min()
            maxerr   = (gotcol - expcol).abs().max() / exprange
            corr     = gotcol.corr(expcol)

            assert corr > 0.99

            # fftfinerwrtnull generates a random null
            # distribution, so results are expected to
            # differ
            if col.startswith('fftfinerwrtnull'):
                assert maxerr < 0.10
            else:
                assert maxerr < 0.01


def test_train():
    """Test:
      - fix -t pyfix_training    *.ica
      - fix -t pyfix_training -l *.ica
    """

    with tempdir():
        unzip(TRAINDATA_ARCHIVE)

        indirs = ' '.join(sorted(glob.glob('*.ica')))

        fix.main(shlex.split(f'-t training1 {indirs}'))
        assert op.exists('training1.pyfix_model')

        fix.main(shlex.split(f'-t training2 -l {indirs}'))
        assert op.exists('training2.pyfix_model')
        assert op.exists('training2_LOO_results')


def test_classify():
    """Test:
      - fix -c rest.ica training.pyfix_model <threshold>
      - fix -C training.pyfix_model *.ica
    """

    with tempdir():

        icadir = '1.ica'
        unzip(TRAINDATA_ARCHIVE)
        fix.main(shlex.split(f'-c {icadir} {TRAINED_MODEL} 30'))

        model     = op.splitext(op.basename(TRAINED_MODEL))[0]
        labelfile = legacy.generate_label_filename(icadir, TRAINED_MODEL, 30)

        assert op.exists(labelfile)
        loadLabelFile(labelfile)

        thresholds = [1, 2, 5, 10]
        icadirs    = list(glob.glob('?.ica'))

        fix.main(shlex.split(f'-C {TRAINED_MODEL} '
                             f'-t {",".join(map(str, thresholds))} '
                             f'{" ".join(icadirs)}'))

        for icadir, threshold in  it.product(icadirs, thresholds):
            labelfile = legacy.generate_label_filename(
                icadir, TRAINED_MODEL, threshold)
            assert op.exists(labelfile)
