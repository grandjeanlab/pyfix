#!/usr/bin/env python
#
# test_normal_usage.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import            glob
import            os
import os.path as op
import            shlex

import numpy   as np
import pandas  as pd

from fsl.utils.tempdir import tempdir

from pyfix import fix
from pyfix.tests import unzip


DATADIR           = op.join(op.dirname(__file__), 'testdata')
DATA_ARCHIVE      = op.join(DATADIR, 'rest.ica.zip')
FIXDIR_ARCHIVE    = op.join(DATADIR, 'rest.ica.fixdir.zip')
TRAINDATA_ARCHIVE = op.join(DATADIR, 'training_data.zip')


def test_print_help():
    fix.main([])
    fix.main(['-h'])



def test_feature_extraction():
    """
    """
    with tempdir():

        unzip(DATA_ARCHIVE)
        unzip(FIXDIR_ARCHIVE)

        fix.main(shlex.split('-f rest.ica'))

        for filename in ['features.csv', 'features.pyfix_features']:
            got    = op.join('rest.ica', 'fix', filename)
            expect = op.join(            'fix', filename)
            assert np.all(np.isclose(got.values, expect.values))



def test_train():
    """
    """
    with tempdir():
        unzip(TRAINDATA_ARCHIVE)
        indirs = ' '.join(sorted(glob.glob('*.ica')))

        fix.main(shlex.split(f'-t training1 {indirs}'))
        assert op.exists('training1.pyfix_model')

        fix.main(shlex.split(f'-t training2 -l {indirs}'))
        assert op.exists('training2.pyfix_model')
        assert op.exists('training2_LOO_results')
