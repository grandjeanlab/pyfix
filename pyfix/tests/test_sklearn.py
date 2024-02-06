#!/usr/bin/env python
#
# test_sklearn.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import glob
import shlex

from fsl.utils.tempdir import tempdir
from pyfix             import fix
from pyfix.tests       import (unzip,
                               main,
                               TRAINDATA_ARCHIVE)


def test_sklearn_train_classify():
    with tempdir():
        unzip(TRAINDATA_ARCHIVE)

        modelcls = 'sklearn.ensemble.RandomForestClassifier'
        indirs   = ' '.join(sorted(glob.glob('*.ica')))

        main(f'-t training -m {modelcls} -l {indirs}')
        main('-c 1.ica training.pyfix_model 20')
