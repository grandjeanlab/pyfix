#!/usr/bin/env python
#
# test_old_models.py - Test use of old pyfix saved models
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import glob


from fsl.utils.tempdir import tempdir

from pyfix             import (fix,
                               io)
from pyfix.tests       import (main,
                               unzip,
                               TRAINDATA_ARCHIVE)


def test_classify_0_3_0():
    with tempdir():
        unzip(TRAINDATA_ARCHIVE)

        # 0.3.0 file format only supports sklearn classifiers
        model_cls = 'sklearn.ensemble.RandomForestClassifier'
        indirs    = list(glob.glob('*.ica'))
        model     = fix.run_train(meldir=indirs,
                                  basename='training',
                                  model_class=model_cls)

        # resave the model in 0.3.0 format
        io.save_model('training.pyfix_model', model, '0.3.0')
        main('-c 1.ica training.pyfix_model 20')



def test_classify_0_4_0():
    with tempdir():
        unzip(TRAINDATA_ARCHIVE)

        indirs    = list(glob.glob('*.ica'))
        model     = fix.run_train(meldir=indirs, basename='training')

        # resave the model in 0.4.0 format
        io.save_model('training.pyfix_model', model, '0.4.0')
        main('-c 1.ica training.pyfix_model 20')

def test_classify_0_5_0():
    with tempdir():
        unzip(TRAINDATA_ARCHIVE)

        indirs    = list(glob.glob('*.ica'))
        model     = fix.run_train(meldir=indirs, basename='training')

        # resave the model in 0.5.0 format
        io.save_model('training.pyfix_model', model, '0.5.0')
        main('-c 1.ica training.pyfix_model 20')
