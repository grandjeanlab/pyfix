#!/usr/bin/env python
#
# test_feature_subset.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import glob
import os.path as op
import shlex

from fsl.utils.tempdir import tempdir

import pytest

from pyfix import (feature,
                   fix,
                   io)
from .     import (main,
                   unzip,
                   TRAINDATA_ARCHIVE)


def generate_feature_subset_data():

    subset  = feature.FEATURE_LABELS[:-10]
    missing = feature.FEATURE_LABELS[-10:]

    unzip(TRAINDATA_ARCHIVE)

    indirs = list(glob.glob('*.ica'))

    # artificially remove some feature
    # columns from each dataset
    for icadir in indirs:
        featurefile = op.join(icadir, 'fix', 'features.csv')
        features = io.read_features(featurefile)
        features.drop(missing, axis='columns', inplace=True)
        io.save_features(featurefile, features, header=False)

    with open('columns.txt', 'wt') as f:
        for col in subset:
            f.write(f'{col}\n')

    return indirs, subset


def test_train_feature_subset():

    with tempdir():

        indirs, _ = generate_feature_subset_data()

        # Training should fail if a
        # column list is not provided
        with pytest.raises(Exception):
            main(f'-t training {" ".join(indirs)}')
        assert not op.exists('training.pyfix_model')

        # But should work when
        # given a column list
        main(f'-t training -f columns.txt {" ".join(indirs)}')

        assert op.exists('training.pyfix_model')


def test_classify_feature_subset():

    with tempdir():
        indirs, _ = generate_feature_subset_data()
        indir     = indirs[0]
        main(f'-t training -f columns.txt {" ".join(indirs)}')
        main(f'-c {indir} training.pyfix_model 20')

        # gen fname
        lbls = io.generate_label_filename(indir, 'training.pyfix_model', 20)
        assert op.exists(lbls)
