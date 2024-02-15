#!/usr/bin/env python
#
# test_io.py - Test pyfix.io routines.
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import pandas as pd
import numpy  as np

import pytest

from fsl.utils.tempdir import tempdir

from pyfix import legacy, io, feature


def check_equal(df1, df2):

    assert df1.index.equals(df2.index)
    assert df1.columns.equals(df2.columns)
    assert np.all(np.isclose(df1.values, df2.values))



def test_read_save_features():

    # pyfix label -> old fix label
    oldfix_mapping = {v : k for k, v in legacy.LEGACY_FEATURE_LABELS.items()}
    pyfix_labels   = feature.FEATURE_LABELS
    oldfix_labels  = [oldfix_mapping[l] for l in pyfix_labels]
    nfeatures      = len(pyfix_labels)
    nics           = 20

    with tempdir():

        data  = np.random.random((nics, nfeatures))
        refdf = pd.DataFrame(data, columns=pyfix_labels)

        # read/save_features round trip
        io.save_features('features.csv', refdf)
        df = io.read_features('features.csv')
        check_equal(df, refdf)

        # all features, no column headers in file;
        # should be labelled as pyfix features
        refdf.to_csv('features.csv', header=None, index=None)
        df = io.read_features('features.csv')
        check_equal(df, refdf)
        df = io.read_features('features.csv', columns=pyfix_labels)
        check_equal(df, refdf)

        # specifying old legacy FIX column headers -
        # they should be converted to pyfix labels
        df = io.read_features('features.csv', columns=oldfix_labels)
        check_equal(df, refdf)

        # all features with column headers in file
        refdf.to_csv('features.csv', index=None)
        df = io.read_features('features.csv')
        check_equal(df, refdf)
        df = io.read_features('features.csv', columns=pyfix_labels)
        check_equal(df, refdf)

        # all features with old FIX column headers
        # should be renamed to pyfix columns
        refdf.rename(columns=oldfix_mapping).to_csv('features.csv', index=None)
        df = io.read_features('features.csv')
        check_equal(df, refdf)
        df = io.read_features('features.csv', columns=pyfix_labels)
        check_equal(df, refdf)



def test_read_features_partial():

    # pyfix label -> old fix label
    oldfix_mapping = {v : k for k, v in legacy.LEGACY_FEATURE_LABELS.items()}
    pyfix_labels   = feature.FEATURE_LABELS
    nfeatures      = len(pyfix_labels)
    nics           = 20

    with tempdir():

        data  = np.random.random((nics, nfeatures))
        refdf = pd.DataFrame(data, columns=pyfix_labels)

        tests = [
            refdf.drop(columns=pyfix_labels[:10]),
            refdf.drop(columns=pyfix_labels[-10:]),
            refdf.drop(columns=pyfix_labels[10:-10]),
        ]

        for refsubdf in tests:
            labels    = list(refsubdf.columns)
            oldlabels = [oldfix_mapping[l] for l in labels]

            # Save file without column headers, then
            # try to read without specifying expected
            # columns - should raise an error,
            refsubdf.to_csv('features.csv', index=None, header=None)
            with pytest.raises(RuntimeError):
                io.read_features('features.csv')

            # but should be fine if we specify expected
            # column headers
            df = io.read_features('features.csv', columns=labels)
            check_equal(df, refsubdf)

            # Save file with column headers - same as above -
            # should error unless we specify expected columns
            refsubdf.to_csv('features.csv', index=None)
            with pytest.raises(RuntimeError):
                df = io.read_features('features.csv')

            df = io.read_features('features.csv', columns=labels)
            check_equal(df, refsubdf)

            # save file with old fix column headers - same
            # behaviour as above
            refsubdf.rename(columns=oldfix_mapping).to_csv(
                'features.csv', index=None)

            with pytest.raises(RuntimeError):
                df = io.read_features('features.csv')

            # We should be able to specify expected columns
            # using  either pyfix or old fix feature names
            df = io.read_features('features.csv', columns=labels)
            check_equal(df, refsubdf)

            df = io.read_features('features.csv', columns=oldlabels)
            check_equal(df, refsubdf)
