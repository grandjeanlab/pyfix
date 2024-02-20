#!/usr/bin/env python
#
# test_species.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import            glob
import os.path as op

from unittest import mock

from fsl.utils.tempdir import tempdir

from pyfix import io, util

from pyfix.tests import (TRAINDATA_ARCHIVE,
                         DATA_ARCHIVE,
                         FIXDIR_ARCHIVE,
                         main,
                         unzip)


def test_get_mask():
    rdir = util.get_resource_path()
    mdir = op.join(rdir, 'masks')
    assert util.get_mask('sag_mask0')               == op.join(mdir, 'sag_mask0')
    assert util.get_mask('sag_mask0', '')           == op.join(mdir, 'sag_mask0')
    assert util.get_mask('sag_mask0', 'mni152_1mm') == op.join(mdir, 'sag_mask0')
    assert util.get_mask('sag_mask0', 'macaque')    == op.join(mdir, 'macaque', 'sag_mask0')


def test_fix_species_read_from_model():
    with tempdir():
        unzip(TRAINDATA_ARCHIVE)

        indirs = ' '.join(sorted(glob.glob('*.ica')))

        main(f'-t training -s macaque {indirs}')

        assert op.exists('training.pyfix_model')
        assert io.load_model('training.pyfix_model').species == 'macaque'

        unzip(DATA_ARCHIVE)
        unzip(FIXDIR_ARCHIVE, 'rest.ica')

        got_species = [None]
        def mock_run_feature_extraction(meldir, species, **kwargs):
            got_species[0] = species

        with mock.patch('pyfix.fix.run_feature_extraction',
                        mock_run_feature_extraction):
            main(f'rest.ica training.pyfix_model 20')

        assert got_species[0] == 'macaque'
