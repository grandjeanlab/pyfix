#!/usr/bin/env python
#
# test_species.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import            glob
import os.path as op
import            os

import            shutil

from unittest import mock

from fsl.utils.tempdir import tempdir

from pyfix import io, util

from pyfix.tests import (TRAINDATA_ARCHIVE,
                         DATA_ARCHIVE,
                         FIXDIR_ARCHIVE,
                         main,
                         unzip,
                         mock_features)


def fake_custom_masks(destdir):
    rdir    = util.get_resource_path()
    mdir    = op.join(rdir, 'masks')
    destdir = op.abspath(destdir)

    os.makedirs(destdir, exist_ok=True)

    for mfile in glob.glob(op.join(mdir, 'sag_mask*')):
        shutil.copy(mfile, destdir)

    return destdir



def test_get_mask():
    rdir = util.get_resource_path()
    mdir = op.join(rdir, 'masks')
    assert util.get_mask('sag_mask0')               == op.join(mdir, 'sag_mask0.nii.gz')
    assert util.get_mask('sag_mask0', '')           == op.join(mdir, 'sag_mask0.nii.gz')
    assert util.get_mask('sag_mask0', 'mni152_1mm') == op.join(mdir, 'sag_mask0.nii.gz')
    assert util.get_mask('sag_mask0', 'macaque')    == op.join(mdir, 'macaque', 'sag_mask0.nii.gz')

    with tempdir():
        spdir = fake_custom_masks('random_species')
        assert util.get_mask('sag_mask0', spdir) == op.join(spdir, 'sag_mask0.nii.gz')


def test_fix_feature_extraction_species():

    with tempdir():

        maskdir = fake_custom_masks('custom_masks')
        unzip(DATA_ARCHIVE)
        unzip(FIXDIR_ARCHIVE, 'rest.ica')

        checked = [False]

        def feature_hook(data, ic_idx):
            assert data.species == maskdir
            checked[0] = True

        with mock_features(feature_hook):
            main(f'-f rest.ica -s {maskdir}')
        assert checked[0]


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
