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

from fsl.data.image     import Image
from fsl.data.fixlabels import loadLabelFile
from fsl.utils.tempdir  import tempdir

from pyfix              import (fix,
                                io,
                                legacy)
from pyfix.tests        import (unzip,
                                indir)


DATADIR              = op.join(op.dirname(__file__), 'testdata')
DATA_ARCHIVE         = op.join(DATADIR, 'rest.ica.zip')
FIXDIR_ARCHIVE       = op.join(DATADIR, 'rest.ica.fixdir.zip')
TRAINDATA_ARCHIVE    = op.join(DATADIR, 'training_data.zip')
TRAINED_MODEL        = op.join(DATADIR, 'training.pyfix_model')
CLEANED_DATA_ARCHIVE = op.join(DATADIR, 'cleaned_data.zip')


def main(cmd=''):
    return fix.main(shlex.split(cmd))


def compare_arrays(got, exp):
    exprange = exp.max() - exp.min()
    maxerr   = (got - exp).abs().max() / exprange
    return maxerr


def compare_4D_data(got, exp):

    assert np.all(got.shape == exp.shape)
    assert compare_arrays(got, exp) < 0.01

    got      = got.reshape(-1, got.shape[3])
    exp      = exp.reshape(-1, exp.shape[3])
    nvoxels  = got.shape[0]
    corrs    = np.zeros(nvoxels)

    for i in range(nvoxels):
        corrs[i] = np.corrcoef(got[i], exp[i])[0, 1]

    assert corrs.min() > 0.99


def crop(data, params):
    xlo, xlen, ylo, ylen, zlo, zlen = params[:6]
    return data[xlo:xlo + xlen,
                ylo:ylo + ylen,
                zlo:zlo + zlen]


def test_print_help():
    """Test the following commands:
      - fix
      - fix -h
    """
    main()
    main('-h')


def test_feature_extraction():
    """Test the following commands:
      - fix -f rest.ica
    """
    with tempdir():
        unzip(DATA_ARCHIVE)
        unzip(FIXDIR_ARCHIVE)

        main('-f rest.ica')

        got = legacy.read_features(op.join('rest.ica', 'fix', 'features.csv'))
        exp = legacy.read_features(op.join(            'fix', 'features.csv'))

        for col in legacy.LEGACY_FEATURE_LABELS:
            gotcol = got[col]
            expcol = exp[col]
            if np.all(np.isclose(gotcol, expcol)):
                continue

            maxerr = compare_arrays(gotcol, expcol)
            corr   = gotcol.corr(expcol)

            assert corr > 0.99

            # fftfinerwrtnull generates a random null
            # distribution, so results are expected to
            # differ
            if col.startswith('fftfinerwrtnull'):
                assert maxerr < 0.10
            else:
                assert maxerr < 0.01


def test_train():
    """Test the following commands:
      - fix -t pyfix_training    *.ica
      - fix -t pyfix_training -l *.ica
    """

    with tempdir():
        unzip(TRAINDATA_ARCHIVE)

        indirs = ' '.join(sorted(glob.glob('*.ica')))

        main(f'-t training1 {indirs}')
        assert op.exists('training1.pyfix_model')

        main(f'-t training2 -l {indirs}')
        assert op.exists('training2.pyfix_model')
        assert op.exists('training2_LOO_results')


def test_classify():
    """Test the following commands:
      - fix -c rest.ica training.pyfix_model <threshold>
      - fix -C training.pyfix_model *.ica
    """

    with tempdir():

        icadir = '1.ica'
        unzip(TRAINDATA_ARCHIVE)
        main(f'-c {icadir} {TRAINED_MODEL} 30')

        model     = op.splitext(op.basename(TRAINED_MODEL))[0]
        labelfile = legacy.generate_label_filename(icadir, TRAINED_MODEL, 30)

        assert op.exists(labelfile)
        loadLabelFile(labelfile)

        thresholds = [1, 2, 5, 10]
        icadirs    = list(glob.glob('?.ica'))

        main(f'-C {TRAINED_MODEL} '
             f'-t {",".join(map(str, thresholds))} '
             f'{" ".join(icadirs)}')

        for icadir, threshold in  it.product(icadirs, thresholds):
            labelfile = legacy.generate_label_filename(
                icadir, TRAINED_MODEL, threshold)
            assert op.exists(labelfile)


def test_cleanup():
    """Test the following commands:
      - fix -a labels.txt mel.ica
      - fix -a labels.txt mel.ica -m
      - fix -a labels.txt mel.ica -A
    """

    flag_variants = ['', '-m', '-A']

    for flags in flag_variants:
        with tempdir():
            unzip(DATA_ARCHIVE)
            unzip(CLEANED_DATA_ARCHIVE)

            meldir    = io.InputDirectory.from_input_dir('rest.ica')
            labelfile = meldir.input_tree.get('hand_labels')

            main(f'-a {labelfile} {flags}')

            got      = meldir.input_tree.get('filtered_func_data_clean')
            gotvn    = meldir.input_tree.get('filtered_func_data_clean_vn')
            cleandir = ('clean' + ' ' + flags).strip()
            cleandir = cleandir.replace(' ', '_')
            exp   = op.join(cleandir, op.basename(got))
            expvn = op.join(cleandir, op.basename(gotvn))

            got   = Image(got).data
            exp   = Image(exp).data
            gotvn = Image(gotvn).data
            expvn = Image(expvn).data

            roiparams = np.loadtxt('fslroi_params.txt', dtype=int)
            got       = crop(got,   roiparams)
            gotvn     = crop(gotvn, roiparams)

            assert np.all(gotvn.shape == expvn.shape)
            compare_4D_data(got, exp)


def test_full_run():
    """Test the following commands:
      - fix rest.ica training.pyfix_model <threshold>
    """
    with tempdir():
        unzip(DATA_ARCHIVE)
        unzip(FIXDIR_ARCHIVE, 'rest.ica')
        main(f'rest.ica {TRAINED_MODEL} 20')

        meldir = io.InputDirectory.from_input_dir('rest.ica')
        got    = meldir.input_tree.get('filtered_func_data_clean')

        assert op.exists(got)
