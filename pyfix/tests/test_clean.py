#!/usr/bin/env python


import os

import numpy   as np

from fsl.data.image     import Image
from fsl.utils.tempdir  import tempdir
from fsl.wrappers       import imtest
from file_tree          import FileTree

from pyfix              import  util
from pyfix.tests        import (main,
                                unzip,
                                crop_image,
                                compare_4D_data,
                                DATA_ARCHIVE,
                                CLEANED_DATA_ARCHIVE)


def test_clean_mcflirt_params():

    # Normal use-case - FIX should load MCFLIRT parameter estimates
    # from mc/prefiltered_func_data_mcf.par, filter them, and save
    # them to mc/prefiltered_func_data_mcf_conf_hp.nii.gz
    with tempdir():
        unzip(DATA_ARCHIVE)

        intree        = util.get_resource('melodic.tree')
        intree        = FileTree.read(intree, 'rest.ica')
        data_file     = intree.get('filtered_func_data')
        cleaned_file  = intree.get('filtered_func_data_clean')
        mcparams_file = intree.get('motion_params_filtered')

        # work with a ROI to save time/memory
        unzip(CLEANED_DATA_ARCHIVE, paths=['fslroi_params.txt'])
        roiparams = np.loadtxt('fslroi_params.txt', dtype=int)
        crop_image(data_file, data_file, roiparams)

        assert not imtest(mcparams_file)

        main('-a rest.ica/hand_labels_noise.txt rest.ica -m -h 20')

        assert imtest(mcparams_file)

        mcparams = Image(mcparams_file).data
        cleaned1 = Image(cleaned_file).data


    # If MCFLIRT estimates do not exist at mc/prefiltered_func_data_mcf.par,
    # mc/prefiltered_func_data_mcf_conf_hp.nii.gz should be loaded instead
    # (and not high-pass filtered, as it is assumed that they have already
    # been filtered).
    with tempdir():
        unzip(DATA_ARCHIVE)

        intree            = util.get_resource('melodic.tree')
        intree            = FileTree.read(intree, 'rest.ica')
        data_file         = intree.get('filtered_func_data')
        cleaned_file      = intree.get('filtered_func_data_clean')
        mcparams_file     = intree.get('motion_params_filtered')
        raw_mcparams_file = intree.get('motion_params')

        # work with a ROI to save time/memory
        unzip(CLEANED_DATA_ARCHIVE, paths=['fslroi_params.txt'])
        roiparams = np.loadtxt('fslroi_params.txt', dtype=int)
        crop_image(data_file, data_file, roiparams)

        os.remove(raw_mcparams_file)
        Image(mcparams).save(mcparams_file)

        main('-a rest.ica/hand_labels_noise.txt rest.ica -m -h 20')

        cleaned2 = Image(cleaned_file).data

    compare_4D_data(cleaned1, cleaned2)
