#!/usr/bin/env python
#
# test_builtin_models.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import os.path as op

from fsl.utils.tempdir  import tempdir
from fsl.data.fixlabels import loadLabelFile
from pyfix.tests        import (unzip,
                                main,
                                TRAINDATA_ARCHIVE)
from pyfix              import  io


def test_classify_builtin_model():
    """
    """

    modelnames = [
        'Standard',
        'UKBiobank',
        'WhII_Standard',
        'WhII_MB6',
        'HCP25_hp2000',
        'HCP7T_hp2000',
        # Some builtin models have an alias which doesn't exactly match
        # the model file name. pyFIX should accept both, but will name
        # output label files according to the model file name.
        'NHPHCP_Macaque_RIKEN30SRFIX',
        ('NHP_HCP_Macaque', 'NHPHCP_Macaque_RIKEN30SRFIX'),
        'NHPHCP_MacaqueCyno_BOLD',
        ('NHP_HCP_MacaqueCyno', 'NHPHCP_MacaqueCyno_BOLD'),
        'HCP_Style_Single_Multirun_Dedrift'
    ]

    with tempdir():
        icadir = '1.ica'
        unzip(TRAINDATA_ARCHIVE)

        for model in modelnames:

            if isinstance(model, tuple):
                model, prefix = model
            else:
                prefix = model

            main(f'-c {icadir} {model} 30')
            labelfile = io.generate_label_filename(icadir, prefix, 30)
            assert op.exists(labelfile)
            loadLabelFile(labelfile)
