#!/usr/bin/env python
#
# test_convert.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import shlex

import pytest

from fsl.utils.tempdir import tempdir

from pyfix             import (convert,
                               fix)
from pyfix.tests       import (TRAINING_RDATA_MODEL,
                               TRAINDATA_ARCHIVE,
                               unzip)


pytestmark = pytest.mark.need_r


def test_convert_fix_model():
    with tempdir():
        modelfile = 'training.pyfix_model'

        # convert RData model to pyfix model
        convert.main(shlex.split(f'{TRAINING_RDATA_MODEL} {modelfile} -l'))

        # Make sure we can use it for classification
        unzip(TRAINDATA_ARCHIVE)
        fix.main(shlex.split(f'-c 1.ica {modelfile} 30'))
