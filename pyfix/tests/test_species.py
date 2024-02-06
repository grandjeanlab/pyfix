#!/usr/bin/env python
#
# test_species.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

from pyfix import util


def test_get_mask():
    rdir = util.get_resource_path()
    mdir = op.join(rdir, 'masks')
    assert util.get_mask('sag_mask0')               == op.join(mdir, 'sag_mask0')
    assert util.get_mask('sag_mask0', '')           == op.join(mdir, 'sag_mask0')
    assert util.get_mask('sag_mask0', 'mni152_1mm') == op.join(mdir, 'sag_mask0')
    assert util.get_mask('sag_mask0', 'macaque')    == op.join(mdir, 'macaque', 'sag_mask0')
