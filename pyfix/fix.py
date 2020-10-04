#!/usr/bin/env python
""" FIX (FMRIB's ICA-based Xnoiseifier).

Usage:
    fix.py extract --func=FUNC --struct2func=STRUCT2FUNC
                   --ic_timeseries=ICTIME --ic_map=ICMAP
                   --motparam=MOTPARAM (--struct=STRUCT | --struct_seg=SEG)
                   [--ic_threshold=THRESHOLD] [--hp=HP]
                   [--func_brain_mask=MASK] [--standard2func_tx=TX]
                   [--overwrite]
    fix.py extract --featdir=FEATDIR
    fix.py train --output=DIRNAME [--loo] FIXDIR FIXDIR...
    fix.py classify --fixdir=DIR --classifier=CLF
    fix.py apply --func=FUNC --ic_timeseries=ICTIME --rmidx=IDX
                 [--motparam=MOTPARAM]

"""
from docopt import docopt
import logging

log = logging.getLogger(__name__)


if __name__ == '__main__':
    args = docopt(__doc__, version='fix.py 0.1.0a')

    if args['extract']:
        if args['--featdir'] is not None:
            print('import from featdir')
        else:
            print('extract from files')

    if args['train']:
        print('FIX train')

    if args['classify']:
        print('FIX classify')

    if args['apply']:
        print('FIX apply')
