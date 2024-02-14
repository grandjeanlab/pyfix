#!/usr/bin/env python
#
# convert.py - Convert an old FIX .RData model file to a pyfix model file.
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#
"""Script which converts an old FIX .RData model file to a pyfix model file.
Note that this script requires Rscript to be installed and available on the
PATH.
"""


import               argparse
import               logging
import os.path    as op
import               os
import               shlex
import               shutil
import subprocess as sp
import               sys
import textwrap   as tw

import pandas     as pd

from fsl.utils.tempdir import  tempdir
from pyfix.util        import  indir
from pyfix             import (fix,
                               io)


log = logging.getLogger('pyfix.convert')


def r(cmd):
    """Calls out to Rscript."""
    exe = shutil.which('Rscript')

    if exe is None:
        raise RuntimeError('Cannot find Rscript executable')

    cmd = f'{exe} {cmd}'

    log.debug('Calling: %s', cmd)

    result = sp.run(shlex.split(cmd), check=False, capture_output=True)

    log.debug('Result: %u',          result.returncode)
    log.debug('Standard output: %s', result.stdout.decode().strip())
    log.debug('Standard error: %s',  result.stderr.decode().strip())

    if result.returncode != 0:
        raise RuntimeError(f'Command returned non-zero exit status: {cmd}')


def extract_features(model_file, workdir):
    """Extracts FIX features and hand labels from an old FIX .RData model file.
    Returns a pandas dataframe containing the features and hand labels for all
    subjects.
    """

    # R script which loads a FIX .RData model, and
    # outputs features+hand labels to a CSV file
    features   = op.join(workdir, 'all_features.csv')
    scriptfile = op.join(workdir, 'script.r')
    script     = tw.dedent(f"""
    load("{model_file}")
    write.csv(hcp.data, "{features}")
    """).strip()

    with open(scriptfile, 'wt') as f:
        f.write(script)

    r(scriptfile)

    return pd.read_csv(features, index_col=0)


def split_features(features, workdir):
    """Splits the data frame loaded by extract_features into separate tables
    for each subject. Saves the features to a set of files within workdir,
    which can be used to train pyfix.

    Returns a tuple containing:
      - A list of directories which can be passed to the fix -t command,
        for training a pyfix model.
      - The path to a text file containing feature column labels, to be
        passed to the --feature_columns option when calling fix -t.
    """

    # outputs - list of ICA directories,
    # and file containing column names
    icadirs     = []
    column_file = op.join(workdir, 'columns.txt')

    # Generate a file containing column labels
    # to pass to pyfix for training
    columns = list(features.drop(columns=['class.labs']).columns)
    with open(column_file, 'wt') as f:
        for c in columns:
            f.write(f'{c}\n')

    # The features dataframe contains a row
    # for all ICs and for all subjects. We
    # We use the number.of.ics column (which,
    # for a given subject, is the same for
    # every IC) to calculate a row offset
    # for each subject. The subj_slices list
    # contains a (offset, nics) tuple for
    # each subject.
    all_nics    = list(features['number.of.ics'])
    subj_slices = []
    i           = 0
    while True:
        nics = all_nics[i]
        subj_slices.append((i, nics))
        i += nics
        if i >= len(all_nics):
            break

    # Create a pseudo melodic/fix directory
    # structure containing the features and
    # hand labels for every subject.
    for i, (start, nics) in enumerate(subj_slices):

        icadir       = op.join(workdir, f'{i + 1}.ica')
        fixdir       = op.join(icadir,   'fix')
        feature_file = op.join(fixdir,   'features.csv')
        label_file   = op.join(icadir,   'hand_labels_noise.txt')

        icadirs.append(icadir)

        os.makedirs(fixdir, exist_ok=True)

        subj_features = features.iloc[start:start+nics, :]
        labels        = list(subj_features['class.labs'])
        subj_features = subj_features.drop(columns=['class.labs'])

        subj_features.to_csv(feature_file, header=False, index=False)

        # generate list of noise IC indices
        labels = [i for (i, l) in enumerate(labels, start=1) if l == 0]

        with open(label_file, 'wt') as f:
            f.write(f'[{", ".join(map(str, labels))}]\n')

    return icadirs, column_file


def parse_args(argv=None):
    """Parses command-line arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument('in_model', metavar='<model.RData>',
                        help='Path to old FIX .RData model file to convert')
    parser.add_argument('out_model', metavar='<model.pyfix_model>',
                        help='Path to pyFIX model to create')
    parser.add_argument('-l', '--loo',  action='store_true',
                        help='Perform LOO evaluation on trained pyFIX model')
    parser.add_argument('-s', '--species')
    parser.add_argument('-d', '--datadir',
                        help='Save training files to this directory '
                        '(default: temporary directory')
    parser.add_argument('-lf', '--logfile',
                        default=op.join(os.getcwd(), 'pyfix.log'))

    args = parser.parse_args(argv)

    args.in_model  = op.abspath(op.expanduser(args.in_model))
    args.out_model = op.abspath(op.expanduser(args.out_model))

    # turn out_model into a basename, as
    # pyfix will auto-add the pyfix_model suffix
    if args.out_model.endswith('.pyfix_model'):
        args.out_model = args.out_model[:-12]

    if args.datadir is not None:
        args.datadir = op.abspath(op.expanduser(args.datadir))

    return args


def main(argv=None):
    """Main routine. Extracts features/hand labels from a .RData file,
    and trains a pyfiox model on them.
    """

    args = parse_args(argv)
    fix.setup_logging(args.logfile, argv)

    try:
        r('--version')
    except Exception as e:
        print('Unable to call Rscript - R must be installed to be able '
              'to convert an old FIX .RData model into a pyFIX model.')
        sys.exit(1)

    if args.datadir is not None:
        os.makedirs(args.datadir, exist_ok=True)

    with tempdir(override=args.datadir) as workdir:

        features         = extract_features(args.in_model, workdir)
        icadirs, columns = split_features(features, workdir)
        columns          = io.read_feature_columns(columns)
        modelname        = op.basename(args.out_model)

        fix.run_train(meldir=icadirs,
                      basename=modelname,
                      species=args.species,
                      feature_columns=columns,
                      loo=args.loo)
        shutil.copy(f'{modelname}.pyfix_model', op.dirname(args.out_model))


if __name__ == '__main__':
    main()
