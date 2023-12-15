#!/usr/bin/env python
"""
FIX (FMRIB's ICA-based Xnoiseifier).
"""

import argparse
import importlib
import itertools as it
import logging
import multiprocessing as mp
import os
import os.path as op
import sys
import warnings
from collections import defaultdict

from typing import List

import nibabel as nb
import numpy as np

from file_tree import FileTree

from fsl.data import fixlabels
from fsl.data import melodicanalysis, featanalysis, melodicimage

from pyfix import (extract,
                   feature,
                   legacy,
                   io,
                   classify,
                   clean,
                   __version__)
from pyfix.util import weak_import, get_resource


log = logging.getLogger(__name__)


USAGE = f'''
FIX (FMRIB's ICA-based Xnoiseifier) [version {__version__}]

Simple usage, assuming training data already exists:

  fix <mel.ica> <training.pyfix_model> <thresh>  [fix -a options]
    Extract features, classify and run cleanup - equivalent to:
      fix -f <mel.ica> ;
      fix -c <mel.ica> <train>.pyfix_model <thresh> ;
      fix -a <mel.ica>/fix4melview_<train>_<thresh>.txt [-a options]

    For the [fix -a options], see "fix -a" usage below.
    The first argument should be a FEAT/MELODIC output directory
    that includes a filtered_func_data.ica melodic output.

    Example:    fix study1/subject1.ica HCP_noHP.pyfix_model 20 -m -h 200
    Example:    fix subject2.feat Standard_noHP.pyfix_model 20 -m

    (ICA must have been turned on in FEAT pre-stats)

Usage for each stage separately, including creation of training data:

  fix -f <mel.ica>
    Feature extraction (for later training and/or classifying).

  fix -t <training-output-basename> [-l] [-u] <mel1.ica> <mel2.ica> ...

    Train classifier (Training output basename followed by list of Melodic
    directories).  Every Melodic directory must contain hand_labels_noise.txt
    listing the artefact components, e.g.: [1, 4, 99, ... 140].

    -l : optional LOO accuracy testing.
    -u : optional hyperparameter optimisation using Optuna (if installed).

  fix -c <mel.ica> <training.pyfix_model> <thresh>
    Classify ICA components.

  fix -C <training.pyfix_model> <mel1.ica> <mel2.ica> ...

    Classify components for all listed Melodic directories over a range of
    thresholds and produce LOO-style accuracy testing using existing hand
    classifictions.  Every Melodic directory must contain hand_labels_noise.txt
    listing the artefact components, e.g.: [1, 4, 99, ... 140].

  fix -a <labels.txt> [<mel.ica>] [-m] [-h <highpass>] [-A]

    Apply cleanup, using artefacts listed in the .txt file, to the data inside
    the enclosing Feat/Melodic directory.

    If <mel.ica> is not provided, it is assumed that <labels.txt> is contained
    within <mel.ica>.

    -h -1          apply no highpass filtering.
    -h 0           apply linear detrending only.
    -h <highpass>  with a positive <highpass> value, apply highpass with
                   <highpass> being full-width (2*sigma) in seconds.

    -m             optionally also cleanup motion confounds, with highpass
                   filtering of motion confounds controlled by -h:
                    - if -h is omitted, fix will look to see if a design.fsf
                      file is present, to find the highpass cutoff.
                    - if -h is omitted, and no design.fsf is present, no
                      filtering of the motion confounds will take place.

    -A             apply aggressive (full variance) cleanup, instead of the
                   default less-aggressive (unique variance) cleanup.
'''

def assert_is_melodic_dir(meldir):

    if melodicanalysis.isMelodicDir(meldir):
        return

    if featanalysis.hasMelodicDir(meldir):
        return

    raise RuntimeError(f'Not a conforming melodic directory: {meldir}')


def _has_features(meldir):
    return op.exists(op.join(meldir, 'fix', 'features.csv'))


def assert_has_features(meldir):
    if not _has_features(meldir):
        raise RuntimeError(f'Missing features: {meldir}/fix/features.csv')


def _has_hand_labels(meldir):
    return op.exists(op.join(meldir, 'hand_labels_noise.txt'))


def assert_has_hand_labels(meldir):
    if not _has_hand_labels(meldir):
        raise RuntimeError(f'Missing hand labels: {meldir}/hand_labels_noise.txt')


def run_feature_extraction(meldir     : str,
                           input_tree : str = None,
                           fix_tree   : str = None,
                           outdir     : str = None,
                           **kwargs):

    assert_is_melodic_dir(meldir)

    if outdir is None:
        outdir = op.join(meldir)

    # instantiate FixData object
    indir = io.InputDirectory.from_input_dir(
        meldir, input_tree, fix_tree, outdir)
    d     = extract.FixData.from_input_dir(indir)

    # extract legacy features
    f = extract.extract_features(
        data=d,
        features=legacy.LEGACY_FEATURE_EXTRACTORS)

    # save features (pyfix and legacy-fix formats)
    f.to_csv(d.fix_tree.get('pyfix_features'), index=None)
    legacy.save_features(d.fix_tree.get('legacy_features'), f)


def run_train(meldir      : List[str],
              basename    : str,
              model_class : str  = None,
              loo         : bool = False,
              tune        : bool = False,
              n_jobs      : int  = None,
              n_trials    : int  = None,
              input_tree  : str  = None,
              fix_tree    : str  = None,
              **kwargs):

    if tune:
        optuna = weak_import('optuna')
        if optuna is None:
            raise RuntimeError('You must install the optuna package in order '
                               'to perform hyperparameter optimisation with '
                               'the -u option')

    for d in meldir:
        assert_has_features(d)
        assert_has_hand_labels(d)

    file_table = legacy.generate_file_table(*meldir)

    #  save file table
    io.save_file_table(basename, file_table)

    # read and collate all features into a training table
    training_data = classify.collate_features(file_table, legacy_features=True)

    # Training with optional LOO cross validation
    if not tune:
        threshold = None
        clf, loo = classify.train(training_data,
                                  model_class,
                                  loo=loo,
                                  n_jobs=n_jobs)

    # training with automatic estimation
    # of an optimal probability threshold
    else:
        clf, loo, threshold = classify.tuned_train(training_data,
                                                   model_class,
                                                   loo=loo,
                                                   n_trials=n_trials,
                                                   n_jobs=n_jobs)

    if loo is not None:
        report = legacy.report_loo(*loo)
        with open(basename + "_LOO_results", 'w') as f:
            f.write(report)

    # save model
    io.save_model(basename, clf, threshold)


def run_classify(meldir     : str,
                 model      : str,
                 threshold  : int = None,
                 input_tree : str = None,
                 fix_tree   : str = None,
                  **kwargs):

    assert_has_features(meldir)

    indir   = io.InputDirectory.from_input_dir(meldir,
                                               input_tree,
                                               fix_tree)
    fixdir  = indir.fixdir
    intree  = indir.input_tree
    fixtree = indir.fix_tree

    if not op.exists(fixdir):
        raise RuntimeError(f'Cannot find fix directory in: {meldir}')

    features0 = legacy.read_features(fixtree.get('legacy_features'))

    clf, saved_threshold = io.load_model(model)
    ypb                  = classify.classify(features0, clf)[1]

    if threshold is None:
        threshold = saved_threshold
    if threshold is None:
        raise RuntimeError('You must specify a probability threshold - '
                           f'{args.model} does not contain an optimal '
                           'threshold.')

    y = legacy.prob_threshold(ypb, threshold=threshold / 100)

    # Save generated labels to FIX label file.
    # Invert y so that signal==False and noise==True
    # (as per legacy fix4melview files)
    y          = ~y
    labels     = [['Noise'] if i else ['Signal'] for i in y]
    modelname  = op.splitext(op.basename(modelfile))[0]
    label_file = fixtree \
                   .update(threshold=threshold, model=modelname) \
                   .get('fix_labels')
    fixlabels.saveLabelFile(labels, label_file, meldir)

    return label_file


def run_evaluate(meldir     : List[str],
                 thresholds : List[int],
                 model      : str,
                 **kwargs):

    modelname = op.splitext(op.basename(model))[0]

    for md in meldir:
        indir = io.InputDirectory.from_melodic_dir(md)

        for threshold in thresholds:

            label_file = indir.fixtree \
                .update(threshold=threshold, model=modelname) \
                .get('fix_labels')
            if not op.exists(label_file):
                run_classify(md, model, threshold)

    print(legacy.report_evaluate(meldir, model, thresholds))


def run_apply(labels     : str,
              meldir     : str   = None,
              highpass   : float = None,
              motion     : bool  = False,
              aggressive : bool  = True,
              input_tree : str   = None,
              fix_tree   : str   = None,
              **kwargs):

    if meldir is None:
        meldir = op.dirname(labels)

    indir = io.InputDirectory.from_input_dir(meldir, input_tree, fix_tree)
    ftree = indir.input_tree

    nifti_filt              = ftree.get('filtered_func_data')
    nifti_clean             = ftree.get('filtered_func_data_clean')
    nifti_clean_varnorm     = ftree.get('filtered_func_data_clean_vn')

    cifti                   = ftree.get('cifti_dtseries')
    cifti_filt              = ftree.get('cifti_preclean')
    cifti_clean             = ftree.get('cifti_clean')
    cifti_clean_varnorm     = ftree.get('cifti_clean_vn')

    icamix                  = ftree.get('melodic_mix')

    motion_parameters       = ftree.get('motion_params')
    motion_parameters_filt  = ftree.get('motion_params_filtered')
    motion_parameters_clean = ftree.get('motion_params_cleaned')

    ddremove = legacy.read_labels(labels)

    tr = clean.get_tr(nifti_filt)

    # highpass settings (get from FSF is present and highpass not provided)
    fsf = ftree.get('design')
    if (highpass is None) and op.exists(fsf):
        fsf = featanalysis.loadFsf(fsf)

        if int(fsf['temphp_yn']) == 1:
            highpass = float(fsf['paradigm_hp'])

    # check if HCP-style filtered motion parameters (*_mcf_conf_hp.nii.gz) exist
    # create if they don't
    if (not op.exists(motion_parameters_filt)) and \
       op.exists(motion_parameters):
        mp0 = clean.motion_confounds(
            motion_parameters, tr=tr, temporal_fwhm=highpass, tmpdir=meldir)
        clean.create_dummy_nifti(
            mp0.astype(np.float32), tr=tr, fname=motion_parameters_filt)

    # clean filtered_func_data.nii.gz
    clean.apply(
        func_fname=nifti_filt,
        icamix=icamix,
        noise_idx=ddremove,
        func_clean_fname=nifti_clean,
        do_motion_regression=motion,
        motion_parameters=motion_parameters_filt,
        func_tr=tr,
        aggressive=aggressive,
        varnorm_fname=nifti_clean_varnorm,
        motparams_clean_fname=motion_parameters_clean,
    )

    # clean Atlas.dtseries.nii if it exists
    if op.exists(cifti):

        dd = nb.load(cifti)
        d0_tr = clean.get_tr(dd)
        d0, mask0 = clean.get_fdata(dd)
        d0 = clean.highpass_filter(
            d0, highpass, d0_tr, tmpdir=meldir, retain_mean=True)
        clean.save_image(dd, d0, mask0, fname=cifti_filt)

        clean.apply(
            func_fname=cifti_filt,
            icamix=icamix,
            noise_idx=ddremove,
            func_clean_fname=cifti_clean,
            do_motion_regression=motion,
            motion_parameters=motion_parameters_filt,
            func_tr=tr,
            aggressive=aggressive,
            varnorm_fname=cifti_clean_varnorm,
        )


def run_main(**kwargs):
    run_feature_extraction(**kwargs)
    args.labels = run_classify(**kwargs)
    run_apply(**kwargs)


def setup_logging(logfile):

    # Make warnings emitted via the warnings
    # module get directed to the log
    logging.captureWarnings(True)

    hd  = logging.FileHandler(logfile)
    fmt = logging.Formatter(fmt='[%(asctime)s - %(name)s.%(funcName)s ] '
                            '%(levelname)s : %(message)s')
    hd.setFormatter(fmt)
    log.addHandler(hd)
    log.setLevel(logging.DEBUG)

    # make libxgboost (compiled library) quiet
    xgboost = weak_import('xgboost')
    if xgboost is not None:
        xgboost.set_config(verbosity=0)

    # Tell optuna to use the root logger
    optuna = weak_import('optuna')
    if optuna is not None:
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

        # optuna emits optuna.exceptions.ExperimentalWarning
        # regarding JournalStorage
        warnings.filterwarnings(
            'ignore', category=optuna.exceptions.ExperimentalWarning)

    # xgboost emits UserWarning regarding unused parameters
    warnings.filterwarnings('ignore', module='xgboost')


def main(argv=None):

    # options passed to all ArgumentParser objects
    opts = {'add_help'     : False,
            'allow_abbrev' : False}

    # arguments accepted by all sub-parsers
    shared_parser = argparse.ArgumentParser(**opts)
    shared_parser.add_argument('-lf', '--logfile',
                               default=op.join(os.getcwd(), 'pyfix.log'))
    shared_parser.add_argument('-it', '--input_tree',
                               default=get_resource('melodic.tree'))
    shared_parser.add_argument('-ft', '--fix_tree',
                               default=get_resource('pyfix.tree'))
    shared_parser.add_argument('-nj', '--n_jobs', type=int,
                               default=mp.cpu_count())

    opts['parents'] = [shared_parser]

    # MAIN parser
    main_parser = argparse.ArgumentParser(**opts)
    main_parser.add_argument('meldir', metavar='<mel.ica>')
    main_parser.add_argument('model', metavar='<training.pyfix_model>')
    main_parser.add_argument('threshold', metavar='<thresh>', type=int,
                             nargs='?')
    main_parser.add_argument('-m', '--motion', action='store_true')
    main_parser.add_argument('-h', '--highpass', metavar='<highpass>',
                             type=float, default=-1.0)
    main_parser.add_argument('-A', '--aggressive', action='store_true')

    # FEATURE parser
    feature_parser = argparse.ArgumentParser(**opts)
    feature_parser.add_argument('meldir', metavar='<mel.ica>')
    feature_parser.add_argument('-o', '--outdir')

    # TRAIN parser
    train_parser = argparse.ArgumentParser(**opts)
    train_parser.add_argument('basename',
                              metavar='<training-output-basename>')
    train_parser.add_argument('-l', '--loo',  action='store_true')
    train_parser.add_argument('-u', '--tune', action='store_true')
    train_parser.add_argument('-m', '--model_class')
    train_parser.add_argument('-n', '--n_trials', type=int,
                              default=100)
    train_parser.add_argument('meldir', metavar='<mel.ica>', nargs='+')

    # CLASSIFY parser
    classify_parser = argparse.ArgumentParser(**opts)
    classify_parser.add_argument('meldir', metavar='<mel.ica>')
    classify_parser.add_argument('model', metavar='<training.pyfix_model>')
    classify_parser.add_argument('threshold', metavar='<thresh>', type=int,
                                 nargs='?')

    # EVALUATE parser
    def parse_thresholds(thresholds):
        return [int(t) for t in thresholds.split(',')]

    eval_parser = argparse.ArgumentParser(**opts)
    eval_parser.add_argument('model', metavar='<training.pyfix_model>')
    eval_parser.add_argument('meldir', metavar='<mel.ica>', nargs='+')
    eval_parser.add_argument('-t', '--thresholds', metavar='1,2,5,...',
                             default='1,2,5,10,20,30,40,50',
                             type=parse_thresholds)

    # APPLY parser
    apply_parser = argparse.ArgumentParser(**opts)
    apply_parser.add_argument('labels', metavar='<labelfile>')
    apply_parser.add_argument('meldir', metavar='<mel.ica>', nargs='?')
    apply_parser.add_argument('-m', '--motion', action='store_true')
    apply_parser.add_argument('-h', '--highpass', metavar='<highpass>',
                              type=float, default=-1)
    apply_parser.add_argument('-A', '--aggressive', action='store_true')


    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 0:
        print(USAGE)
        return

    # all sub-commands except for main
    # are selected with a switch
    switch = argv[0]
    if switch.startswith('-'):
        argv = argv[1:]

    if switch == '-h':
        print(USAGE)
        return

    switches = defaultdict(
        lambda: (main_parser,     run_main),
        {'-f' : (feature_parser,  run_feature_extraction),
         '-t' : (train_parser,    run_train),
         '-c' : (classify_parser, run_classify),
         '-C' : (eval_parser,     run_evaluate),
         '-a' : (apply_parser,    run_apply)})

    parser, runfunc = switches[switch]
    args            = parser.parse_args(argv)

    setup_logging(args.logfile)
    log.info('Pyfix command: %s', ' '.join(argv))

    runfunc(**vars(args))


if __name__ == '__main__':
    main()
