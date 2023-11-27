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

import nibabel as nb
import numpy as np

from fsl.data import fixlabels
from fsl.data import melodicanalysis, featanalysis, melodicimage

from pyfix import extract, feature, legacy, io, classify, clean, __version__
from pyfix.util import weak_import


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

  fix -a <mel.ica/fix4melview_TRAIN_thr.txt>  [-m] [-h <highpass>]  [-A]

    Apply cleanup, using artefacts listed in the .txt file, to the data inside
    the enclosing Feat/Melodic directory.

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

FIXDIR = 'fix'


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


def run_feature_extraction(meldir):

    assert_is_melodic_dir(meldir)

    # instantiate FixData object
    d = extract.FixData.from_melodic_dir(
        meldir,
        fixdir=op.join(meldir, FIXDIR)
    )

    # extract legacy features
    f = extract.extract_features(
        data=d,
        features=legacy.LEGACY_FEATURE_EXTRACTORS,
    )

    # save features (pyfix and legacy-fix formats)
    io.save_features(op.join(meldir, FIXDIR, 'features'), f)
    legacy.save_features(op.join(meldir, FIXDIR), f)


def run_train(args):

    if args.tune:
        optuna = weak_import('optuna')
        if optuna is None:
            raise RuntimeError('You must install the optuna package in order '
                               'to perform hyperparameter optimisation with '
                               'the -u option')

    for d in args.meldir:
        assert_has_features(d)
        assert_has_hand_labels(d)

    file_table = legacy.generate_file_table(*args.meldir)

    #  save file table
    io.save_file_table(args.basename, file_table)

    # read and collate all features into a training table
    training_data = classify.collate_features(file_table, legacy_features=True)

    # Training with optional LOO cross validation
    if not args.tune:
        threshold = None
        clf, loo = classify.train(training_data,
                                  args.model_class,
                                  loo=args.loo,
                                  n_jobs=args.n_jobs)

    # training with automatic estimation
    # of an optimal probability threshold
    else:
        clf, loo, threshold = classify.tuned_train(training_data,
                                                   args.model_class,
                                                   loo=args.loo,
                                                   n_trials=args.n_trials,
                                                   n_jobs=args.n_jobs)

    if loo is not None:
        report = legacy.report_loo(*loo)
        with open(args.basename + "_LOO_results", 'w') as f:
            f.write(report)

    # save model
    io.save_model(args.basename, clf, threshold)


def run_classify(meldir, model, threshold):

    assert_has_features(meldir)

    fixdir = op.join(meldir, 'fix')
    if not op.exists(fixdir):
        raise RuntimeError(f'Cannot find fix directory in: {meldir}')

    features0 = legacy.read_features(op.join(fixdir, 'features.csv'))

    clf, saved_threshold = io.load_model(model)
    ypb                  = classify.classify(features0, clf)[1]

    if threshold is None:
        threshold = saved_threshold
    if threshold is None:
        raise RuntimeError('You must specify a probability threshold - '
                           f'{model} does not contain an optimal threshold.')

    y = legacy.prob_threshold(ypb, threshold=threshold / 100)

    # TODO: move legacy label saving to legacy module
    # invert y so that signal==False and noise==True (as per legacy fix4melview files)
    y = ~y
    labels = [['Noise'] if i else ['Signal'] for i in y]

    label_file = legacy.generate_label_filename(meldir, model, threshold)
    fixlabels.saveLabelFile(labels, label_file, meldir)

    return label_file


def run_evaluate(meldir, model, thresholds):

    for md in meldir:
        assert_has_hand_labels(md)

    for md, threshold in it.product(meldir, thresholds):
        label_file = legacy.generate_label_filename(md, model, threshold)
        if not op.exists(label_file):
            run_classify(md, model, threshold)

    print(legacy.report_evaluate(meldir, model, thresholds))


def run_apply(labels, motion, highpass, aggressive):

    meldir = op.dirname(labels)
    assert_is_melodic_dir(meldir)

    nifti_filt = op.join(meldir, 'filtered_func_data.nii.gz')
    nifti_clean = op.join(meldir, 'filtered_func_data_clean.nii.gz')
    nifti_clean_varnorm = op.join(meldir, 'filtered_func_data_clean_vn.nii.gz')

    cifti = op.join(meldir, 'Atlas.dtseries.nii')
    cifti_filt = op.join(meldir, 'Atlas_hp_preclean.dtseries.nii')
    cifti_clean = op.join(meldir, 'Atlas_clean.dtseries.nii')
    cifti_clean_varnorm = op.join(meldir, 'Atlas_clean_vn.dscalar.nii')

    icamix = op.join(meldir, 'filtered_func_data.ica', 'melodic_mix')

    motion_parameters = op.join(meldir, 'mc/prefiltered_func_data_mcf.par')
    motion_parameters_filt = op.join(meldir, 'mc/prefiltered_func_data_mcf_conf_hp.nii.gz')
    motion_parameters_clean = op.join(meldir, 'mc/prefiltered_func_data_mcf_conf_hp_clean.nii.gz')

    ddremove = legacy.read_labels(labels)

    tr = clean.get_tr(nifti_filt)

    # highpass settings (get from FSF is present and highpass not provided)

    if (highpass is None) and featanalysis.isFEATDir(meldir):

        fsf = featanalysis.loadSettings(meldir)
        if int(fsf['temphp_yn']) == 1:
            highpass = float(fsf['paradigm_hp'])

    # check if HCP-style filtered motion parameters (*_mcf_conf_hp.nii.gz) exist
    # create if they don't
    if (not op.exists(motion_parameters_filt)) and op.exists(motion_parameters):
        mp0 = clean.motion_confounds(motion_parameters, tr=tr, temporal_fwhm=highpass, tmpdir=meldir)
        clean.create_dummy_nifti(mp0.astype(np.float32), tr=tr, fname=motion_parameters_filt)

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
        d0 = clean.highpass_filter(d0, highpass, d0_tr, tmpdir=meldir, retain_mean=True)
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


def run_main(meldir, model, threshold, motion, highpass, aggressive):

    run_feature_extraction(meldir)
    label_fname = run_classify(meldir, model, threshold)
    run_apply(label_fname, motion, highpass, aggressive)


def set_log_file(basedir):

    log = logging.getLogger()

    logging.captureWarnings(True)

    hd  = logging.FileHandler(op.join(basedir, 'pyfix.log'))
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


def main():

    # MAIN parser
    main_parser = argparse.ArgumentParser(add_help=False)
    main_parser.add_argument('meldir', metavar='<mel.ica>')
    main_parser.add_argument('model', metavar='<training.pyfix_model>')
    main_parser.add_argument('threshold', metavar='<thresh>', type=int,
                             nargs='?')
    main_parser.add_argument('-m', '--motion', action='store_true')
    main_parser.add_argument('-h', '--highpass', metavar='<highpass>',
                             type=float, default=-1.0)
    main_parser.add_argument('-A', '--aggressive', action='store_true')

    # FEATURE parser
    feature_parser = argparse.ArgumentParser(add_help=False)
    feature_parser.add_argument('meldir', metavar='<mel.ica>')

    # TRAIN parser
    train_parser = argparse.ArgumentParser(add_help=False)
    train_parser.add_argument('basename',
                              metavar='<training-output-basename>')
    train_parser.add_argument('-l', '--loo',  action='store_true')
    train_parser.add_argument('-u', '--tune', action='store_true')
    train_parser.add_argument('-m', '--model_class')
    train_parser.add_argument('-n', '--n_trials', type=int,
                              default=100)
    train_parser.add_argument('-j', '--n_jobs', type=int,
                              default=mp.cpu_count())
    train_parser.add_argument('meldir', metavar='<mel.ica>', nargs='+')

    # CLASSIFY parser
    classify_parser = argparse.ArgumentParser(add_help=False)
    classify_parser.add_argument('meldir', metavar='<mel.ica>')
    classify_parser.add_argument('model', metavar='<training.pyfix_model>')
    classify_parser.add_argument('threshold', metavar='<thresh>', type=int,
                                 nargs='?')

    # EVALUATE parser
    def parse_thresholds(thresholds):
        return [int(t) for t in thresholds.split(',')]

    eval_parser = argparse.ArgumentParser(add_help=False)
    eval_parser.add_argument('model', metavar='<training.pyfix_model>')
    eval_parser.add_argument('meldir', metavar='<mel.ica>', nargs='+')
    eval_parser.add_argument('-t', '--thresholds', metavar='1,2,5,...',
                             default='1,2,5,10,20,30,40,50',
                             type=parse_thresholds)

    # APPLY parser
    apply_parser = argparse.ArgumentParser(add_help=False)
    apply_parser.add_argument('labels', metavar='<mel.ica/fix4melview_TRAIN_thr.txt>')
    apply_parser.add_argument('-m', '--motion', action='store_true')
    apply_parser.add_argument('-h', '--highpass', metavar='<highpass>', type=float, default=-1.0)
    apply_parser.add_argument('-A', '--aggressive', action='store_true')

    if len(sys.argv) == 1:
        print(USAGE)
        sys.exit()

    switch = sys.argv[1]

    if switch == '-h':
        print(USAGE)
        sys.exit()
    elif switch == '-f':
        args = feature_parser.parse_args(sys.argv[2:])
        set_log_file(args.meldir)
        run_feature_extraction(**vars(args))
    elif switch == '-t':
        args = train_parser.parse_args(sys.argv[2:])
        set_log_file(os.getcwd())
        run_train(args)
    elif switch == '-c':
        args = classify_parser.parse_args(sys.argv[2:])
        set_log_file(args.meldir)
        run_classify(**vars(args))
    elif switch == '-C':
        args = eval_parser.parse_args(sys.argv[2:])
        set_log_file(os.getcwd())
        run_evaluate(**vars(args))
    elif switch == '-a':
        args = apply_parser.parse_args(sys.argv[2:])
        set_log_file(op.dirname(args.labels))
        run_apply(**vars(args))
    else:
        args = main_parser.parse_args(sys.argv[1:])
        set_log_file(args.meldir)
        run_main(**vars(args))


if __name__ == '__main__':
    main()
