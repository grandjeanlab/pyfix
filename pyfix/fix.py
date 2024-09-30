#!/usr/bin/env python

"""
FIX (FMRIB's ICA-based Xnoiseifier).
"""

import argparse
import calendar
import contextlib
import datetime
import functools as ft
import logging
import multiprocessing as mp
import os
import os.path as op
import sys
import warnings
from collections import defaultdict

from typing import List

import nibabel as nb

from fsl.data import fixlabels
from fsl.data import melodicanalysis, featanalysis
from fsl.wrappers import wrapperconfig

from pyfix      import (extract,
                        feature,
                        legacy,
                        io,
                        classify,
                        clean,
                        report,
                        __version__)
from pyfix.util import (weak_import,
                        get_resource,
                        get_builtin_model,
                        list_models)


log = logging.getLogger('pyfix.fix')


USAGE = f"""FIX (FMRIB's ICA-based Xnoiseifier) [version {__version__}]

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

    -m  optionally also cleanup motion confounds, with highpass filtering of
        motion confounds controlled by -h

    -h  Highpass filtering of motion confounds and CIfTI time series data.
         - if -h is omitted, fix will use the highpass cutoff in a design.fsf
           file, ifpresent.
         - if -h is omitted, and no design.fsf is present, no filtering of the
           motion confounds or CIfTI data will take place.
         - If -h -1, no highpass filtering will be performed.
         - -h 0 apply linear detrending only.
         - -h <highpass> with a positive <highpass> value, apply highpass with
           <highpass> being full-width (2*sigma) in seconds.

    -A  apply aggressive (full variance) cleanup, instead of the default less-
        aggressive (unique variance) cleanup.

The following built-in models can be used for classification:
""" + '\n'.join(f' - {m}' for m in list_models())

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
                           species    : str = None,
                           input_tree : str = None,
                           fix_tree   : str = None,
                           outdir     : str = None,
                           **kwargs):

    assert_is_melodic_dir(meldir)

    # instantiate FixData object
    indir = io.InputDirectory.from_input_dir(
        meldir, input_tree, fix_tree, outdir, species)
    d     = extract.FixData.from_input_dir(indir)

    # extract legacy features
    f = extract.extract_features(
        data=d,
        features=sorted(feature.FEATURE_EXTRACTORS.keys()))

    # save features
    io.save_features(d.fix_tree.get('features'), f)


def run_train(meldir          : List[str],
              basename        : str,
              model_class     : str       = None,
              feature_columns : List[str] = None,
              species         : str       = None,
              loo             : bool      = False,
              tune            : bool      = False,
              n_jobs          : int       = None,
              n_trials        : int       = None,
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

    file_table = io.generate_file_table(*meldir)

    # save file table
    io.save_file_table(basename, file_table)

    # read and collate all features into a training table
    training_data, feature_columns = classify.collate_features(
        file_table, feature_columns)

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
        report_text = report.report_loo(*loo)
        with open(basename + "_LOO_results", 'w') as f:
            f.write(report_text)

    # save model
    model = io.Model(clf, feature_columns, species, threshold)
    io.save_model(basename, model)
    return model


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

    model    = io.load_model(model)
    features = io.read_features(fixtree.get('features'), model.features)
    ypb      = classify.classify(features, model.model)[1]

    if threshold is None:
        threshold = model.threshold

    if threshold is None:
        raise RuntimeError('You must specify a probability threshold - '
                           f'{model.filename} does not contain an optimal '
                           'threshold.')

    # threshold classifier probabilities
    # for the signal class (column 1)
    # so that signal==True and noise==False
    y   = legacy.prob_threshold(ypb, threshold=threshold / 100, col=1)

    # Save generated labels to FIX label file.
    # Invert y so that signal==False and noise==True
    # (as per legacy fix4melview files)
    y          = ~y
    ypb        = ypb[:, 1]
    labels     = [['Noise'] if i else ['Signal'] for i in y]
    label_file = intree \
                   .update(threshold=threshold, model=model.name) \
                   .get('fix_labels')
    icadir     = intree.get('melodic_dir')
    icadir     = op.relpath(icadir, meldir)

    fixlabels.saveLabelFile(labels, label_file, icadir, probabilities=ypb)

    return label_file


def run_evaluate(meldir     : List[str],
                 thresholds : List[int],
                 model      : str,
                 **kwargs):

    modelname = op.splitext(op.basename(model))[0]

    for md in meldir:
        indir = io.InputDirectory.from_input_dir(md)

        for threshold in thresholds:

            label_file = indir.input_tree \
                .update(threshold=threshold, model=modelname) \
                .get('fix_labels')
            if not op.exists(label_file):
                run_classify(md, model, threshold)

    print(report.report_evaluate(meldir, model, thresholds))


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
    motion_parameters_nifti = ftree.get('motion_params_nifti')
    motion_parameters_filt  = ftree.get('motion_params_filtered')
    motion_parameters_clean = ftree.get('motion_params_cleaned')

    ddremove = io.read_labels(labels)

    tr = clean.get_tr(nifti_filt)

    # highpass settings (get from FSF is present and highpass not provided)
    fsf = ftree.get('design')
    if (highpass is None) and op.exists(fsf):
        fsf = featanalysis.loadFsf(fsf)
        if int(fsf['temphp_yn']) == 1:
            highpass = float(fsf['paradigm_hp'])

    # -h -1 -> disable highpass
    if highpass == -1:
        highpass = None

    # clean filtered_func_data.nii.gz
    clean.apply(
        func_fname=nifti_filt,
        icamix=icamix,
        noise_idx=ddremove,
        func_clean_fname=nifti_clean,
        do_motion_regression=motion,
        motion_parameters=motion_parameters,
        func_tr=tr,
        highpass=highpass,
        aggressive=aggressive,
        varnorm_fname=nifti_clean_varnorm,
        motparams_nifti_fname=motion_parameters_nifti,
        motparams_filt_fname=motion_parameters_filt,
        motparams_clean_fname=motion_parameters_clean,
    )

    # clean Atlas.dtseries.nii if it exists
    if op.exists(cifti):

        dd = nb.load(cifti)
        d0_tr = clean.get_tr(dd)
        d0, mask0 = clean.get_fdata(dd)

        if highpass is not None:
            d0 = clean.highpass_filter(
                d0, highpass, d0_tr, tmpdir=meldir, retain_mean=True)

        clean.save_image(dd, d0, mask0, fname=cifti_filt)

        clean.apply(
            func_fname=cifti_filt,
            icamix=icamix,
            noise_idx=ddremove,
            func_clean_fname=cifti_clean,
            do_motion_regression=motion,
            motion_parameters=motion_parameters,
            func_tr=tr,
            aggressive=aggressive,
            varnorm_fname=cifti_clean_varnorm,
        )


def run_main(meldir    : str,
             model     : str,
             threshold : int,
             species   : str = None,
             **kwargs):

    # The species is saved in a model file, so if
    # it hasn't been specified on the CLI, default
    # to whatever is stored in the model file.
    if species is None:
        species = io.load_model(model).species

    run_feature_extraction(meldir, species, **kwargs)
    labels = run_classify(meldir, model, threshold, **kwargs)
    run_apply(labels, meldir, **kwargs)


def setup_logging(logfile, argv):

    # Make warnings emitted via the warnings
    # module get directed to the log
    logging.captureWarnings(True)

    logger = logging.getLogger('pyfix')
    hd     = logging.FileHandler(logfile)
    fmt    = logging.Formatter('%(asctime)s '
                               '%(levelname)8.8s '
                               '%(filename)15.15s '
                               '%(lineno)4d: '
                               '%(message)s',
                               '%H:%M:%S')
    hd.setFormatter(fmt)
    logger.addHandler(hd)
    logger.setLevel(logging.DEBUG)

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

    date = datetime.date.today()
    if argv is None:
        argv = list(sys.argv)

    logger.info('pyfix    %s', __version__)
    logger.info('date:    %s (%s)', date, calendar.day_name[date.weekday()])
    logger.info('pwd:     %s', os.getcwd())
    logger.info('command: %s', ' '.join(argv))


@contextlib.contextmanager
def logwrappers():
    """Context manager which configures fsl.wrapper functions so that
    called commands and all standard output/error are logged to the
    pyfix log.
    """

    def logoutput(what, output):
        output = output.strip()
        if output != '':
            log.debug('%s: %s', what, output, stacklevel=2)

    with wrapperconfig(log={
            'tee'    : False,
            'cmd'    : ft.partial(logoutput, 'Calling command'),
            'stdout' : ft.partial(logoutput, 'Standard output'),
            'stderr' : ft.partial(logoutput, 'Standard error')}):
        yield


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
    main_parser.add_argument('-s', '--species')
    main_parser.add_argument('-m', '--motion', action='store_true')
    main_parser.add_argument('-h', '--highpass', metavar='<highpass>',
                             type=float, default=None)
    main_parser.add_argument('-A', '--aggressive', action='store_true')

    # FEATURE parser
    feature_parser = argparse.ArgumentParser(**opts)
    feature_parser.add_argument('meldir', metavar='<mel.ica>')
    feature_parser.add_argument('-o', '--outdir')
    feature_parser.add_argument('-s', '--species')

    # TRAIN parser
    train_parser = argparse.ArgumentParser(**opts)
    train_parser.add_argument('basename',
                              metavar='<training-output-basename>')
    train_parser.add_argument('-l', '--loo',  action='store_true')
    train_parser.add_argument('-u', '--tune', action='store_true')
    train_parser.add_argument('-m', '--model_class')
    train_parser.add_argument('-n', '--n_trials', type=int,
                              default=100)
    train_parser.add_argument('-s', '--species')
    train_parser.add_argument('-f', '--feature_columns',
                              type=io.read_feature_columns)
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
    eval_parser.add_argument('-f', '--feature_columns',
                             type=io.read_feature_columns)

    # APPLY parser
    apply_parser = argparse.ArgumentParser(**opts)
    apply_parser.add_argument('labels', metavar='<labelfile>')
    apply_parser.add_argument('meldir', metavar='<mel.ica>', nargs='?')
    apply_parser.add_argument('-m', '--motion', action='store_true')
    apply_parser.add_argument('-h', '--highpass', metavar='<highpass>',
                              type=float, default=None)
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

    # allow built-in models to be selected by
    # name, and auto-load associated settings
    if hasattr(args, 'model') and (not op.exists(args.model)):
        model      = get_builtin_model(args.model)
        args.model = model['file']

    setup_logging(args.logfile, argv)

    with logwrappers():
        runfunc(**vars(args))


if __name__ == '__main__':
    main()
