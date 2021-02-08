#!/usr/bin/env python
""" FIX (FMRIB's ICA-based Xnoiseifier).

Usage:
    fix -f <mel.ica>...
    fix -t <training-output-basename> [-l] <mel.ica>...
    fix -c <mel.ica> <training.pyfix_model> <thresh>


"""

import argparse
import sys
from pyfix import extract, feature, legacy, io, classify, clean
import os
import os.path as op
import logging
from fsl.data import fixlabels
from fsl.data import melodicanalysis, featanalysis, melodicimage
import nibabel as nb
import numpy as np

USAGE = '''
FIX (FMRIB's ICA-based Xnoiseifier)

Simple usage, assuming training data already exists:

  fix <mel.ica> <train>.RData <thresh>  [fix -a options]
    Extract features, classify and run cleanup - equivalent to:
      fix -f <mel.ica> ; fix -c <mel.ica> <train>.RData <thresh> ; fix -a <mel.ica>/fix4melview_<train>_<thresh>.txt [-a options]
    For the [fix -a options], see "fix -a" usage below.
    The first argument should be a FEAT/MELODIC output directory that includes a filtered_func_data.ica melodic output.
    Example:    fix study1/subject1.ica /usr/local/FIX/HCP_noHP.RData 20 -m -h 200
    Example:    fix subject2.feat /home/FIX/Standard_noHP.RData 20 -m   (ICA must have been turned on in FEAT pre-stats)

Usage for each stage separately, including creation of training data:

  fix -f <mel.ica>
    Feature extraction (for later training and/or classifying).
    
  fix -t <training-output-basename> [-l] <mel1.ica> <mel2.ica> ...
    Train classifier (Training output basename followed by list of Melodic directories).
    Every Melodic directory must contain hand_labels_noise.txt listing the artefact components, e.g.:  [1, 4, 99, ... 140].
    -l : optional LOO accuracy testing.
    
  fix -c <mel.ica> <training.pyfix_model>
    Classify ICA components.
    
  fix -a <mel.ica/fix4melview_TRAIN_thr.txt>  [-m] [-h <highpass>]  [-A]
    Apply cleanup, using artefacts listed in the .txt file, to the data inside the enclosing Feat/Melodic directory.
    -h -1          apply no highpass filtering.
    -h 0           apply linear detrending only.
    -h <highpass>  with a positive <highpass> value, apply highpass with <highpass> being full-width (2*sigma) in seconds.
    -m : optionally also cleanup motion confounds, with highpass filtering of motion confounds controlled by -h:
       - if -h is omitted, fix will look to see if a design.fsf file is present, to find the highpass cutoff.
       - if -h is omitted, and no design.fsf is present, no filtering of the motion confounds will take place.
    -A : apply aggressive (full variance) cleanup, instead of the default less-aggressive (unique variance) cleanup.
'''

FIXDIR = 'fix'


def assert_is_melodic_dir(meldir):

    if melodicanalysis.isMelodicDir(meldir):
        return


    if featanalysis.hasMelodicDir(meldir):
        return

    raise RuntimeError(f'Not a conforming melodic directory: {meldir}')


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


def run_train(basename, meldir, loo):

    for d in meldir:
        assert_is_melodic_dir(d)

    file_table = legacy.generate_file_table(*meldir)

    #  save file table
    io.save_file_table(basename, file_table)

    # read and collate all features into a training table
    training_data = classify.collate_features(file_table, legacy_features=True)

    # TODO: save collated features

    # classify
    clf = classify.train(training_data, loo=loo)[0]

    # save model
    io.save_model(basename, clf)


def run_classify(meldir, model, threshold=None):

    assert_is_melodic_dir(meldir)

    fixdir = op.join(meldir, 'pyfix')
    if not op.exists(fixdir):
        fixdir = op.join(meldir, 'fix')
        if not op.exists(fixdir):
            raise RuntimeError(f'Cannot find fix directory in: {meldir}')

    features0 = legacy.read_features(op.join(fixdir, 'features.csv'))

    clf = io.load_model(model)
    Y = classify.classify(features0, clf)

    # TODO: move legacy label saving to legacy module
    labels = [['Noise'] if i else ['Signal'] for i in Y]

    label_fname = f'{meldir}/fix4melview_{op.splitext(op.basename(model))[0]}.txt'
    fixlabels.saveLabelFile(labels, label_fname, meldir)

    return label_fname


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
    motion_parameters_expanded = op.join(meldir, 'mc/prefiltered_func_data_mcf_conf.nii.gz')
    motion_parameters_filt = op.join(meldir, 'mc/prefiltered_func_data_mcf_conf_hp.nii.gz')
    motion_parameters_clean = op.join(meldir, 'mc/prefiltered_func_data_mcf_conf_hp_clean.nii.gz')

    ddremove = legacy.read_labels(labels)

    tr = clean.get_tr(nifti_filt)

    # highpass settings (get from FSF is present and highpass not provided)

    if (highpass is None) and featanalysis.isFEATDir(meldir):

        fsf = featanalysis.loadSettings(meldir)
        if int(fsf['temphp_yn']) == 1:
            highpass = float(fsf['paradigm_hp'])

    # legacy FIX expands and filters the motion confounds irrespective of whether they are used
    # for motion regression or not, and saves them as nifti

    if not op.exists(motion_parameters_filt):

        mp = clean.load_motion_parameters(motion_parameters)
        conf = clean.expand_motion_parameters(mp)
        clean.create_dummy_nifti(conf.astype(np.float32), tr=tr, fname=motion_parameters_expanded)

        if highpass is not None:
            conf = clean.highpass_filter(conf, highpass, tr, tmpdir=meldir)
            clean.create_dummy_nifti(conf.astype(np.float32), tr=tr, fname=motion_parameters_filt)

    # clean filtered_func_data.nii.gz
    clean.apply(
        func_fname=nifti_filt,
        icamix=icamix,
        noise_idx=ddremove,
        func_clean_fname=nifti_clean,
        do_motion_regression=motion,
        do_filter_func=False,
        do_filter_motparams=False,
        temporal_fwhm=highpass,
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
            do_filter_func=False,
            do_filter_motparams=False,
            temporal_fwhm=highpass,
            motion_parameters=motion_parameters_filt,
            func_tr=tr,
            aggressive=aggressive,
            varnorm_fname=cifti_clean_varnorm,
        )


def run_main(meldir, model, motion, highpass, aggressive):

    run_feature_extraction(meldir)
    label_fname = run_classify(meldir, model)
    run_apply(label_fname, motion, highpass, aggressive)


if __name__ == '__main__':

    logging.basicConfig(
        # filename='myapp.log',
        level=logging.INFO,
    )

    # MAIN parser
    main_parser = argparse.ArgumentParser(add_help=False)
    main_parser.add_argument('meldir', metavar='<mel.ica>')
    main_parser.add_argument('model', metavar='<training.pyfix_model>')
    # main_parser.add_argument('threshold', metavar='<thresh>', type=int)
    main_parser.add_argument('-m', '--motion', action='store_true')
    main_parser.add_argument('-h', '--highpass', metavar='<highpass>', type=float, default=-1.0)
    main_parser.add_argument('-A', '--aggressive', action='store_true')

    # FEATURE parser
    feature_parser = argparse.ArgumentParser(add_help=False)
    feature_parser.add_argument('meldir', metavar='<mel.ica>')

    # TRAIN parser
    train_parser = argparse.ArgumentParser(add_help=False)
    train_parser.add_argument('basename', metavar='<training-output-basename>')
    train_parser.add_argument('-l', '--loo', action='store_true')
    train_parser.add_argument('meldir', metavar='<mel.ica>', nargs='+')

    # CLASSIFY parser
    classify_parser = argparse.ArgumentParser(add_help=False)
    classify_parser.add_argument('meldir', metavar='<mel.ica>')
    classify_parser.add_argument('model', metavar='<training.pyfix_model>')
    # classify_parser.add_argument('threshold', metavar='<thresh>', default=None)

    # APPLY parser
    apply_parser = argparse.ArgumentParser(add_help=False)
    apply_parser.add_argument('labels', metavar='<mel.ica/fix4melview_TRAIN_thr.txt>')
    apply_parser.add_argument('-m', '--motion', action='store_true')
    apply_parser.add_argument('-h', '--highpass', metavar='<highpass>', type=float, default=-1.0)
    apply_parser.add_argument('-A', '--aggressive', action='store_true')

    #  PARSE

    if len(sys.argv) == 1:
        print(USAGE)
        sys.exit()

    switch = sys.argv[1]

    if switch == '-f':
        args = feature_parser.parse_args(sys.argv[2:])
        run_feature_extraction(**vars(args))
    elif switch == '-h':
        print(USAGE)
        sys.exit()
    elif switch == '-t':
        args = train_parser.parse_args(sys.argv[2:])
        run_train(**vars(args))
    elif switch == '-c':
        args = classify_parser.parse_args(sys.argv[2:])
        run_classify(**vars(args))
    elif switch == '-a':
        args = apply_parser.parse_args(sys.argv[2:])
        run_apply(**vars(args))
    else:
        args = main_parser.parse_args(sys.argv[1:])
        run_main(**vars(args))


    # print(vars(args))