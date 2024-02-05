#!/usr/bin/env python

from __future__ import annotations

import              base64
import              csv
import functools as ft
import              io
import              json
import              logging
import os.path   as op
import              os
import              tempfile
import              warnings

from dataclasses import dataclass
from typing      import List, Any

import                  joblib
import nibabel   as     nb
import numpy     as     np
import pandas    as     pd
import                  sklearn
from   file_tree import FileTree
from   fsl.data  import fixlabels

import                    pyfix
from   pyfix.util import (weak_import,
                          get_resource)
from   pyfix      import (legacy,
                          feature)

# InconsistentVersionWarning was added in sklearn 1.3.0
try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:
    InconsistentVersionWarning = UserWarning


xgboost = weak_import('xgboost')


log = logging.getLogger(__name__)


def read_file_table(filename : str) -> pd.DataFrame:
    """Read a table of files for training.

    Args:
        filename:

    Returns:

    """
    return pd.read_csv(filename, index_col=None)


def save_file_table(filename : str, tbl : pd.DataFrame):
    """
    Save pyFIX features to a file
    """

    if not filename.endswith('.file_table'):
        filename = filename + '.file_table'

    if op.exists(filename):
        log.warning(f'File table already exists: {filename}')

    workdir = op.expanduser(op.dirname(filename))
    if workdir == '':
        workdir = './'
    if not op.exists(workdir):
        os.makedirs(workdir)

    log.info(f'Saving file table to: {filename}')
    tbl.to_csv(filename, index=None)


def generate_file_table(*args):
    """Generate a table of files from FIX folders.

    Args:
        *args:

    Returns:

    """

    d = pd.DataFrame(
        {
            'features': [op.join(f, 'fix/features.csv') for f in args],
            'labels': [op.join(f, 'hand_labels_noise.txt') for f in args],
            'subid': [i for i, _ in enumerate(args)]
        }
    )

    return d


@dataclass(frozen=True)
class InputDirectory:
    """Data structure which represents a FIX input directory. """

    indir      : str
    fixdir     : str
    input_tree : FileTree
    fix_tree   : FileTree
    species    : str

    @staticmethod
    def from_input_dir(
            input_dir  : str,
            input_tree : str = None,
            fix_tree   : str = None,
            fix_dir    : str = None,
            species    : str = None
    ) -> InputDirectory:

        if input_tree is None: input_tree = get_resource('melodic.tree')
        if fix_tree   is None: fix_tree   = get_resource('pyfix.tree')

        input_tree = FileTree.read(input_tree, input_dir)

        if fix_dir is None:
            fix_dir = input_tree.get('fixdir')

        fix_tree  = FileTree.read(fix_tree, fix_dir)
        input_dir = op.abspath(input_dir)
        fix_dir   = op.abspath(fix_dir)

        return InputDirectory(input_dir, fix_dir, input_tree, fix_tree, species)


    @property
    @ft.lru_cache
    def nics(self):
        """Return the number of ICs for this input directory."""

        melic = self.input_tree.get('melodic_IC')
        if op.exists(melic):
            return nb.load(melic).shape[3]

        features = self.fix_tree.get('features')
        if op.exists(features):
            features = read_features(features)
            return features['nic'].loc[0]

        raise RuntimeError(f'Cannot determine nics for {self.input_dir}')


def load_motion_parameters(fname : str) -> pd.DataFrame:
    """Load motion parameters from one of:
      - MCFLIRT output files
      - eddy output files
      - BIDS motion parameter files

    Args:
       fname: File containing motion parameters

    Returns:
       pandas.DataFrame containing trans_[x,y,z] and rot_[x,y,z] columns.
    """
    log.info(f'Load motion parameters: {fname}')

    if fname.endswith('.par'):

        log.info('Looks like a mcflirt motion parameter file (*.par)')
        mp = np.loadtxt(fname)
        mp = pd.DataFrame(mp, columns=['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z'])

    elif fname.endswith('_motion.tsv'):

        log.info('Looks like BIDS motion parameter file (*_motion.tsv)')
        mp = pd.read_csv(fname, delimiter='\t', index_col=None)

    elif fname.endswith('.eddy_parameters'):

        log.info('Looks like EDDY motion parameter file (*.eddy_parameters)')
        mp = np.loadtxt(fname)
        mp = pd.DataFrame(mp, columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'])

    else:
        raise RuntimeError('Unknown motion parameter filetype')

    return mp


@dataclass
class Model:
    """Class which represents a pyfix model, used by save_model and load_model.
    This class is just a container with a reference to an xgboost/sklearn
    classifier, and information needed to use it.
    """

    model : Any
    """A classifier object, such as a sklearn or xgboost classifier. """

    features : List[str]
    """A list of feature labels denoting the pyfix features that were used to
    train the classifier.
    """

    species : str
    """Species the model was trained on. At the moment this simply determines
    which set of mask images were used in feature extraction.
    """

    threshold : float = None
    """An optimal probability threshold to use for classification,
    or None if an optimal threshold was not estimated.
    """

    filename : str = None
    """File this model was loaded from. May be None. """

    def __init__(self,
                 model     : Any,
                 features  : List[str] = None,
                 species   : str       = None,
                 threshold : float     = None,
                 filename  : str       = None):
        """Create a Model object. """

        if features is None:
            features = feature.FEATURE_LABELS

        if species in (None, ''):
            species = 'mni152_1mm'

        self.model     = model
        self.features  = list(features)
        self.species   = species
        self.threshold = threshold
        self.filename  = filename


    @property
    def name(self):
        """Return a name for this Model generated from the saved file,
        or "none" if it is not saved.
        """
        if self.filename is None:
            return 'none'
        else:
            return op.splitext(op.basename(self.filename))[0]

    @property
    def model_type(self) -> str:
        """Return a string identifier for the model, either 'xgboost' or
        'sklearn'.
        """

        if (xgboost is not None) and \
           isinstance(self.model, xgboost.XGBClassifier):
            return 'xgboost'
        elif isinstance(self.model, sklearn.base.ClassifierMixin):
            return 'sklearn'
        else:
            raise RuntimeError(
                f'Unsupported model {type(model)}! Pyfix only '
                'recognises scikit-learn or xgboost classifiers')


    def dump_model(self) -> bytes:
        """Return a binary representation of the classifier object. """

        if self.model_type == 'xgboost':
            # xgboost.save_model doesn't have the
            # ability to dump to memory, so we save
            # to a temp file and read it back in
            with tempfile.TemporaryDirectory() as tdir:
                tmpfile = op.join(tdir, 'model.json')
                self.model.save_model(tmpfile)
                with open(tmpfile, 'rb') as f:
                    dumped_model = f.read()

        elif self.model_type == 'sklearn':
            with io.BytesIO() as f:
                joblib.dump(self.model, f)
                dumped_model = f.getvalue()

        return dumped_model


def save_model(fname : str, model : Model):
    """Save a pyFIX model to file.

    Args:
        fname: file to which the model should be saved
        model: Model to save

    Note that the pyfix model file format is subject to change without notice
    between versions. However, the developers strive to guarantee backwards-
    compatibility (notwithstanding the limitations imposed by scikit-learn as
    outlined below), so old pyfix model files should be able to be loaded.
    Saved pyfix model files are *not* guaranteed to be forward-compatible -
    i.e. a model file saved with a particular pyfix version is not guaranteed
    to work with an older version of pyfix.

    Pyfix 0.5.0
    -----------

    Pyfix models are saved in the same manner as Pyfix 0.4.0 models, but with
    two additional fields, called "features" and "species":

        {
            "pyfix_version"   : "<pyfix-version>",
            "sklearn_version" : "<sklearn-version>",
            "xgboost_version" : "<xgboost-version>",
            "features"        : ["<featurelist>"],
            "species"         : "<data-species>",
            "model_type"      : "<model-type>",
            "model_contents"  : "<model-contents>",
            "model_threshold" : "<model-threshold>"  # optional
        }

    where:

      - "<featurelist>" is a JSON list containing the names of all pyfix
        features that were used to train the model.
      - "<data-species>" is a species identifier, e.g "mni152_1mm", "macaque",
        etc.

    Pyfix 0.4.0
    -----------

    Pyfix models are saved as a plain-text file which begins with "PYFIX\n" and
    then contains the following JSON-encoded structure:

        {
            "pyfix_version"   : "<pyfix-version>",
            "sklearn_version" : "<sklearn-version>",
            "xgboost_version" : "<xgboost-version>",
            "model_type"      : "<model-type>",
            "model_contents"  : "<model-contents>",
            "model_threshold" : "<model-threshold>"  # optional
        }

    where:

      - <pyfix-version> is the version of pyfix used to save the model
      - <sklearn-version> is the version of scikit-learn used to save the model
        (if relevant)
      - <xgboost-version> is the version of xgboost used to save the model (if
        relevant)
      - <model-type> is one of "xgboost" or "sklearn"
      - <model-contents> is the base64-encoded serialised model object.
      - <model-threshold> is the optimal threshold to use, automatically
        determined when tuning a model with optuna.

    At the moment we are able to save:

     - xgboost.XGBoostClassifier models. These are saved in the "Universal
       Binary JSON" format, which is a backwards-compatible format - i.e. a
       model created and saved with a particular version of xgboost is able
       to be loaded and used with newer versions of xgboost.

     - scikit-learn classifiers. At the moment these are saved using
       joblib.dump, which unfortunately means that they are only guaranteed
       to work with the same scikit-learn version.

    See these pages for more details:

      - https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
      - https://scikit-learn.org/stable/model_persistence.html
      - https://joblib.readthedocs.io/en/stable/persistence.html#persistence


    Pyfix 0.3.0
    -----------

    Pyfix models are saved using joblib.dump. A saved model file is only
    guaranteed to work with the same scikit-learn version.
    """

    if not fname.endswith('.pyfix_model'):
        fname = fname + '.pyfix_model'

    dumped_model = base64.b64encode(model.dump_model()).decode()
    to_save      = {
        'pyfix_version'   : pyfix.__version__,
        'sklearn_version' : sklearn.__version__,
        'xgboost_version' : getattr(xgboost, '__version__', ''),
        'features'        : model.features,
        'species'         : model.species,
        'model_type'      : model.model_type,
        'model_contents'  : dumped_model,
    }
    if model.threshold is not None:
        to_save['model_threshold'] = model.threshold

    if op.exists(fname):
        log.warning(f'Model file already exists - overwriting: {fname}')

    fdir = op.abspath(op.expanduser(op.dirname(fname)))
    if not op.exists(fdir):
        os.makedirs(fdir)

    log.info(f'Saving trained model to: {fname}')
    with open(fname, 'wt', encoding='utf-8') as f:
        f.write('PYFIX\n')
        json.dump(to_save, f)

    model.filename = fname


def load_model(fname : str) -> Model:
    """Load pyFIX model. See save_model for important information on the
    pyfix model file format, and compatibility considerations.

    Args:
        fname: File to load

    Returns a Model object
    """

    sklearn_warning = (
        'Pyfix model file {fname} was saved with a different version of '
        'scikit-learn [{oldver}]! You are using scikit-learn {skver}. You '
        'may  experience errors loading or using this model. If you '
        'experience errors, the only work-around is to use the same version '
        'of scikit-learn that was used to save the model.')

    # pyfix < 0.4.0 - a sklearn classifier
    # stored with raw joblib.dump output.
    def load_old_pyfix_model_file():
        log.debug(f'Loading pyfix < 0.4.0 file using joblib: {fname}')

        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)

            try:
                model = joblib.load(fname)
            except InconsistentVersionWarning as e:
                skver  = sklearn.__version__
                oldver = getattr(e, 'original_sklearn_version', '???')

                log.warning(sklearn_warning.format(
                    fname=fname, skver=skver, oldver=oldver))

                model = joblib.load(fname)

        return Model(model, None, None, None, fname)

    # pyfix >= 0.4.0 file - format as
    # described in the save_model docs.
    def load_new_pyfix_model_file():
        log.debug(f'Loading pyfix >= 0.4.0 file: {fname}')

        with open(fname, 'rt') as f:
            f.seek(6)
            loaded = json.load(f)

        mtype    = loaded['model_type']
        mthres   = loaded.get('model_threshold', None)
        species  = loaded.get('species', None)
        features = loaded.get('features', feature.FEATURE_LABELS)
        mdata    = base64.b64decode(loaded['model_contents'].encode('utf-8'))

        if mtype == 'xgboost':

            if xgboost is None:
                raise RuntimeError(f'Pyfix model file {fname} contains a '
                                   'xgboost classifier, but xgboost is not '
                                   'installed! You need to install xgboost '
                                   'in order to use this model.')

            model = xgboost.XGBClassifier()
            model.load_model(bytearray(mdata))

        elif mtype == 'sklearn':

            oldver = loaded['sklearn_version']
            skver  = sklearn.__version__
            if skver != oldver:
                log.warning(sklearn_warning.format(
                    fname=fname, skver=skver, oldver=oldver))

            with io.BytesIO(mdata) as f:
                model = joblib.load(f)
        else:
            raise RuntimeError(f'Unrecognised model {type(model)}! This '
                               f'version of Pyfix ({pyfix.__version__}) '
                               'only recognises scikit-learn or xgboost '
                               'classifiers')

        return Model(model, features, species, mthres, fname)

    # Sniff the first few bytes of
    # the file to see what we have
    with open(fname, 'rb') as f:
        header = f.read(6)

    if header == b'PYFIX\n':
        return load_new_pyfix_model_file()
    else:
        return load_old_pyfix_model_file()


def generate_label_filename(meldir, modelfile, threshold):
    """Create a FIX/melview-style file name for storing IC labels. """
    model     = op.splitext(op.basename(modelfile))[0]
    threshold = int(threshold)
    return op.join(meldir, f'fix4melview_{model}_thr{threshold}.txt')


def read_labels(filename: str, nics : int = None) -> np.ndarray:
    """Read legacy label files, melview and hand_labels_noise.txt.

    If nics is None, a sequence of one-indexed noise IC labels is returned.
    Otherwise, a sequence with a value for each IC is returned, containing
    1 for noise ICs, and 0 for signal ICs.
    """

    indices = fixlabels.loadLabelFile(filename, returnIndices=True)[2]
    indices = np.array(indices, dtype=int)

    if nics is None:
        return indices
    else:
        labels = np.zeros(nics)
        labels[indices - 1] = 1

    return labels


def read_features(path: str, columns : List[str] = None):
    """Read legacy FIX features from file.

    Args:
        path (str):     Path to feature file
        columns (list): List of column names - defaults to
                        pyfix.feature.FEATURE_LABELS

    Returns:
        features table (pd.DataFrame):
    """

    log.debug(f'Loading feature file {path}')

    with open(path, 'rt') as f:
        data = f.read()

    # do we have column names?
    header = csv.Sniffer().has_header(data)
    ncols  = len(data.split('\n')[0].split(','))

    # If columns have been provided, we assume
    # that the user knows what they're doing.
    if columns is not None:
        header = 0 if header else None

        log.debug(f'Expecting a custom set of {len(columns)} columns')

    else:
        # Otherwise, if the file has no header,
        # we assume it has FEATURE_LABELS columns
        if not header:
            columns = feature.FEATURE_LABELS
            header  = None
            log.debug(f'Expecting all {len(columns)} pyfix '
                      f'{pyfix.__version__} columns')

        # otherwise the file contains column labels
        else:
            header = 0
            log.debug(f'Expecting a subset of pyfix '
                      f'{pyfix.__version__} columns')

    df = pd.read_csv(io.StringIO(data), header=header, names=columns)

    if columns is None:
        columns = df.columns

    if len(columns) != ncols:
        msg = (f'Feature file {path} contains an unexpected '
               f'number of columns! Expected {len(columns)} '
               f'columns, but the file contains {ncols} columns')
        log.error(msg)
        raise RuntimeError(msg)

    # Convert legacy FIX labels into pyfix labels
    df = df.rename(columns=legacy.LEGACY_FEATURE_LABELS)

    for column in df.columns:
        if column not in feature.FEATURE_LABELS:
            log.warning(f'Unrecognised feature column in {path}: {column}')

    for column in feature.FEATURE_LABELS:
        if column not in df.columns:
            log.warning(f'Missing feature column in {path}: {column}')

    return df


def save_features(path     : str,
                  features : pd.DataFrame,
                  header   : bool = True):
    """Save FIX features to file."""
    labels = [f for f in feature.FEATURE_LABELS if f in features.columns]
    features[labels].to_csv(path, index=None, header=header)
