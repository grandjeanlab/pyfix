#!/usr/bin/env bash

source activate /dev.env
pip install -e ".[extra]"

export FSLDIR=${CONDA_PREFIX}
source $FSLDIR/etc/fslconf/fsl.sh

pytest
