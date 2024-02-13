#!/usr/bin/env bash

envdir=${1}

channels=(-c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/development/
          -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/
          -c conda-forge)

conda create -y ${channels[@]}   \
      -p ./${envdir}             \
      cachetools                 \
      coverage                   \
      file-tree                  \
      file-tree-fsl              \
      fsl-avwutils               \
      fsl-cluster                \
      fsl-fast4                  \
      fsl-flirt                  \
      fsl-data_standard          \
      jinja2                     \
      joblib                     \
      nibabel                    \
      numpy                      \
      pandas                     \
      pytest                     \
      pytest-cov                 \
      python=3.11                \
      r-base                     \
      scikit-learn               \
      scipy                      \
      spectrum

echo "channels:"                                                        > ${envdir}/.condarc
echo "  https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/development/" >> ${envdir}/.condarc
echo "  https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/"      >> ${envdir}/.condarc
echo "  conda-forge"                                                   >> ${envdir}/.condarc

conda clean -y -all
pip cache purge
