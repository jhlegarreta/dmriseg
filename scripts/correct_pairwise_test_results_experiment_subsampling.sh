#!/bin/bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

function get_folder_label() {
  local _contrast="$1"

  # Build the labels for the i/o dirs/files
  if [[ ${_contrast} == "dwi" ]]; then
    _folder_label=dwi_subsampling
  elif [[ ${_contrast} == "dwi1k" ]]; then
    _folder_label=dwi1k_subsampling
  else
    echo "Contrast not available:" ${_contrast}
    echo "Aborting."
    exit 0
  fi

  echo ${_folder_label}
}

interest_contrast_name=$1
measure_name=$2

in_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/pairwise_test/subsampling/parametric_true/within_subjects

contrast_dir_label=$(get_folder_label "${interest_contrast_name}")

echo ${contrast_dir_label}

in_dirname=${in_root_dirname}/${contrast_dir_label}
out_dirname=${in_dirname}

in_pairwise_results_fname=${in_dirname}/${measure_name}_pairwise_t_test.tsv
out_fname=${out_dirname}/${measure_name}_pairwise_t_test_pcorr.tsv

echo "Correcting pairwise test p-values..."
python ~/src/dmriseg/scripts/correct_pairwise_test_results.py \
  ${in_pairwise_results_fname} \
  ${out_fname} \
  ${interest_contrast_name}

echo "Finished correcting pairwise test p-values."
