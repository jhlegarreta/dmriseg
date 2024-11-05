#!/bin/bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

interest_contrast_name=$1
measure_name=$2

in_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/pairwise_test/ismrm/parametric_true/within_subjects/
out_dirname=${in_dirname}

in_pairwise_results_fname=${in_dirname}/${measure_name}_pairwise_t_test.tsv
out_fname=${out_dirname}/${measure_name}_pairwise_t_test_pcorr.tsv

echo "Correcting pairwise test p-values..."
python ~/src/dmriseg/scripts/correct_pairwise_test_results.py \
  ${in_pairwise_results_fname} \
  ${out_fname} \
  ${interest_contrast_name}

echo "Finished correcting pairwise test p-values."
