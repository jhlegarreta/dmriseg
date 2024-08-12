#!/bin/bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

contrast=$1

# Build the labels for the i/o dirs/files
if [[ ${contrast} == "t1" ]]; then
  contrast_folder_label=dmri_hcp_t1
elif [[ ${contrast} == "t2" ]]; then
  contrast_folder_label=dmri_hcp_t2
elif [[ ${contrast} == "b0" ]]; then
  contrast_folder_label=dmri_hcp_b0
elif [[ ${contrast} == "dwi" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b1000-2000-3000
elif [[ ${contrast} == "dwi1k" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b1000
elif [[ ${contrast} == "dwi2k" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b2000
elif [[ ${contrast} == "dwi3k" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b3000
elif [[ ${contrast} == "fa" ]]; then
  contrast_folder_label=dmri_hcp_fa
elif [[ ${contrast} == "md" ]]; then
  contrast_folder_label=dmri_hcp_md
elif [[ ${contrast} == "rd" ]]; then
  contrast_folder_label=dmri_hcp_rd
elif [[ ${contrast} == "evalse1" ]]; then
  contrast_folder_label=dmri_hcp_evals_e1
elif [[ ${contrast} == "evalse2" ]]; then
  contrast_folder_label=dmri_hcp_evals_e2
elif [[ ${contrast} == "evalse3" ]]; then
  contrast_folder_label=dmri_hcp_evals_e3
elif [[ ${contrast} == "ak" ]]; then
  contrast_folder_label=dmri_hcp_ak
elif [[ ${contrast} == "mk" ]]; then
  contrast_folder_label=dmri_hcp_mk
elif [[ ${contrast} == "rk" ]]; then
  contrast_folder_label=dmri_hcp_rk
elif [[ ${contrast} == "dwisub20" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b1000-2000-3000_subsampled_dirs20
elif [[ ${contrast} == "dwisub30" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b1000-2000-3000_subsampled_dirs30
elif [[ ${contrast} == "dwisub60" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b1000-2000-3000_subsampled_dirs60
elif [[ ${contrast} == "dwi1ksub20" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b1000_subsampled_dirs20
elif [[ ${contrast} == "dwi1ksub30" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b1000_subsampled_dirs30
elif [[ ${contrast} == "dwi1ksub60" ]]; then
  contrast_folder_label=dmri_hcp_sphm_b1000_subsampled_dirs60
else
  echo "Contrast not available:" ${contrast}
  echo "Aborting."
  exit 0
fi

in_fold_performance_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline

out_dirname=${in_fold_performance_root_dirname}/${contrast_folder_label}/aggregate_performance
mkdir ${out_dirname}

perf_script_dirname=/home/jhlegarreta/src/dmriseg/scripts

echo "Starting performance data aggregation..."

python ${perf_script_dirname}/aggregate_performance_folds.py \
  --in_fold_performance_dirnames \
  ${in_fold_performance_root_dirname}/${contrast_folder_label}/fold-0/results/performance \
  ${in_fold_performance_root_dirname}/${contrast_folder_label}/fold-1/results/performance \
  ${in_fold_performance_root_dirname}/${contrast_folder_label}/fold-2/results/performance \
  ${in_fold_performance_root_dirname}/${contrast_folder_label}/fold-3/results/performance \
  ${in_fold_performance_root_dirname}/${contrast_folder_label}/fold-4/results/performance \
  --fold_names \
  fold-0 \
  fold-1 \
  fold-2 \
  fold-3 \
  fold-4 \
  --out_dirname \
  ${out_dirname}

echo "Finished."
