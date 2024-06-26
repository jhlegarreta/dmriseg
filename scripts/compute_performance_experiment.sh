#!/usr/bin/env bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

contrast=$1

# Folds
folds=("fold-0" "fold-1" "fold-2" "fold-3" "fold-4")

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
else
  echo "Contrast not available:" ${contrast}
  echo "Aborting."
  exit 0
fi

test_split_label=test

data_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
contrast_dirname=${data_root_dirname}/${contrast_folder_label}

folds_root_dirname=/mnt/data/connectome/participant_folds

gnd_th_labelmap_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/labelmaps

in_labels_fname=/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv

perf_script_dirname=/home/jhlegarreta/src/dmriseg/scripts

echo "Starting performance computation..."
for fold in "${folds[@]}"; do

  echo ${fold}

  fold_dirname=${contrast_dirname}/${fold}

  in_participants_fname=${folds_root_dirname}/${fold}/cerebellum_participants_hcp_qc_no_sub_prefix_${fold}_split-${test_split_label}.tsv
  in_gnd_th_labelmap_dirname=${gnd_th_labelmap_root_dirname}/${fold}/test_set
  in_pred_labelmap_dirname=${contrast_dirname}/${fold}/results/prediction

  perf_dirname=${fold_dirname}/results/performance

  mkdir ${perf_dirname}

  python ${perf_script_dirname}/compute_performance.py \
    ${in_participants_fname} \
    ${in_gnd_th_labelmap_dirname} \
    ${in_pred_labelmap_dirname} \
    ${in_labels_fname} \
    ${perf_dirname}

done

echo "Finished"
