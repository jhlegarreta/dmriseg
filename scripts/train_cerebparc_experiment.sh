#!/bin/bash

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
elif [[ ${contrast} == "ad" ]]; then
  contrast_folder_label=dmri_hcp_ad
elif [[ ${contrast} == "md" ]]; then
  contrast_folder_label=dmri_hcp_md
elif [[ ${contrast} == "rd" ]]; then
  contrast_folder_label=dmri_hcp_rd
elif [[ ${contrast} == "evals_e1" ]]; then
  contrast_folder_label=dmri_hcp_evals_e1
elif [[ ${contrast} == "evals_e2" ]]; then
  contrast_folder_label=dmri_hcp_evals_e2
elif [[ ${contrast} == "evals_e3" ]]; then
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

train_split_label=train_set
valid_split_label=valid_set

data_root_dirname=/mnt/data/cerebellum_parc/experiments

train_scipt_dirname=/home/jhlegarreta/src/dmriseg/scripts

contrast_dirname=${data_root_dirname}/${contrast_folder_label}

for fold in "${folds[@]}"; do

  fold_dirname=${contrast_dirname}/${fold}

  img_train_dirname=${fold_dirname}/${train_split_label}
  img_valid_dirname=${fold_dirname}/${valid_split_label}

  labelmap_train_dirname=${data_root_dirname}/labelmaps/${fold}/${train_split_label}
  labelmap_valid_dirname=${data_root_dirname}/labelmaps/${fold}/${valid_split_label}

  learning_out_dirname=${fold_dirname}/results/learning/segresnet16_batchsz1

  mkdir -p ${learning_out_dirname}

  python ${train_scipt_dirname}/train_cerebparc.py \
    ${img_train_dirname} \
    ${labelmap_train_dirname} \
    ${img_valid_dirname} \
    ${labelmap_valid_dirname} \
    ${learning_out_dirname}

done
