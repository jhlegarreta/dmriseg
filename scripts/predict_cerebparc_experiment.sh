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

test_split_label=test_set

data_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
contrast_dirname=${data_root_dirname}/${contrast_folder_label}

predict_script_dirname=/home/jhlegarreta/src/dmriseg/scripts

echo "Starting prediction..."
for fold in "${folds[@]}"; do

  echo ${fold}

  fold_dirname=${contrast_dirname}/${fold}
  img_test_dirname=${fold_dirname}/${test_split_label}
  weights_dirname=${fold_dirname}/results/learning/segresnet16_batchsz1
  pred_dirname=${fold_dirname}/results/prediction

  mkdir ${pred_dirname}

  python ${predict_script_dirname}/predict_cerebparc.py \
    ${img_test_dirname} \
    ${weights_dirname}/model_best.pth \
    ${pred_dirname}

done

echo "Finished"
