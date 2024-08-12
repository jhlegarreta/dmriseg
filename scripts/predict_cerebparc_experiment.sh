#!/usr/bin/env bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

subdwi_contrast="dwisub"
subdwi1k_contrast="dwi1ksub"

subsampled_dwi_contrasts=("dwisub20" "dwisub30" "dwisub60")
subsampled_dwi1k_contrasts=("dwi1ksub20" "dwi1ksub30" "dwi1ksub60")

function subsampled_contrast_type() {
  local _contrast="$1"

  _subsampled_contrast_type=""

  # Check if the string equals any element in the array
  for element in "${subsampled_dwi_contrasts[@]}"; do
    if [[ "${_contrast}" == "$element" ]]; then
      # echo "Match found: $element"
      _subsampled_contrast_type=${subdwi_contrast}
      break
    fi
  done

  for element in "${subsampled_dwi1k_contrasts[@]}"; do
    if [[ "${_contrast}" == "$element" ]]; then
      # echo "Match found: $element"
      _subsampled_contrast_type=${subdwi1k_contrast}
      break
    fi
  done

  echo ${_subsampled_contrast_type}
}

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

echo $contrast_folder_label

test_split_label=test_set

data_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
contrast_dirname=${data_root_dirname}/${contrast_folder_label}

predict_script_dirname=/home/jhlegarreta/src/dmriseg/scripts

subsampled_contrast_type=$(subsampled_contrast_type "${contrast}")
echo "Subsampled contrast type (empty if not applicable)": ${subsampled_contrast_type}

echo "Starting prediction..."
for fold in "${folds[@]}"; do

  echo ${fold}

  fold_dirname=${contrast_dirname}/${fold}
  img_test_dirname=${fold_dirname}/${test_split_label}

  echo "Fold dirname:" ${fold_dirname}
  echo "Test image dirname:" ${img_test_dirname}

  # For the subsampled cases we did not train a model, and are using the one
  # trained on all directions
  if [[ ${subsampled_contrast_type} == "${subdwi_contrast}" ]]; then
    weights_dirname=${data_root_dirname}/dmri_hcp_sphm_b1000-2000-3000/${fold}/results/learning/segresnet16_batchsz1
  elif [[ ${subsampled_contrast_type} == "${subdwi1k_contrast}" ]]; then
    weights_dirname=${data_root_dirname}/dmri_hcp_sphm_b1000/${fold}/results/learning/segresnet16_batchsz1
  elif [[ ${subsampled_contrast_type} == "" ]]; then
    weights_dirname=${fold_dirname}/results/learning/segresnet16_batchsz1
  else
    echo "Contrast type for weights could not be identified:" ${contrast}
    echo "Aborting."
    exit 0
  fi

  echo "Weights dirname:" ${weights_dirname}

  pred_dirname=${fold_dirname}/results/prediction

  mkdir -p ${pred_dirname}

  python ${predict_script_dirname}/predict_cerebparc.py \
    ${img_test_dirname} \
    ${weights_dirname}/model_best.pth \
    ${pred_dirname}

done

echo "Finished"
