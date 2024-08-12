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

test_split_label=test

data_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
contrast_dirname=${data_root_dirname}/${contrast_folder_label}

folds_root_dirname=/mnt/data/connectome/participant_folds

gnd_th_labelmap_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/labelmaps

in_labels_fname=/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv

perf_script_dirname=/home/jhlegarreta/src/dmriseg/scripts

subsampled_contrast_type=$(subsampled_contrast_type "${contrast}")
echo "Subsampled contrast type (empty if not applicable)": ${subsampled_contrast_type}

echo "Starting performance computation..."
for fold in "${folds[@]}"; do

  echo ${fold}

  fold_dirname=${contrast_dirname}/${fold}

  if [[ ${subsampled_contrast_type} == "${subdwi_contrast}" || ${subsampled_contrast_type} == "${subdwi1k_contrast}" ]]; then
    in_participants_fname=${folds_root_dirname}/${fold}/cerebellum_participants_hcp_qc_no_sub_prefix_dwi_subsampling_${fold}_split-${test_split_label}.tsv
  else
    in_participants_fname=${folds_root_dirname}/${fold}/cerebellum_participants_hcp_qc_no_sub_prefix_${fold}_split-${test_split_label}.tsv
  fi

  echo "Participants fname:" ${in_participants_fname}

  in_gnd_th_labelmap_dirname=${gnd_th_labelmap_root_dirname}/${fold}/test_set
  in_pred_labelmap_dirname=${contrast_dirname}/${fold}/results/prediction

  perf_dirname=${fold_dirname}/results/performance

  mkdir ${perf_dirname}


  if [[ ${subsampled_contrast_type} == "${subdwi_contrast}" || ${subsampled_contrast_type} == "${subdwi1k_contrast}" ]]; then
    python ${perf_script_dirname}/compute_performance.py \
      ${in_participants_fname} \
      ${in_gnd_th_labelmap_dirname} \
      ${in_pred_labelmap_dirname} \
      ${in_labels_fname} \
      ${perf_dirname} \
      --subsampled_dwi
  else
    python ${perf_script_dirname}/compute_performance.py \
      ${in_participants_fname} \
      ${in_gnd_th_labelmap_dirname} \
      ${in_pred_labelmap_dirname} \
      ${in_labels_fname} \
      ${perf_dirname}
  fi

done

echo "Finished"
