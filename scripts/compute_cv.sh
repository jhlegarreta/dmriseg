#!/usr/bin/env bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

contrast=$1

function get_io_labels_from_contrast() {
  local _contrast="$1"

  # Build the labels for the i/o dirs/files
  if [[ ${_contrast} == "t1" ]]; then
    in_folder_label=dmri_hcp_t1
    file_basename_label=t1_resized
    out_folder_label=dmri_hcp_t1
  elif [[ ${_contrast} == "t2" ]]; then
    in_folder_label=dmri_hcp_t2
    file_basename_label=t2_resized
    out_folder_label=dmri_hcp_t2
  elif [[ ${_contrast} == "b0" ]]; then
    in_folder_label=dmri_hcp_b0
    file_basename_label=dwi_spherical_mean-b0_resized
    out_folder_label=dmri_hcp_b0
  elif [[ ${_contrast} == "dwi" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000-2000-3000
    file_basename_label=dwi_spherical_mean-b1000-2000-3000_resized
    out_folder_label=dmri_hcp_sphm_b1000-2000-3000
  elif [[ ${_contrast} == "dwi1k" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000
    file_basename_label=dwi_spherical_mean-b1000_resized
    out_folder_label=dmri_hcp_sphm_b1000
  elif [[ ${_contrast} == "dwi2k" ]]; then
    in_folder_label=dmri_hcp_sphm_b2000
    file_basename_label=dwi_spherical_mean-b2000_resized
    out_folder_label=dmri_hcp_sphm_b2000
  elif [[ ${_contrast} == "dwi3k" ]]; then
    in_folder_label=dmri_hcp_sphm_b3000
    file_basename_label=dwi_spherical_mean-b3000_resized
    out_folder_label=dmri_hcp_sphm_b3000
  elif [[ ${_contrast} == "fa" ]]; then
    in_folder_label=dmri_hcp_fa
    file_basename_label=fa_resized
    out_folder_label=dmri_hcp_fa
  elif [[ ${_contrast} == "md" ]]; then
    in_folder_label=dmri_hcp_md
    file_basename_label=md_resized
    out_folder_label=dmri_hcp_md
  elif [[ ${_contrast} == "rd" ]]; then
    in_folder_label=dmri_hcp_rd
    file_basename_label=rd_resized
    out_folder_label=dmri_hcp_rd
  elif [[ ${_contrast} == "evalse1" ]]; then
    in_folder_label=dmri_hcp_evals_e1
    file_basename_label=evals_e1_resized
    out_folder_label=dmri_hcp_evals_e1
  elif [[ ${_contrast} == "evalse2" ]]; then
    in_folder_label=dmri_hcp_evals_e2
    file_basename_label=evals_e2_resized
    out_folder_label=dmri_hcp_evals_e2
  elif [[ ${_contrast} == "evalse3" ]]; then
    in_folder_label=dmri_hcp_evals_e3
    file_basename_label=evals_e3_resized
    out_folder_label=dmri_hcp_evals_e3
  elif [[ ${_contrast} == "ak" ]]; then
    in_folder_label=dmri_hcp_ak
    file_basename_label=ak_resized
    out_folder_label=dmri_hcp_ak
  elif [[ ${_contrast} == "mk" ]]; then
    in_folder_label=dmri_hcp_mk
    file_basename_label=mk_resized
    out_folder_label=dmri_hcp_mk
  elif [[ ${_contrast} == "rk" ]]; then
    in_folder_label=dmri_hcp_rk
    file_basename_label=rk_resized
    out_folder_label=dmri_hcp_rk
  else
    echo "Contrast not available:" ${contrast} >&2
    exit 1
  fi

  echo ${in_folder_label} ${file_basename_label} ${out_folder_label}

}


in_participants_fname="/mnt/data/connectome/cerebellum_participants_hcp_qc_no_sub_prefix.tsv"
in_participants_fold_dirname="/mnt/data/connectome/participant_folds/"
in_dwi_scalarmap_dirname="/mnt/data/connectome/preprocess_dwi_minimal/"
in_pred_labelmap_root_dirname="/mnt/data/cerebellum_parc/experiments_minimal_pipeline/"
in_labels_fname="/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv"
out_root_dirname="/mnt/data/cerebellum_parc/experiments_minimal_pipeline/coefficient_variation"

result=$(get_io_labels_from_contrast "${contrast}")
status=$?

if [ $status -ne 0 ]; then
  echo "Aborting."
  exit 1
fi

read -r -a result_array <<< "$result"

contrast_folder_label=${result_array[0]}
pred_fname_label=${result_array[1]}
out_folder_label=${result_array[2]}

dti_metrics=("fa" "md")
out_dirname=${out_root_dirname}/${out_folder_label}

mkdir ${out_dirname}

echo "Computing coefficient of variation..."

for _metric in "${dti_metrics[@]}"; do

  echo "Metric:" ${_metric}

  python /home/jhlegarreta/src/dmriseg/scripts/compute_cv.py \
    ${in_participants_fname} \
    ${in_participants_fold_dirname} \
    ${in_dwi_scalarmap_dirname} \
    ${in_pred_labelmap_root_dirname} \
    ${contrast_folder_label} \
    ${pred_fname_label} \
    ${_metric} \
    ${in_labels_fname} \
    ${out_dirname}

done

echo "Finished computing coefficient of variation."
