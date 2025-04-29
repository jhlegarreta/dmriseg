#!/bin/bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

function get_folder_label() {
  local _contrast="$1"

  # Build the labels for the i/o dirs/files
  if [[ ${_contrast} == "dwi" ]]; then
    _folder_label=dmri_hcp_sphm_b1000-2000-3000
  elif [[ ${_contrast} == "dwi1k" ]]; then
    _folder_label=dmri_hcp_sphm_b1000
  elif [[ ${_contrast} == "dwisub20" ]]; then
    _folder_label=dmri_hcp_sphm_b1000-2000-3000_subsampled_dirs20
  elif [[ ${_contrast} == "dwisub30" ]]; then
    _folder_label=dmri_hcp_sphm_b1000-2000-3000_subsampled_dirs30
  elif [[ ${_contrast} == "dwisub60" ]]; then
    _folder_label=dmri_hcp_sphm_b1000-2000-3000_subsampled_dirs60
  elif [[ ${_contrast} == "dwi1ksub20" ]]; then
    _folder_label=dmri_hcp_sphm_b1000_subsampled_dirs20
  elif [[ ${_contrast} == "dwi1ksub30" ]]; then
    _folder_label=dmri_hcp_sphm_b1000_subsampled_dirs30
  elif [[ ${_contrast} == "dwi1ksub60" ]]; then
    _folder_label=dmri_hcp_sphm_b1000_subsampled_dirs60
  else
    echo "Contrast not available:" ${_contrast}
    echo "Aborting."
    exit 0
  fi

  echo ${_folder_label}
}

measure_names=("dice" "hd95" "msd" "cm_dist" "vs")

in_subsampled_dwi_contrasts=("dwi" "dwisub20" "dwisub30" "dwisub60")
in_subsampled_dwi1k_contrasts=("dwi1k" "dwi1ksub20" "dwi1ksub30" "dwi1ksub60")

in_contrast_groups=("${in_subsampled_dwi_contrasts[*]}" "${in_subsampled_dwi1k_contrasts[*]}")

lut_fname=/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv

in_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
out_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/pairwise_test/subsampling/parametric_true/within_subjects

echo "Computing pairwise test..."

for _contrast_group in "${in_contrast_groups[@]}"; do

  echo ${_contrast_group}

  for _measure_name in "${measure_names[@]}"; do

    echo ${_measure_name}

    in_performance_fnames=()

    # Build array of filenames
    for _contrast_name in ${_contrast_group}; do

      folder_label=$(get_folder_label "$_contrast_name")

      in_performance_fnames+=(${in_root_dirname}/${folder_label}/aggregate_performance/${_measure_name}.tsv)

    done

    IFS=' ' read -r -a _contrast_group_items <<< "${_contrast_group}"

    echo "Contrasts (""${#_contrast_group_items[@]}""):" "${_contrast_group_items[@]}"
    echo "Filenames (""${#in_performance_fnames[@]}""):"
    printf "%s\n" "${in_performance_fnames[@]}"

    _out_subsampling_dirname=${out_dirname}/${_contrast_group_items[0]}_subsampling

    # Perform the statistical test
    python /home/jhlegarreta/src/dmriseg/scripts/compute_pairwise_test.py \
      ${lut_fname} \
      ${_measure_name} \
      ${_out_subsampling_dirname} \
      --in_performance_fnames "${in_performance_fnames[@]}" \
      --in_contrast_names "${_contrast_group_items[@]}"

  done

done

echo "Finished computing pairwise test."
