#!/bin/bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

function get_folder_label() {
  local _contrast="$1"

  # Build the labels for the i/o dirs/files
  if [[ ${_contrast} == "t1" ]]; then
    _folder_label=dmri_hcp_t1
  elif [[ ${_contrast} == "t2" ]]; then
    _folder_label=dmri_hcp_t2
  elif [[ ${_contrast} == "b0" ]]; then
    _folder_label=dmri_hcp_b0
  elif [[ ${_contrast} == "dwi" ]]; then
    _folder_label=dmri_hcp_sphm_b1000-2000-3000
  elif [[ ${_contrast} == "dwi1k" ]]; then
    _folder_label=dmri_hcp_sphm_b1000
  elif [[ ${_contrast} == "dwi2k" ]]; then
    _folder_label=dmri_hcp_sphm_b2000
  elif [[ ${_contrast} == "dwi3k" ]]; then
    _folder_label=dmri_hcp_sphm_b3000
  elif [[ ${_contrast} == "fa" ]]; then
    _folder_label=dmri_hcp_fa
  elif [[ ${_contrast} == "md" ]]; then
    _folder_label=dmri_hcp_md
  elif [[ ${_contrast} == "rd" ]]; then
    _folder_label=dmri_hcp_rd
  elif [[ ${_contrast} == "evalse1" ]]; then
    _folder_label=dmri_hcp_evals_e1
  elif [[ ${_contrast} == "evalse2" ]]; then
    _folder_label=dmri_hcp_evals_e2
  elif [[ ${_contrast} == "evalse3" ]]; then
    _folder_label=dmri_hcp_evals_e3
  elif [[ ${_contrast} == "ak" ]]; then
    _folder_label=dmri_hcp_ak
  elif [[ ${_contrast} == "mk" ]]; then
    _folder_label=dmri_hcp_mk
  elif [[ ${_contrast} == "rk" ]]; then
    _folder_label=dmri_hcp_rk
  else
    echo "Contrast not available:" ${_contrast}
    echo "Aborting."
    exit 0
  fi

  echo ${_folder_label}
}

measure_names=("dice" "hd95" "msd" "cm_dist" "vs")

in_contrast_names=("t1" "b0" "dwi" "dwi1k" "dwi2k" "dwi3k" "fa" "md" "rd" "evalse1" "evalse2" "evalse3" "ak" "mk" "rk")

in_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
out_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/anova/

echo "Plotting measure distribution..."
for _measure_name in "${measure_names[@]}"; do

  echo ${_measure_name}

  in_performance_dirnames=()

  # Build array of dirnames
  for _contrast_name in "${in_contrast_names[@]}"; do

    folder_label=$(get_folder_label "$_contrast_name")

    in_performance_dirnames+=(${in_root_dirname}/${folder_label}/aggregate_performance)

  done

  echo "Contrasts (""${#in_contrast_names[@]}""):" "${in_contrast_names[@]}"
  echo "Dirnames (""${#in_performance_dirnames[@]}""):"
  printf "%s\n" "${in_performance_dirnames[@]}"

  python ~/src/dmriseg/scripts/plot_measure_distribution.py \
    --in_performance_dirnames "${in_performance_dirnames[@]}" \
    --contrast_names "${in_contrast_names[@]}" \
    --measure_name ${_measure_name} \
    --out_fname ${out_dirname}/${_measure_name}_distribution.svg
    # --out_fname ${out_dirname}/${_measure_name}_distribution_kde.svg --kde

done

echo "Finished plotting measure distribution."
