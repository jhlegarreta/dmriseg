#!/usr/bin/env bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

measure=$1

lut_fname=/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv
in_performance_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
out_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/figures/performance_plots/ismrm

ref_contrast=t1

# Initialize a flag to track if "-c" was found
contrasts_found=false
contrasts=()

# Loop through all arguments
for arg in "$@"; do
  if [ "$arg" == "-c" ]; then
    contrasts_found=true
  elif ${contrasts_found}; then
    # Add arguments after "-c" to the array
    contrasts+=("$arg")
  fi
done

if ! ${contrasts_found}; then
  echo "Error: No contrast found. Specify the contrast names after the -c flag."
  exit 1
fi

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
    echo "Contrast not available:" ${contrast}
    echo "Aborting."
    exit 0
  fi

  echo ${in_folder_label}
}

ref_contrast_folder_label=$(get_io_labels_from_contrast "${ref_contrast}")

contrast_folder_labels=()
for _contrast in "${contrasts[@]}"; do
  _folder_label=$(get_io_labels_from_contrast "${_contrast}")
  contrast_folder_labels+=("${_folder_label}")
done

echo "Contrast folder labels:"
echo "Reference:" ${ref_contrast_folder_label}
echo "Contrasts:" "${contrast_folder_labels[@]}"

contrast_performance_dirnames=()
for _folder_label in "${contrast_folder_labels[@]}"; do
  contrast_performance_dirnames+=(${in_performance_dirname}/${_folder_label}/aggregate_performance)
done

# For now, the significance, description and measurement files are not used
python /home/jhlegarreta/src/dmriseg/scripts/plot_performance_groups_violin.py \
  --in_performance_dirnames \
  ${in_performance_dirname}/${ref_contrast_folder_label}/aggregate_performance \
  "${contrast_performance_dirnames[@]}" \
  --out_dirname ${out_dirname} \
  --measure_name ${measure} \
  --in_labels_fname ${lut_fname}
