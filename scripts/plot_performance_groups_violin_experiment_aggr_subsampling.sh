#!/usr/bin/env bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

ref_contrast=$1

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
  if [[ ${_contrast} == "dwi" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000-2000-3000
  elif [[ ${_contrast} == "dwi1k" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000
  elif [[ ${_contrast} == "dwisub20" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000-2000-3000_subsampled_dirs20
  elif [[ ${_contrast} == "dwisub30" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000-2000-3000_subsampled_dirs30
  elif [[ ${_contrast} == "dwisub60" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000-2000-3000_subsampled_dirs60
  elif [[ ${_contrast} == "dwi1ksub20" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000_subsampled_dirs20
  elif [[ ${_contrast} == "dwi1ksub30" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000_subsampled_dirs30
  elif [[ ${_contrast} == "dwi1ksub60" ]]; then
    in_folder_label=dmri_hcp_sphm_b1000_subsampled_dirs60
  else
    echo "Contrast not available:" ${contrast}
    echo "Aborting."
    exit 0
  fi

  echo ${in_folder_label}
}

measure_names=("dice" "hd95" "msd" "cm_dist" "vs" "lab_detect_rate")

lut_fname=/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv
in_performance_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
out_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/figures/performance_plots_aggr/subsampling

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

for _measure_name in "${measure_names[@]}"; do

  echo "Measure name:" ${_measure_name}

  # For now, the significance, description and measurement files are not used
  python /home/jhlegarreta/src/dmriseg/scripts/plot_performance_groups_violin.py \
    --in_performance_dirnames \
    ${in_performance_dirname}/${ref_contrast_folder_label}/aggregate_performance \
    "${contrast_performance_dirnames[@]}" \
    --out_dirname ${out_dirname} \
    --measure_name ${_measure_name} \
    --in_labels_fname ${lut_fname} \
    --aggregate_only

done
