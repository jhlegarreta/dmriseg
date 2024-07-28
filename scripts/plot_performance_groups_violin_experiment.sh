#!/usr/bin/env bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

measure=$1
contrast1=$2
contrast2=$3

lut_fname=/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv
in_performance_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
out_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/figures/performance_plots/

ref_contrast=t1

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
contrast1_folder_label=$(get_io_labels_from_contrast "${contrast1}")
contrast2_folder_label=$(get_io_labels_from_contrast "${contrast2}")

echo "Contrast folder labels:"
echo "Reference:" ${ref_contrast_folder_label}
echo "Contrast1:" ${contrast1_folder_label}
echo "Contrast2:" ${contrast2_folder_label}

# For now, the significance, description and measurement files are not used
python /home/jhlegarreta/src/dmriseg/scripts/plot_performance_groups_violin.py \
  --in_performance_dirnames \
  ${in_performance_dirname}/${ref_contrast_folder_label}/aggregate_performance \
  ${in_performance_dirname}/${contrast1_folder_label}/aggregate_performance \
  ${in_performance_dirname}/${contrast2_folder_label}/aggregate_performance \
  --out_dirname ${out_dirname} \
  --measure_name ${measure} \
  --in_labels_fname ${lut_fname} \
  --in_significance_fnames \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc/wilcoxon_ranksum_dice_sphm_bs_vs_b0.tsv \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc/wilcoxon_ranksum_dice_sphm_bs_vs_b0.tsv \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc/wilcoxon_ranksum_dice_sphm_bs_vs_b0.tsv \
  --in_description_fnames \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc/wilcoxon_ranksum_dice_t1_dwi_description.tsv \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc/wilcoxon_ranksum_dice_t1_dwi1k_description.tsv \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc/wilcoxon_ranksum_dice_dwi_dwi1k_description.tsv \
  --in_measurement_fnames \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc/wilcoxon_ranksum_dice_t1_dwi_measurements.tsv \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc/wilcoxon_ranksum_dice_t1_dwi1k_measurements.tsv \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc/wilcoxon_ranksum_dice_dwi_dwi1k_measurements.tsv
