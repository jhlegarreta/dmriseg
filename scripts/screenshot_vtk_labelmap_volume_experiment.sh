#!/bin/bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

participant_id=$1
contrast=$2
fold=$3

# Build the labels for the i/o dirs/files
if [[ ${contrast} == "t1" ]]; then
  in_folder_label=dmri_hcp_t1
  file_basename_label=t1_resized
  out_folder_label=dmri_hcp_t1
elif [[ ${contrast} == "t2" ]]; then
  in_folder_label=dmri_hcp_t2
  file_basename_label=t2_resized
  out_folder_label=dmri_hcp_t2
elif [[ ${contrast} == "b0" ]]; then
  in_folder_label=dmri_hcp_b0
  file_basename_label=dwi_spherical_mean-b0_resized
  out_folder_label=dmri_hcp_b0
elif [[ ${contrast} == "dwi" ]]; then
  in_folder_label=dmri_hcp_sphm_b1000-2000-3000
  file_basename_label=dwi_spherical_mean-b1000-2000-3000_resized
  out_folder_label=dmri_hcp_sphm_b1000-2000-3000
elif [[ ${contrast} == "dwi1k" ]]; then
  in_folder_label=dmri_hcp_sphm_b1000
  file_basename_label=dwi_spherical_mean-b1000_resized
  out_folder_label=dmri_hcp_sphm_b1000
elif [[ ${contrast} == "dwi2k" ]]; then
  in_folder_label=dmri_hcp_sphm_b2000
  file_basename_label=dwi_spherical_mean-b2000_resized
  out_folder_label=dmri_hcp_sphm_b2000
elif [[ ${contrast} == "dwi3k" ]]; then
  in_folder_label=dmri_hcp_sphm_b3000
  file_basename_label=dwi_spherical_mean-b3000_resized
  out_folder_label=dmri_hcp_sphm_b3000
elif [[ ${contrast} == "fa" ]]; then
  in_folder_label=dmri_hcp_fa
  file_basename_label=fa_resized
  out_folder_label=dmri_hcp_fa
elif [[ ${contrast} == "md" ]]; then
  in_folder_label=dmri_hcp_md
  file_basename_label=md_resized
  out_folder_label=dmri_hcp_md
elif [[ ${contrast} == "rd" ]]; then
  in_folder_label=dmri_hcp_rd
  file_basename_label=rd_resized
  out_folder_label=dmri_hcp_rd
elif [[ ${contrast} == "evalse1" ]]; then
  in_folder_label=dmri_hcp_evals_e1
  file_basename_label=evals_e1_resized
  out_folder_label=dmri_hcp_evals_e1
elif [[ ${contrast} == "evalse2" ]]; then
  in_folder_label=dmri_hcp_evals_e2
  file_basename_label=evals_e2_resized
  out_folder_label=dmri_hcp_evals_e2
elif [[ ${contrast} == "evalse3" ]]; then
  in_folder_label=dmri_hcp_evals_e3
  file_basename_label=evals_e3_resized
  out_folder_label=dmri_hcp_evals_e3
elif [[ ${contrast} == "ak" ]]; then
  in_folder_label=dmri_hcp_ak
  file_basename_label=ak_resized
  out_folder_label=dmri_hcp_ak
elif [[ ${contrast} == "mk" ]]; then
  in_folder_label=dmri_hcp_mk
  file_basename_label=mk_resized
  out_folder_label=dmri_hcp_mk
elif [[ ${contrast} == "rk" ]]; then
  in_folder_label=dmri_hcp_rk
  file_basename_label=rk_resized
  out_folder_label=dmri_hcp_rk
elif [[ ${contrast} == "labelmap" ]]; then
  in_folder_label=labelmaps
  file_basename_label=cer_seg_resized
  out_folder_label=labelmaps
else
  echo "Contrast not available:" ${contrast}
  echo "Aborting."
  exit 0
fi

lut_fname=/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv

in_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline
out_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/figures

in_dirname=${in_root_dirname}/${in_folder_label}/${fold}
out_dirname=${out_root_dirname}/${out_folder_label}

if [[ ${contrast} == "labelmap" ]]; then
  in_fname=${in_dirname}/test_set/${participant_id}__${file_basename_label}.nii.gz
else
  in_fname=${in_dirname}/results/prediction/${participant_id}__${file_basename_label}_pred.nii.gz
fi

group_names=("all" "dcn")
view_names=("axial_superior" "axial_inferior" "coronal_anterior" "coronal_posterior" "sagittal_left" "sagittal_right")

for group_name in "${group_names[@]}"; do

  for view_name in "${view_names[@]}"; do

    out_fname=${out_dirname}/${participant_id}__${file_basename_label}_pred_labelmap_volume_${group_name}_${view_name}.png

    python /home/jhlegarreta/src/dmriseg/scripts/screenshot_vtk_labelmap_volume.py \
      ${in_fname} \
      ${lut_fname} \
      ${out_fname} \
      ${view_name} \
      --group_name ${group_name}

  done

done
