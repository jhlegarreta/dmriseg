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

# Plot only the labelmap 3D volumes
for group_name in "${group_names[@]}"; do

  for view_name in "${view_names[@]}"; do

    if [[ ${contrast} == "labelmap" ]]; then
      out_fname=${out_dirname}/${participant_id}__${file_basename_label}_gnd_th_labelmap_volume_${group_name}_${view_name}.png
    else
      out_fname=${out_dirname}/${participant_id}__${file_basename_label}_pred_labelmap_volume_${group_name}_${view_name}.png
    fi

    python /home/jhlegarreta/src/dmriseg/scripts/screenshot_vtk_labelmap_volume.py \
      ${in_fname} \
      ${lut_fname} \
      ${out_fname} \
      ${view_name} \
      --group_name ${group_name}

  done

done

# Plot the slice and labelmap 3D volumes, and mask to the brain mask
in_processed_data_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/processed_data
if [[ ${contrast} == "labelmap" ]]; then
  bckgnd_contrast_folder_label=dmri_hcp_t1
  bckgnd_contrast_fname_label=t1_resized
  in_fname=${in_processed_data_root_dirname}/${participant_id}/labelmap/${participant_id}__cer_seg_resized_cropped.nii.gz
else
  bckgnd_contrast_folder_label=${in_folder_label}
  bckgnd_contrast_fname_label=${file_basename_label}
  in_fname=${in_processed_data_root_dirname}/${participant_id}/${in_folder_label}/prediction/${participant_id}__${file_basename_label}_pred_cropped.nii.gz
fi

in_ref_anat_img_fname=${in_processed_data_root_dirname}/${participant_id}/${bckgnd_contrast_folder_label}/volume/${participant_id}__${bckgnd_contrast_fname_label}_brainmasked_cropped.nii.gz
mask_fname=${in_processed_data_root_dirname}/${participant_id}/brainmask/${participant_id}__brainmask_resized_cropped.nii.gz

group_names=("dcn")
view_names=("axial_superior")

# Notes
# 117021 cam.SetFocalPoint(center1[0], center1[1], center1[2] + 30) looks OK (for the resizeD_brainmaked_cropped volume)
# 101107 cam.SetFocalPoint(center1[0], center1[1], center1[2] + 34) looks OK (for the resizeD_brainmaked_cropped volume)
# So the center1[2] + {} value was changed manually for each case
# 101107 in fold-1 misses label 34 (Right Fastigial) in t1 prediction; dwi does not; dwi1k does not
# 121719 in fold-1 misses labels 32 (Right_Interposed) and 34 (Right Fastigial) in t1 prediction; dwi does not; dwi1k does not
# 129028 in fold-3 misses label 34 (Right Fastigial) in t1 prediction; dwi does not; dwi1k does not
# All fold-1 and fold-2 cases miss label 34 (Right Fastigial) in t1 prediction
#

for group_name in "${group_names[@]}"; do

  for view_name in "${view_names[@]}"; do

    if [[ ${contrast} == "labelmap" ]]; then
      out_fname=${out_dirname}/${participant_id}__${file_basename_label}_gnd_th_labelmap_volume_${group_name}_${view_name}_slice.png
    else
      out_fname=${out_dirname}/${participant_id}__${file_basename_label}_pred_labelmap_volume_${group_name}_${view_name}_slice.png
    fi

    python /home/jhlegarreta/src/dmriseg/scripts/screenshot_vtk_labelmap_volume.py \
      ${in_fname} \
      ${lut_fname} \
      ${out_fname} \
      ${view_name} \
      --group_name ${group_name} \
      --in_ref_anat_img_fname ${in_ref_anat_img_fname} \
      --mask_fname ${mask_fname}

  done

done
