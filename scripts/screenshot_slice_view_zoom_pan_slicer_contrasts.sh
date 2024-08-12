#!/bin/bash

contrast=$1

slicer_path=/opt/Slicer-5.2.2-linux-amd64/Slicer

in_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/processed_data
out_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/figures

#view_name="axial"
#offset=-25.7  # Adjust as necessary
view_name="coronal"
offset=-52.2  #-51.2
#view_name="sagittal"
#offset=-6.2

# These values have only been tested with coronal slices
zoom_factor=2.3
translation_value=-30

# Get the best participant name
_sub_id=117021  # Adjust as necessary
# fold="fold-2"  # Not needed

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
else
  echo "Contrast not available:" ${contrast}
  echo "Aborting."
  exit 0
fi

# Create the output folder if it does not exist
mkdir ${out_root_dirname}/${out_folder_label}

# Screenshot the volume
in_vol_fname=${in_root_dirname}/${_sub_id}/${in_folder_label}/volume/${_sub_id}__${file_basename_label}_brainmasked_cropped.nii.gz
in_mask_fname=${in_root_dirname}/${_sub_id}/brainmask/${_sub_id}__brainmask_resized_cropped.nii.gz

out_fname=${out_root_dirname}/${out_folder_label}/${_sub_id}__${file_basename_label}_${view_name}_cerebellum.png

${slicer_path} --no-splash --python-script \
  /home/jhlegarreta/src/dmriseg/scripts/screenshot_slice_view_zoom_pan_slicer.py \
  ${in_vol_fname} \
  ${out_fname} \
  ${view_name} \
  ${offset} \
  ${zoom_factor} \
  ${translation_value} \
  --in_mask_filename ${in_mask_fname}
