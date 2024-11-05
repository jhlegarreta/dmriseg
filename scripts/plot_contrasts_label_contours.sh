#!/bin/bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)
workon dmriseg

participant_id=117021
#participant_id=129129
learning_split="test_set"
fold="fold-2"
#fold="fold-0"
# Dentate
labels=(29 30)
label_name="dentate"
origin=(96 69 55)

echo "Participant:" ${participant_id}
echo "Labels:" ${labels[@]}
echo "Label name:" ${label_name}

gnd_th_labelmap_label="Silver standard"
t1w_labelmap_label="T1w prediction"
dwi_labelmap_label="SM prediction"
dwi1k_labelmap_label="SM1k prediction"

in_ref_anat_img_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/dmri_hcp_t1
in_ref_anat_img_fname=${in_ref_anat_img_root_dirname}/${fold}/${learning_split}/${participant_id}__t1_resized.nii.gz

in_gnd_th_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/labelmaps
in_gnd_th_fname=${in_gnd_th_root_dirname}/${fold}/${learning_split}/${participant_id}__cer_seg_resized.nii.gz

inf_pred_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline

out_root_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/figures

# T1
in_folder_label=dmri_hcp_t1
file_basename_label=t1_resized
in_pred_root_dirname=${inf_pred_root_dirname}/${in_folder_label}/${fold}/results/prediction/
in_pred_fname=${in_pred_root_dirname}/${participant_id}__${file_basename_label}_pred.nii.gz
in_labelmap_fnames=("${in_gnd_th_fname}" "${in_pred_fname}")
in_labelmap_names=("${gnd_th_labelmap_label}" "${t1w_labelmap_label}")

out_folder_label=dmri_hcp_t1
out_fname=${out_root_dirname}/${out_folder_label}/${participant_id}__${file_basename_label}_${label_name}_contour.png

echo "Contrast:" ${in_folder_label}

python ~/src/dmriseg/scripts/plot_roi_contour.py \
  ${in_ref_anat_img_fname} \
  ${out_fname} \
  --labels ${labels[@]} \
  --in_labelmap_fnames "${in_labelmap_fnames[@]}" \
  --in_labelmap_names "${in_labelmap_names[@]}" \
  --origin ${origin[@]} \
  --transparent_bckgnd \
  --adjust_to_bbox

# DWI
in_folder_label=dmri_hcp_sphm_b1000-2000-3000
file_basename_label=dwi_spherical_mean-b1000-2000-3000_resized
in_pred_root_dirname=${inf_pred_root_dirname}/${in_folder_label}/${fold}/results/prediction/
in_pred_fname=${in_pred_root_dirname}/${participant_id}__${file_basename_label}_pred.nii.gz
in_labelmap_fnames=("${in_gnd_th_fname}" "${in_pred_fname}")
in_labelmap_names=("${gnd_th_labelmap_label}" "${dwi_labelmap_label}")

out_folder_label=dmri_hcp_sphm_b1000-2000-3000
out_fname=${out_root_dirname}/${out_folder_label}/${participant_id}__${file_basename_label}_${label_name}_contour.png

echo "Contrast:" ${in_folder_label}

python ~/src/dmriseg/scripts/plot_roi_contour.py \
  ${in_ref_anat_img_fname} \
  ${out_fname} \
  --labels ${labels[@]} \
  --in_labelmap_fnames "${in_labelmap_fnames[@]}" \
  --in_labelmap_names "${in_labelmap_names[@]}" \
  --origin ${origin[@]} \
  --transparent_bckgnd \
  --adjust_to_bbox

# DWI1k
in_folder_label=dmri_hcp_sphm_b1000
file_basename_label=dwi_spherical_mean-b1000_resized
in_pred_root_dirname=${inf_pred_root_dirname}/${in_folder_label}/${fold}/results/prediction/
in_pred_fname=${in_pred_root_dirname}/${participant_id}__${file_basename_label}_pred.nii.gz
in_labelmap_fnames=("${in_gnd_th_fname}" "${in_pred_fname}")
in_labelmap_names=("${gnd_th_labelmap_label}" "${dwi1k_labelmap_label}")

out_folder_label=dmri_hcp_sphm_b1000
out_fname=${out_root_dirname}/${out_folder_label}/${participant_id}__${file_basename_label}_${label_name}_contour.png

echo "Contrast:" ${in_folder_label}

python ~/src/dmriseg/scripts/plot_roi_contour.py \
  ${in_ref_anat_img_fname} \
  ${out_fname} \
  --labels ${labels[@]} \
  --in_labelmap_fnames "${in_labelmap_fnames[@]}" \
  --in_labelmap_names "${in_labelmap_names[@]}" \
  --origin ${origin[@]} \
  --transparent_bckgnd \
  --adjust_to_bbox
