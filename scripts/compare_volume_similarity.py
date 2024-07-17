#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensure that the volume similarity computed using the three variants yields the
same value.
"""

import nibabel as nib
import numpy as np
import seg_metrics.seg_metrics as sg
import SimpleITK as sitk

from dmriseg.analysis.measures import compute_labelmap_volume


def compute_volume(gt_img, pred_img, label_list):
    img_data = gt_img.get_fdata()
    pred_data = pred_img.get_fdata()

    img_res = gt_img.header.get_zooms()
    pred_res = pred_img.header.get_zooms()
    assert img_res == pred_res

    gt_vol = compute_labelmap_volume(img_data, label_list, img_res)
    pred_vol = compute_labelmap_volume(pred_data, label_list, pred_res)

    return gt_vol, pred_vol


def volume_similarity_voxels(gt, pred):

    pred_vol, gt_vol = np.sum(pred), np.sum(gt)
    return 1.0 - np.abs(pred_vol - gt_vol) / (pred_vol + gt_vol)


def volume_similarity_volume(gt_vol, pred_vol):
    return 1.0 - np.abs(pred_vol - gt_vol) / (pred_vol + gt_vol)


def volume_similarity_rates(tp, fp, fn):
    return 1 - np.abs(fn - fp) / (2 * tp + fp + fn)


def main():

    gdth_fpath = "/mnt/data/cerebellum_parc/experiments_minimal_pipeline/labelmaps/fold-1/test_set/101107__cer_seg_resized.nii.gz"
    pred_fpath = "/mnt/data/cerebellum_parc/experiments_minimal_pipeline/dmri_hcp_t1/fold-1/results/prediction/101107__t1_resized_pred.nii.gz"

    # Read images and convert it to numpy array.
    gdth_img = sitk.ReadImage(gdth_fpath)
    gdth_np = sitk.GetArrayFromImage(gdth_img)

    pred_img = sitk.ReadImage(pred_fpath)
    pred_np = sitk.GetArrayFromImage(
        pred_img
    )  # note: image shape order: (z,y,x)

    spacing = np.array(
        list(reversed(pred_img.GetSpacing()))
    )  # note: after reverseing, spacing order =(z,y,x)

    labels = list(range(1, 35))
    metrics = sg.write_metrics(
        labels=labels,
        gdth_img=gdth_np,
        pred_img=pred_np,
        spacing=spacing,
        TPTNFPFN=True,
    )[0]

    gdth_img_nib = nib.load(gdth_fpath)
    pred_img_nib = nib.load(pred_fpath)

    gdth_img_nib_data = gdth_img_nib.get_fdata()
    pred_img_nib_data = pred_img_nib.get_fdata()

    assert np.allclose(gdth_img_nib_data, np.swapaxes(gdth_np, 0, 2))
    assert np.allclose(pred_img_nib_data, np.swapaxes(pred_np, 0, 2))

    gt_vol, pred_vol = compute_volume(gdth_img_nib, pred_img_nib, labels)

    vs_voxels_sitk = []
    vs_voxels_nib = []
    vs_vols = []
    vs_rates = []
    for idx, label in enumerate(labels):

        vs_voxel_sitk = volume_similarity_voxels(
            gdth_np[gdth_np == label], pred_np[pred_np == label]
        )
        vs_voxels_sitk.append(vs_voxel_sitk)

        vs_voxel_nib = volume_similarity_voxels(
            gdth_img_nib_data[gdth_img_nib_data == label],
            pred_img_nib_data[pred_img_nib_data == label],
        )
        vs_voxels_nib.append(vs_voxel_nib)

        vs_vol = volume_similarity_volume(gt_vol[idx], pred_vol[idx])
        vs_vols.append(vs_vol)

        vs_rate = volume_similarity_rates(
            metrics["TP"][idx],
            metrics["FP"][idx],
            metrics["FN"][idx],
        )
        vs_rates.append(vs_rate)

    assert np.allclose(vs_voxels_sitk, vs_vols)
    assert np.allclose(vs_voxels_sitk, vs_voxels_nib)
    assert np.allclose(vs_voxels_nib, vs_rates)


if __name__ == "__main__":
    main()
