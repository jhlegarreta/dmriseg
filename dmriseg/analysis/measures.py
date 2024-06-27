# -*- coding: utf-8 -*-

import copy
import enum

import nibabel as nib
import numpy as np
import seg_metrics.seg_metrics as sg
from scipy import ndimage
from scipy.spatial.distance import cdist


class Measure(enum.Enum):
    CENTER_OF_MASS_DISTANCE = "cm_dist"
    DICE = "dice"
    HAUSDORFF = "hd"
    HAUSDORFF95 = "hd95"
    HAUSDORFF99 = "hd99"
    JACCARD = "jaccard"
    MEAN_SURFACE_DISTANCE = "msd"
    GT_VOLUME = "pred_volume"
    PRED_VOLUME = "pred_volume"
    VOLUME_ERROR = "vol_err"
    VOLUME_SIMILARITY = "vs"


# ToDo
# Add a reporting module to serialize these into a CSV? or some io module?
def compute_measures():
    # dice = 0  # (sklearn or scipy have surely these)
    # and all the rest
    pass


def compute_surface_distance_old(input1, input2, sampling=1, connectivity=1):
    # input1 - the segmentation that has been created. It can be a multi-class
    # segmentation, but this function will make the image binary
    # input2 - the GT segmentation against which we wish to compare input1
    # sampling - the pixel resolution or pixel size. This is entered as an
    # n-vector where n is equal to the number of dimensions in the segmentation
    # i.e. 2D or 3D. The default value is 1 which means pixels (or rather
    # voxels) are 1 x 1 x 1 mm in size.
    # connectivity - creates either a 2D (3 x 3) or 3D (3 x 3 x 3) matrix
    # defining the neighbourhood around which the function looks for
    # neighbouring pixels. Typically, this is defined as a six-neighbour
    # kernel which is the default behaviour of this function.

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = ndimage.morphology.generate_binary_structure(
        input_1.ndim, connectivity
    )

    s = input_1 - ndimage.morphology.binary_erosion(input_1, conn)
    s_prime = input_2 - ndimage.morphology.binary_erosion(input_2, conn)

    dta = ndimage.morphology.distance_transform_edt(~s, sampling)
    dtb = ndimage.morphology.distance_transform_edt(~s_prime, sampling)

    sds = np.concatenate([np.ravel(dta[s_prime != 0]), np.ravel(dtb[s != 0])])

    return sds


def compute_distance(img1, img2, sampling=1, connectivity=1):
    # Taken from https://mlnotebook.github.io/post/surface-distance-function/
    # The function example below takes two segmentations (which both have
    # multiple classes). The sampling vector is a typical pixel-size from an
    # MRI scan and the 1 indicated I'd like a 6 neighbour (cross-shaped) kernel
    # for finding the edges.
    #  surface_distance = surfd(test_seg, GT_seg, [1.25, 1.25, 10],1)
    # By specifying the value of the voxel-label I'm interested in (assuming
    # we're talking about classes which are contiguous and not spread out),
    # we can find the surface accuracy of that class.
    #  surface_distance = compute_surface_distance_old(test_seg(test_seg==1), GT_seg(GT_seg==1), [1.25, 1.25, 10],1)

    img1_data = img1.get_fdata().astype(int)
    img2_data = img2.get_fdata().astype(int)

    surface_distance = compute_surface_distance_old(
        img1_data, img2_data, sampling=sampling, connectivity=connectivity
    )
    msd = surface_distance.mean()
    # rms = np.sqrt((surface_distance**2).mean())
    # hd = surface_distance.max()
    return msd


def compute_relevant_labels(img_data1, img_data2, labels, exclude_background):

    labels_img1 = sorted(map(int, np.unique(img_data1)))
    labels_img2 = sorted(map(int, np.unique(img_data2)))

    _labels = copy.deepcopy(labels)

    if exclude_background:
        _labels = list(np.asarray(labels)[np.asarray(labels) != 0])
    if exclude_background:
        labels_img1 = list(
            np.asarray(labels_img1)[np.asarray(labels_img1) != 0]
        )
    if exclude_background:
        labels_img2 = list(
            np.asarray(labels_img2)[np.asarray(labels_img2) != 0]
        )

    prsnt_labels = sorted(set(labels_img1).union(set(labels_img2)))

    assert len(prsnt_labels) <= len(_labels)

    msng_idx = np.where(~np.isin(_labels, prsnt_labels))[0]
    msng_labels = np.array(_labels)[msng_idx]
    return prsnt_labels, msng_labels, msng_idx


def fill_missing_values(metrics, labels, msng_idx):

    _metrics = metrics[0]

    prsnt_idx = sorted(set(range(len(labels))) - set(msng_idx))

    metrics_filled = copy.deepcopy(_metrics)

    # Fill all with nans
    for key, vals in metrics_filled.items():
        if key == "label":
            metrics_filled[key] = labels
        else:
            metrics_filled[key] = len(labels) * [np.nan]
            filled_list = metrics_filled[key]
            for value, index in zip(vals, prsnt_idx):
                filled_list[index] = (
                    value
                    if 0 <= index < len(filled_list)
                    else filled_list[index]
                )
            metrics_filled[key] = filled_list

    return [metrics_filled]


def compute_metrics(img1, img2, spacing, labels, exclude_background=True):
    """dice, jaccard, precision, recall, fpr, fnr, vs, hd, hd95, msd, mdsd,
    stdsd"""

    # https://mlnotebook.github.io/post/surface-distance-function/
    # Otherwise, use this: https://github.com/Jingnan-Jia/segmentation_metrics
    # Which uses SimpleITK
    # Colab
    # https://colab.research.google.com/drive/1LUH9cozeeSdmn9W_WdwBjKnrhWb39dq_?
    # Or SimpleITK directly:
    # https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb

    img1_data = img1.get_fdata().astype(int)
    img2_data = img2.get_fdata().astype(int)

    prsnt_labels, msng_labels, msng_idx = compute_relevant_labels(
        img1_data, img2_data, labels, exclude_background
    )

    # For some reason, seg_metrics is unable to work with memmaps
    metrics = sg.write_metrics(
        labels=prsnt_labels,
        gdth_img=np.asarray(img1_data),
        pred_img=np.asarray(img2_data),
        spacing=spacing,
    )

    # Fill in the metrics for the missing labels if any
    _metrics = copy.deepcopy(metrics)
    if len(msng_labels) != 0:
        _labels = labels
        if exclude_background:
            _labels = list(np.asarray(labels)[np.asarray(labels) != 0])
        _metrics = fill_missing_values(metrics, _labels, msng_idx)

    return _metrics  # metrics["msd"]


def compute_center_of_mass_distance(img1, img2, labels):

    aff1 = img1.affine
    aff2 = img2.affine
    assert np.allclose(aff1, aff2)

    img1_data = img1.get_fdata().astype(int)
    img2_data = img2.get_fdata().astype(int)

    # Compute the center of masses for the specified labels
    center_mass1 = [
        ndimage.center_of_mass(
            (img1_data == label).astype(float), labels=img1_data, index=label
        )
        for label in labels
    ]
    center_mass2 = [
        ndimage.center_of_mass(
            (img2_data == label).astype(float), labels=img2_data, index=label
        )
        for label in labels
    ]

    # Convert center of masses to mm
    # pt = [22, 34, 12]
    # p = nib.affines.apply_affine(aff1, pt)

    p = nib.affines.apply_affine(aff1, center_mass1)
    q = nib.affines.apply_affine(aff2, center_mass2)

    # Check if it is more consistent if returned center of masses are in mm
    # dist = np.linalg.norm(p - q, axis=1)
    dist = cdist(p, q)
    return np.diagonal(dist), center_mass1, center_mass2


def compute_labelmap_volume(img_data, label_list, resolution):

    labelwise_data = np.asarray(
        [np.where(img_data != value, 0, img_data) for value in label_list]
    )
    label_count = np.count_nonzero(
        labelwise_data, axis=(1, 2, 3), keepdims=False
    )
    return np.around(label_count * np.prod(resolution), 3)


def compute_error(gt_vol, pred_vol):

    # ToDo
    # Deal with the case where some labels may be absent in the ground truth:
    # the division will raise an encountered invalid value in division warning
    return (pred_vol - gt_vol) / gt_vol


def compute_volume_error(gt_img, pred_img, label_list):

    img_data = gt_img.get_fdata()
    pred_data = pred_img.get_fdata()

    img_res = gt_img.header.get_zooms()
    pred_res = pred_img.header.get_zooms()
    assert img_res == pred_res

    gt_vol = compute_labelmap_volume(img_data, label_list, img_res)
    pred_vol = compute_labelmap_volume(pred_data, label_list, pred_res)

    return compute_error(gt_vol, pred_vol)
