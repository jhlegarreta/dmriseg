# -*- coding: utf-8 -*-

import copy

import ants
import nibabel as nib
import numpy as np
import torch
from dipy.io.utils import is_header_compatible
from PIL import Image
from scipy.ndimage import distance_transform_edt as eucl_distance
from torch.nn.functional import one_hot

from dmriseg.anatomy.utils import Axis


def check_slice_indices(vol_img, axis, slice_ids):
    """Check that the given volume can be sliced at the given slice indices
    along the requested axis.

    Parameters
    ----------
    vol_img : nib.Nifti1Image
        Volume image.
    axis : Axis
        Slicing axis.
    slice_ids : array-like
        Slice indices.
    """

    shape = vol_img.shape
    if axis == Axis.AXIAL:
        idx = 2
    elif axis == Axis.CORONAL:
        idx = 1
    elif axis == Axis.SAGITTAL:
        idx = 0

    _slice_ids = list(filter(lambda x: x > shape[idx], slice_ids))
    if _slice_ids:
        raise ValueError(
            "Slice indices exceed the volume shape along the given axis:\n"
            f"Slices {_slice_ids} exceed shape {shape} along dimension {idx}."
        )


# ToDo
# Compute stats
def get_values(img):
    from scipy import stats

    return stats.describe(img.get_fdata())
    # return np.unique(img.get_fdata())


def create_mask_from_label_image(img, dtype=np.uint8):
    """Create a binary image from the provided image."""

    # ToDo
    # Propose a more elaborated algorithm to merge only, e.g. some of the
    # labels. Done in SCILPY
    return np.asanyarray(img.dataobj).astype(dtype)


# ToDo
# Borrowed from scilpy
def assert_same_resolution(images):
    """
    Check the resolution of multiple images.
    Parameters
    ----------
    images : list of string or string
        List of images or an image.
    """
    if isinstance(images, str):
        images = [images]

    if len(images) == 0:
        raise Exception(
            "Can't check if images are of the same "
            "resolution/affine. No image has been given"
        )

    for curr_image in images[1:]:
        if not is_header_compatible(images[0], curr_image):
            raise Exception("Images are not of the same resolution/affine")


def extract_roi_from_image(img, idx_mask):
    img_data = img.get_fdata()
    roi_img_data = np.where(idx_mask, img_data, 0)
    roi_img = nib.nifti1.Nifti1Image(roi_img_data, affine=img.affine)
    return roi_img


def extract_roi_from_label_image(img, labels):
    img_data = img.get_fdata()
    idx_mask = np.isin(img_data, labels)
    roi_img = extract_roi_from_image(img, idx_mask)
    return roi_img


def get_one_hot1(labels, num_classes):
    """From https://github.com/NathanHowell/kornia/blob/master/kornia/utils/one_hot.py#L59"""

    shape = labels.shape
    _one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], dtype=torch.int64
    )
    _labels = labels.type(torch.int64)
    _labels = _one_hot.scatter_(1, _labels.unsqueeze(1), 1.0)
    return _labels


def get_one_hot_2(labels, num_classes):
    """
    From the paper "A Generalized Surface Loss for Reducing the Hausdorff
    Distance in Medical Imaging Segmentation": https://arxiv.org/pdf/2302.03868.pdf
    https://github.com/aecelaya/gen-surf-loss
    """

    _labels = labels.to(torch.int64)
    _labels = one_hot(_labels, num_classes=num_classes)
    _labels = torch.transpose(_labels, dim0=4, dim1=1)  # dim0=5, dim1=1)
    _labels = torch.squeeze(_labels, dim=4)  # dim=5
    _labels = _labels.to(torch.int8)
    return _labels


def get_one_hot_k(labels, num_classes, dtype):
    """From
    https://github.com/kornia/kornia/blob/master/kornia/utils/one_hot.py#L6"""

    eps = 1e-6
    shape = labels.shape
    _one_hot = torch.zeros((shape[0], num_classes) + shape[1:], dtype=dtype)

    return _one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def one_hot2dist(seg: np.ndarray, resolution=None, dtype=None) -> np.ndarray:

    num_classes = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(num_classes):
        posmask = seg[k].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = (
                eucl_distance(negmask, sampling=resolution) * negmask
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
            )
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


# The below assumes the labelmap is an ANTsPy image, and contains resolution
# information. one_hot2dist is just the same, but takes a NumPy array and the
# resolution to obtain the same objective.
def compute_distance_transform_map(labelmap, labels):

    # Get dimensions of image in standard space
    dims = labelmap.shape

    # Convert to numpy and get one hot encoding
    mask_npy = labelmap.numpy()
    mask_onehot = np.zeros((*dims, len(labels)))
    dtm = np.zeros((*dims, len(labels)))
    for j in range(len(labels)):
        mask_onehot[..., j] = (mask_npy == labels[j]).astype("float32")

        # Only compute DTM if class exists in image
        if np.sum(mask_onehot[..., j]) > 0:
            dtm_j = mask_onehot[..., j].reshape(dims)
            dtm_j = copy.deepcopy(dtm_j)  # labelmap.new_image_like(data=dtm_j)
            dtm_j = ants.iMath(dtm_j, "MaurerDistance")
            # import simpleitk as sitk
            # sitk.SignedMaurerDistanceMap(
            #     dtm_j, squaredDistance=False, useImageSpacing=True)
            dtm[..., j] = dtm_j.numpy()

    return dtm


# ToDo
# This and visualization.scene_utils.create_mask_from_scene should be merged
def create_mask_from_rgb_array(img_array, background):
    # Create a boolean mask where all RGB values are not equal to background
    return ~np.all(img_array == background, axis=-1)


# ToDo
# This and visualization.scene_utils.create_mask_from_scene should be merged
def create_masked_image_from_array(img_array, background):

    _bckgnd_mask = create_mask_from_rgb_array(img_array, background)

    # Transform the boolean mask to the [0, 255] range and convert to uint8 for
    # Pillow
    bckgnd_mask = (_bckgnd_mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(bckgnd_mask)
    img = Image.fromarray(img_array, mode="RGB")
    img.putalpha(mask_img)

    return img
