#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
import pytest
from fury import window

from dmriseg.anatomy.utils import Axis
from dmriseg.image.utils import (
    create_mask_from_label_image,
    extract_roi_from_label_image,
)
from dmriseg.visualization.scene_utils import (
    contour_actor_kwargs_name,
    create_slice_roi_scene,
)


@pytest.mark.skip(reason="Need to have testing data available.")
def test_create_slice_roi_scene():

    # Load a SUIT segmentation
    fname = "/mnt/data/connectome/suit/results/cer_seg_100307.nii"
    label_img = nib.load(fname)

    # Extract ROI using the label of interest
    label = 8
    _label_img = extract_roi_from_label_image(label_img, label)

    # Create a binary image
    img_data = create_mask_from_label_image(_label_img)
    roi_img = nib.nifti1.Nifti1Image(img_data, affine=label_img.affine)

    # Load the corresponding T1
    fname = "/mnt/data/connectome/s1200_download_on_20170516/downloaded/3T_structural_preproc/100307/T1w/T1w_acpc_dc_restore_brain.nii.gz"
    structural_img = nib.load(fname)

    # cmap = get_atlas_cmap(Atlas.DKT)
    import numpy as np

    color = [
        np.array([1, 0, 0]),
    ]
    opacity = [
        1.0,
    ]
    contour_actor_kwargs = dict({"color": color, "opacity": opacity})
    kwargs = dict({contour_actor_kwargs_name: contour_actor_kwargs})

    axis = Axis.AXIAL
    slice_idx = 66
    scene = create_slice_roi_scene(
        structural_img, [roi_img], axis, slice_idx, **kwargs
    )

    size = (1024, 720)
    reset_camera = False
    showm = window.ShowManager(scene, size=size, reset_camera=reset_camera)
    showm.initialize()
    showm.start()
