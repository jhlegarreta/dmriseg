# -*- coding: utf-8 -*-

import enum
from typing import Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing


class AnatomicalView(enum.Enum):
    AXIAL_INFERIOR = "axial_inferior"
    AXIAL_SUPERIOR = "axial_superior"
    CORONAL_ANTERIOR = "coronal_anterior"
    CORONAL_POSTERIOR = "coronal_posterior"
    SAGITTAL_LEFT = "sagittal_left"
    SAGITTAL_RIGHT = "sagittal_right"

    @classmethod
    def _missing_(cls, value):
        choices = list(cls.__members__)
        raise ValueError(
            f"Unsupported value:\n" f"Found: {value}; Available: {choices}"
        )


# ToDo overlay of labels
# Use the methods in vtk_utils
# or else
# Use the fury.actor.contour_from_roi
#
# Option to fill with dimmer color when contouring (like in 3D Slicer)


def compute_plane_normal(anatomical_view):
    """."""

    # ToDo
    # Check these
    if anatomical_view == AnatomicalView.AXIAL_SUPERIOR:
        normal = (0, 0, -1)
    elif anatomical_view == AnatomicalView.AXIAL_INFERIOR:
        normal = (0, 0, 1)
    elif anatomical_view == AnatomicalView.CORONAL_ANTERIOR:
        normal = (0, 1, 0)
    elif anatomical_view == AnatomicalView.CORONAL_POSTERIOR:
        normal = (0, -1, 0)
    elif anatomical_view == AnatomicalView.SAGITTAL_LEFT:
        normal = (1, 0, 0)
    elif anatomical_view == AnatomicalView.SAGITTAL_RIGHT:
        normal = (-1, 0, 0)

    return normal


# ToDo
# Think if these should be split into a separate module from anatomical
# visualization tools
def plot_fury_colorbar(
    colormap, title, size: Tuple[int, int] = (1920, 1080), filename=None
):
    from fury import actor, window

    scene = window.Scene()
    bar = actor.scalar_bar(colormap, title)
    scene.add(bar)

    scene_array = window.snapshot(
        scene, fname=filename, size=size, offscreen=True
    )

    return scene_array


def compute_central_slices(img):
    """."""

    # Get the relevant slices from the image
    x_slice = img.shape[0] // 2
    y_slice = img.shape[1] // 2
    z_slice = img.shape[2] // 2

    return x_slice, y_slice, z_slice


def create_binary_image_from_file(fname):

    img = Image.open(fname).convert("L")

    # Threshold
    img = img.point(lambda p: 255 if p > 1 else 0)
    # Convert to monochrome
    return img.convert("1")


def close_binary_image(img):

    structure = np.ones((5, 5))
    closed_image = binary_closing(img, structure=structure)

    # Convert the result back to a PIL image
    return Image.fromarray((closed_image * 255).astype(np.uint8))


def create_mask_image_from_file(mask_scrnsht_fname):

    # Binarize the mask image
    binarized_img = create_binary_image_from_file(mask_scrnsht_fname)

    # Apply morphological closing to remove the border effects
    return close_binary_image(binarized_img)


def apply_mask_transparency(volume_scrnsht_fname, mask_scrnsht_fname):

    volume_img = Image.open(volume_scrnsht_fname).convert("RGBA")
    mask_img = Image.open(mask_scrnsht_fname).convert("L")  # grayscale
    volume_img.putalpha(mask_img)
    return volume_img


def mask_image(volume_scrnsht_fname, mask_scrnsht_fname):

    mask_img = create_mask_image_from_file(mask_scrnsht_fname)

    # Save the result
    mask_img.save(mask_scrnsht_fname)

    return apply_mask_transparency(volume_scrnsht_fname, mask_scrnsht_fname)
