# -*- coding: utf-8 -*-

import enum
from typing import Tuple


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
