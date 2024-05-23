# -*- coding: utf-8 -*-

import enum

import numpy as np
from fury import window
from PIL import Image, ImageOps

from dmriseg._utils import fill_doc
from dmriseg.anatomy.utils import Axis
from dmriseg.visualization.actor_utils import (
    create_volume_contour_actor,
    create_volume_slice_actor,
)
from dmriseg.visualization.color_utils import get_or_create_cmap
from dmriseg.visualization.utils import AnatomicalView


class CamParams(enum.Enum):
    VIEW_POS = "view_position"
    VIEW_CENTER = "view_center"
    VIEW_UP = "up_vector"
    ZOOM_FACTOR = "zoom_factor"


contour_actor_kwargs_name = "contour_actor_kwargs"
slice_actor_kwargs_name = "slice_actor_kwargs"

# ToDo
# Eventually all this should be put to a specific package to reuse code


@fill_doc
def screenshot_slice(img, axis, slice_ids, size):
    """Take a screenshot of the given image with the appropriate slice data at
    the provided slice indices.

    Parameters
    ----------
    %(img)s
    axis_name : str
        Slicing axis name.
    slice_ids : array-like
        Slice indices.
    %(size)s

    Returns
    -------
    scene_container : list
        Scene screenshot data container.
    """

    scene_container = []

    for idx in slice_ids:

        slice_actor = create_volume_slice_actor(
            img.get_fdata(),
            axis,
            idx,
            offset=0.0,
        )
        scene = create_scene([slice_actor], axis, idx, img.shape)
        scene_arr = window.snapshot(scene, size=size)
        scene_container.append(scene_arr)

    return scene_container


@fill_doc
def check_mosaic_layout(img_count, rows, cols):
    """Check whether a mosaic can be built given the image count and the
    requested number of rows and columns. Raise a `ValueError` if it cannot be
    built.

    Parameters
    ----------
    img_count : int
        Image count to be arranged in the mosaic.
    %(rows)s
    %(cols)s
    """

    cell_count = rows * cols

    if img_count < cell_count:
        raise ValueError(
            f"Less slices than cells requested.\nImage count: {img_count}; "
            f"Cell count: {cell_count} (rows: {rows}; cols: {cols}).\n"
            "Please provide an appropriate value for the rows, cols for the "
            "slice count."
        )
    elif img_count > cell_count:
        raise ValueError(
            f"More slices than cells requested.\nImage count: {img_count}; "
            f"Cell count: {cell_count} (rows: {rows}; cols: {cols}).\n"
            "Please provide an appropriate value for the rows, cols for the "
            "slice count."
        )


# ToDo
# Taken from SCILPY, see if I should use tractolearn's
def create_texture_slicer():
    pass


def create_slice_roi_scene(structural_img, roi_img, axis, slice_idx, **kwargs):

    contour_actor_kwargs = kwargs.pop(contour_actor_kwargs_name, {})
    slice_actor_kwargs = kwargs.pop(slice_actor_kwargs_name, {})

    # Create contour actors
    contour_actor = []
    # from dmriseg.visualization.actor_utils import create_volume_contour_actor2

    for img, color, opacity in zip(
        roi_img, contour_actor_kwargs["color"], contour_actor_kwargs["opacity"]
    ):
        _actor = create_volume_contour_actor(img, color=color, opacity=opacity)
        contour_actor.append(_actor)

    # Create the slice actor
    slice_actor = create_volume_slice_actor(
        structural_img.get_fdata(),
        axis,
        slice_idx,
        **slice_actor_kwargs,
    )

    # ToDo
    # Use create_scene
    scene = window.Scene()
    scene.add(slice_actor)

    for _actor in contour_actor:
        scene.add(_actor)

    return scene


# ToDo
# Taken from SCILPY, see if I should use tractolearn's
def initialize_camera(axis, slice_index, volume_shape):
    """
    Initialize a camera for a given orientation.

    Parameters
    ----------
    axis : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.

    Returns
    -------
    camera : dict
        Dictionary containing camera information.
    """
    camera = dict({})
    # Tighten the view around the data
    camera[CamParams.ZOOM_FACTOR] = 2.0 / max(volume_shape)
    # heuristic for setting the camera position at a distance
    # proportional to the scale of the scene
    eye_distance = max(volume_shape)
    if axis == Axis.AXIAL:
        if slice_index is None:
            slice_index = volume_shape[2] // 2
        camera[CamParams.VIEW_POS] = np.array(
            [
                (volume_shape[0] - 1) / 2.0,
                (volume_shape[1] - 1) / 2.0,
                -eye_distance,
            ]
        )
        camera[CamParams.VIEW_CENTER] = np.array(
            [
                (volume_shape[0] - 1) / 2.0,
                (volume_shape[1] - 1) / 2.0,
                slice_index,
            ]
        )
        camera[CamParams.VIEW_UP] = np.array([0.0, 1.0, 0.0])
    elif axis == Axis.CORONAL:
        if slice_index is None:
            slice_index = volume_shape[1] // 2
        camera[CamParams.VIEW_POS] = np.array(
            [
                (volume_shape[0] - 1) / 2.0,
                eye_distance,
                (volume_shape[2] - 1) / 2.0,
            ]
        )
        camera[CamParams.VIEW_CENTER] = np.array(
            [
                (volume_shape[0] - 1) / 2.0,
                slice_index,
                (volume_shape[2] - 1) / 2.0,
            ]
        )
        camera[CamParams.VIEW_UP] = np.array([0.0, 0.0, 1.0])
    elif axis == Axis.SAGITTAL:
        if slice_index is None:
            slice_index = volume_shape[0] // 2
        camera[CamParams.VIEW_POS] = np.array(
            [
                -eye_distance,
                (volume_shape[1] - 1) / 2.0,
                (volume_shape[2] - 1) / 2.0,
            ]
        )
        camera[CamParams.VIEW_CENTER] = np.array(
            [
                slice_index,
                (volume_shape[1] - 1) / 2.0,
                (volume_shape[2] - 1) / 2.0,
            ]
        )
        camera[CamParams.VIEW_UP] = np.array([0.0, 0.0, 1.0])
    else:
        # raise ValueError(f"Invalid axis name: {orientation}")
        raise ValueError(f"Invalid axis name: {axis}")
    return camera


# ToDo
# Taken from SCILPY, see if I should use tractolearn's
def create_scene(
    actors, orientation, slice_index, volume_shape, bg_color=(0, 0, 0)
):

    camera = initialize_camera(orientation, slice_index, volume_shape)

    scene = window.Scene()
    scene.background(bg_color)
    scene.projection("parallel")
    scene.set_camera(
        position=camera[CamParams.VIEW_POS],
        focal_point=camera[CamParams.VIEW_CENTER],
        view_up=camera[CamParams.VIEW_UP],
    )
    scene.zoom(camera[CamParams.ZOOM_FACTOR])

    # Add actors to the scene
    for curr_actor in actors:
        scene.add(curr_actor)

    return scene


def rgb2gray4pil(rgb_arr, alpha=255):
    """Convert an RGB array to grayscale and convert to `uint8` so that it can
    be appropriately handled by `PIL`.

    Parameters
    ----------
    rgb_arr : ndarray
        RGB value data.

    Returns
    -------
    Grayscale `unit8` data.
    """

    def _rgb2gray(rgb):
        img = Image.fromarray(np.uint8(rgb * 255)).convert("L")
        # ToDo
        # When calling img.show() the below has the expected effect, but adds
        # two channels
        # img.putalpha(alpha)
        return np.array(img)

    # Convert from RGB to grayscale
    gray_arr = _rgb2gray(rgb_arr)

    # Relocate overflow values to the dynamic range
    return (gray_arr * 255).astype("uint8")


@fill_doc
def create_image_from_scene(scene, size, mode=None, cmap_name=None):
    """Create a `PIL.Image` from the scene data.

    Parameters
    ----------
    %(scene)s
    %(size)s
    mode : str, optional
        Type and depth of a pixel in the `Pillow` image.
    cmap_name : str, optional
        Colormap name.

    Returns
    -------
    %(image)s
    """

    _arr = scene
    if cmap_name:
        # Apply the colormap
        # cmap = plt.get_cmap(cmap_name)
        cmap = get_or_create_cmap(cmap_name)
        # data returned by cmap is normalized to the [0,1] range: scale to the
        # [0, 255] range and convert to uint8 for Pillow
        _arr = (cmap(_arr) * 255).astype("uint8")

    # Need to flip the array due to some bug in the FURY image buffer. Might be
    # solved in newer versions of the package.
    image = Image.fromarray(_arr, mode=mode).transpose(Image.FLIP_TOP_BOTTOM)

    return image.resize(size, Image.LANCZOS)


@fill_doc
def create_mask_from_scene(scene, size):
    """Create a binary `PIL.Image` from the scene data.

    Parameters
    ----------
    %(scene)s
    %(size)s

    Returns
    -------
    %(image)s
    """

    _bin_arr = scene > 0
    _bin_arr = rgb2gray4pil(_bin_arr) * 255
    image = create_image_from_scene(_bin_arr, size)

    return image


@fill_doc
def draw_scene_at_pos(
    canvas,
    scene,
    size,
    left_pos,
    top_pos,
    mask=None,
    labelmap_overlay=None,
    vol_cmap_name=None,
    labelmap_cmap_name=None,
):
    """Draw a scene in the given target image at the specified position.

    Parameters
    ----------
    %(canvas)s
    %(scene)s
    %(size)s
    %(left_pos)s
    %(top_pos)s
    %(mask)s
    labelmap_overlay : ndarray
        Labelmap overlay scene data to be drawn.
    vol_cmap_name : str, optional
        Colormap name for the image scene data.
    labelmap_cmap_name : str, optional
        Colormap name for the labelmap overlay scene data.
    """

    image = create_image_from_scene(scene, size, cmap_name=vol_cmap_name)

    mask_img = None
    if mask is not None:
        mask_img = create_image_from_scene(mask, size, mode="L")

    canvas.paste(image, (left_pos, top_pos), mask=mask_img)

    # Draw the labelmap overlay image if any
    if labelmap_overlay is not None:
        labelmap_img = create_image_from_scene(
            labelmap_overlay, size, cmap_name=labelmap_cmap_name
        )

        # Create a mask over the labelmap overlay image
        label_mask = create_mask_from_scene(labelmap_overlay, size)

        canvas.paste(labelmap_img, (left_pos, top_pos), mask=label_mask)


@fill_doc
def compute_canvas_size(
    cell_width,
    cell_height,
    overlap_horiz,
    overlap_vert,
    rows,
    cols,
):
    """Compute the size of a canvas with the given number of rows and columns,
    and the requested cell size and overlap values.

    Parameters
    ----------
    %(cell_width)s
    %(cell_height)s
    %(overlap_horiz)s
    %(overlap_vert)s
    %(rows)s
    %(cols)s
    """

    def _compute_canvas_length(line_count, cell_length, overlap):
        return (line_count - 1) * (cell_length - overlap) + cell_length

    width = _compute_canvas_length(cols, cell_width, overlap_horiz)
    height = _compute_canvas_length(rows, cell_height, overlap_vert)

    return width, height


@fill_doc
def create_canvas(
    cell_width,
    cell_height,
    overlap_horiz,
    overlap_vert,
    rows,
    cols,
):
    """Create a canvas for given number of rows and columns, and the requested
    cell size and overlap values.

    Parameters
    ----------
    %(cell_width)s
    %(cell_height)s
    %(overlap_horiz)s
    %(overlap_vert)s
    %(rows)s
    %(cols)s
    """

    width, height = compute_canvas_size(
        cell_width, cell_height, overlap_horiz, overlap_vert, rows, cols
    )
    mosaic = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    return mosaic


@fill_doc
def compute_cell_topleft_pos(idx, cols, offset_h, offset_v):
    """Compute the top-left position of a cell to be drawn in a mosaic.

    Parameters
    ----------
    idx : int
       Cell index in the mosaic.
    %(cols)s
    offset_h : int
        Horizontal offset (pixels).
    offset_v : int
        Vertical offset (pixels).
    """

    row_idx = int(np.floor(idx / cols))
    top_pos = row_idx * offset_v
    col_idx = idx % cols
    left_pos = col_idx * offset_h

    return top_pos, left_pos


@fill_doc
def compose_mosaic(
    img_scene_container,
    mask_scene_container,
    cell_size,
    rows,
    cols,
    overlap_factor,
    labelmap_scene_container=None,
    vol_cmap_name=None,
    labelmap_cmap_name=None,
):
    """Create the mosaic canvas for given number of rows and columns, and the
    requested cell size and overlap values.

    Parameters
    ----------
    img_scene_container : list
        Image scene data container.
    mask_scene_container : list
        Mask scene data container.
    cell_size : array-like
        Cell size (pixels) (width, height).
    %(rows)s
    %(cols)s
    overlap_factor : array-like
        Overlap factor (horizontal, vertical).
    labelmap_scene_container : list, optional
        Labelmap scene data container.
    vol_cmap_name : str, optional
        Colormap name for the image scene data.
    labelmap_cmap_name : str, optional
        Colormap name for the labelmap scene data.
    """

    def _compute_overlap_length(length, _overlap):
        return round(length * _overlap)

    cell_width = cell_size[0]
    cell_height = cell_size[1]

    overlap_h = _compute_overlap_length(cell_width, overlap_factor[0])
    overlap_v = _compute_overlap_length(cell_width, overlap_factor[1])

    mosaic = create_canvas(*cell_size, overlap_h, overlap_v, rows, cols)

    offset_h = cell_width - overlap_h
    offset_v = cell_height - overlap_v
    from itertools import zip_longest

    for idx, (img_arr, mask_arr, labelmap_arr) in enumerate(
        list(
            zip_longest(
                img_scene_container,
                mask_scene_container,
                labelmap_scene_container,
                fillvalue=tuple(),
            )
        )
    ):

        # Compute the mosaic cell position
        top_pos, left_pos = compute_cell_topleft_pos(
            idx, cols, offset_h, offset_v
        )

        # Convert the scene data to grayscale and adjust for handling with
        # Pillow
        _img_arr = rgb2gray4pil(img_arr)
        _mask_arr = rgb2gray4pil(mask_arr)

        _labelmap_arr = None
        if len(labelmap_arr):
            _labelmap_arr = rgb2gray4pil(labelmap_arr, alpha=127)

        # Draw the image (and labelmap overlay, if any) in the cell
        draw_scene_at_pos(
            mosaic,
            _img_arr,
            (cell_width, cell_height),
            left_pos,
            top_pos,
            mask=_mask_arr,
            labelmap_overlay=_labelmap_arr,
            vol_cmap_name=vol_cmap_name,
            labelmap_cmap_name=labelmap_cmap_name,
        )

    return mosaic


def create_overlay(
    img, mask, label, boundary, label_alpha=0.3, boundary_alpha=0.3
):
    # Assume all are PIL images
    brain_im_rgba = img.copy()
    brain_im_rgba.putalpha(img.convert("L"))

    label_im_rgba = label.copy()
    label_im_rgba.putalpha(mask.convert("L"))

    mask2 = ImageOps.invert(boundary.convert("RGB"))
    boundary.paste(im=mask2, mask=mask2.convert("L"))
    boundary.show()

    label_img2_rgba = boundary.copy()
    label_img2_rgba.putalpha(mask.convert("L"))

    blend1 = Image.blend(brain_im_rgba, label_im_rgba, alpha=label_alpha)

    return Image.blend(blend1, label_img2_rgba, alpha=boundary_alpha)


def transform_scene_focal_point(
    scene, anatomical_view=AnatomicalView.AXIAL_SUPERIOR
):

    # Re-initialize camera
    position = [0, 0, 1]
    focal_point = [0, 0, 0]
    view_up = [0, 1, 0]
    scene.set_camera(position, focal_point, view_up)

    # ToDo
    # Assuming that the volume in the scene is always oriented the same way?
    if anatomical_view == AnatomicalView.AXIAL_SUPERIOR:
        pass
    elif anatomical_view == AnatomicalView.AXIAL_INFERIOR:
        scene.pitch(180)
    elif anatomical_view == AnatomicalView.CORONAL_ANTERIOR:
        scene.pitch(270)
        scene.set_camera(view_up=(0, 0, 1))
    elif anatomical_view == AnatomicalView.CORONAL_POSTERIOR:
        scene.pitch(90)
        scene.set_camera(view_up=(0, 0, 1))
    elif anatomical_view == AnatomicalView.SAGITTAL_LEFT:
        scene.yaw(-90)
        scene.roll(90)
    elif anatomical_view == AnatomicalView.SAGITTAL_RIGHT:
        scene.yaw(90)
        scene.roll(-90)
    else:
        raise ValueError(
            "Unknown anatomical view type.\n"
            f"Found: {anatomical_view}; Available: {AnatomicalView._member_names_}"
        )

    scene.reset_camera()


def compose_scene(
    actors,
    anatomical_view=AnatomicalView.AXIAL_SUPERIOR,
    background=window.colors.black,
):

    scene = window.Scene()

    [scene.add(_actor) for _actor in actors]

    scene.background(background)

    transform_scene_focal_point(scene, anatomical_view)

    return scene
