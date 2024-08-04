# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from vtkmodules.vtkCommonCore import vtkLookupTable

from dmriseg.data.lut.utils import (
    build_atlas_version_kwargs,
    get_atlas_cmap,
    get_atlas_from_cmap_name,
    read_lut_from_tsv2,
)


def rescale_int_colors(colors):
    """Rescale RGB color values in the [0-255] range to the [0-1] range."""

    colors_rescaled = []
    for rgb in colors:
        rgb = map(lambda x: x / 255, rgb)
        colors_rescaled.append(tuple(rgb))

    return colors_rescaled


# ToDo
# Merge with the above
def normalize_colors(lut):

    # Need colors to be in [0,1] for VTK
    cmap = {key: tuple(np.array(values) / 255) for key, values in lut.items()}

    return cmap


def rgb2rgba(rgb, alpha_value=1):
    """Convert an RGB value to an RGBA.

    Parameters
    ----------
    rgb : list
        RGB values.
    alpha_value : int
        Alpha value.

    Returns
    -------
    An RGBA value.
    """

    # ToDo
    # Deal with rgb values that are not a list
    alpha = np.expand_dims(np.ones(np.asarray(rgb).shape[0]) * alpha_value, -1)

    return np.hstack((np.asarray(rgb), alpha))


def create_mpl_cmap_from_atlas(atlas, name, **kwargs):
    cmap = get_atlas_cmap(atlas, **kwargs)
    return mpl.colors.ListedColormap(cmap, name=name)


def get_or_create_cmap(cmap_name):
    from pathlib import Path

    try:
        return plt.get_cmap(cmap_name)
    except ValueError:
        # ToDo
        # Clean this: workaround to be able to load a custom colormap from a
        # file
        if Path(cmap_name).is_file():
            lut = read_lut_from_tsv2(cmap_name)
            # Need colors to be in [0,1] for matplotlib
            _cmap = list(
                [list(map(lambda x: x / 255, val)) for val in lut.values()]
            )
            # _cmap = list([[204/255, 0, 255/255] for val in lut.values()])
            # _cmap = list([[1, 0, 0] for val in lut.values()])
            # Name colormap with the file rootname
            cmap_name = Path(cmap_name).with_suffix("").stem
            cmap = mpl.colors.ListedColormap(_cmap, name=cmap_name)
            # cmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, _cmap)
        else:
            atlas, version = get_atlas_from_cmap_name(cmap_name)
            kwargs = build_atlas_version_kwargs(atlas, version)
            cmap = create_mpl_cmap_from_atlas(atlas, cmap_name, **kwargs)
        # ToDo
        # This should not be called from a user loop
        # Avoid registering colormap twice
        from matplotlib import colormaps

        if cmap_name not in list(colormaps):
            plt.register_cmap(cmap_name, cmap)
        return plt.get_cmap(cmap_name)


def create_vtk_color_lut(cmap_name):

    if not Path(cmap_name).is_file():
        raise NotImplementedError(
            f"Not implemented for ``cmap_name`` values that do not exist as a file. Provided: {cmap_name}."
        )

    lut = read_lut_from_tsv2(cmap_name)
    # Need colors to be in [0,1] for VTK
    colors = list([list(map(lambda x: x / 255, val)) for val in lut.values()])
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(len(colors))
    lut.SetTableRange(0, lut.GetNumberOfTableValues())

    for i in range(lut.GetNumberOfTableValues()):
        lut.SetTableValue(i, [*colors[i], 1.0])

    return lut
