#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import pytest

from dmriseg.data.lut.utils import (
    Atlas,
    SuitAtlasBucknerVersion,
    SuitAtlasXueVersion,
    build_atlas_version_kwargs,
    get_atlas_cmap,
)

# from dmriseg.image.utils import extract_roi_from_label_image
from dmriseg.visualization.plot_utils import show_mplt_colormap

# from dmriseg.visualization.utils import overlay


def test_overlay():
    cmap = get_atlas_cmap(Atlas.DKT)
    print(cmap)


def test_overlay_contour():
    pass


def test_fury_colorbar():
    # scene_array = plot_fury_colorbar(cmap_lut, scalar_bar_title, size, fname)
    # scene_array_list.append(scene_array)
    # anatomical_views += (anatomical_view,)
    # anatomical_view_names += ("Colors",)
    pass


@pytest.mark.skip(reason="Failing. Need to investigate.")
def test_mpl_colorbar():

    atlas = Atlas.DKT
    # cmap = fetch_atlas_cmap(atlas)
    cmap = get_atlas_cmap(atlas)
    # cmap = plt.get_cmap(cmap_name)
    # cmap_mpl = mpl.colors.LinearSegmentedColormap.from_list(
    #    'freesurfer_dkt', cmaplist, len(cmap))
    cmap_mpl = mpl.colors.ListedColormap(cmap)
    # cmap_mpl = mpl.colors.ListedColormap(cmap)

    # cmap_mpl = plt.get_cmap('nipy_spectral')
    figsize = (15, 10)
    fig = show_mplt_colormap(cmap_mpl, figsize=figsize)
    fig.show()

    atlas = Atlas.__members__.values()

    buckner_version = SuitAtlasBucknerVersion.__members__.values()
    xue_version = SuitAtlasXueVersion.__members__.values()

    args = []
    for _atlas in atlas:

        if _atlas == Atlas.BUCKNER:
            for version in buckner_version:
                kwarg_args = build_atlas_version_kwargs(_atlas, version)
                args.append((_atlas, kwarg_args))
        elif _atlas == Atlas.XUE:
            for version in xue_version:
                kwarg_args = build_atlas_version_kwargs(_atlas, version)
                args.append((_atlas, kwarg_args))
        else:
            kwarg_args = dict({})
            args.append((_atlas, kwarg_args))

    for _atlas, kwargs in args:

        cmap = get_atlas_cmap(_atlas, **kwargs)
        cmap_mpl = mpl.colors.ListedColormap(cmap)
        fig = show_mplt_colormap(cmap_mpl, figsize=figsize)
        fig.show()
