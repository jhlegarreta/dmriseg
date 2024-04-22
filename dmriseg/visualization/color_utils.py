# -*- coding: utf-8 -*-


def rescale_int_colors(colors):
    """Rescale RGB color values in the [0-255] range to the [0-1] range."""

    colors_rescaled = []
    for rgb in colors:
        rgb = map(lambda x: x / 255, rgb)
        colors_rescaled.append(tuple(rgb))

    return colors_rescaled
