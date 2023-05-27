# -*- coding: utf-8 -*-


def rescale_float_colors(colors):
    """Rescale RGB color values in the [0-1] range to the [0-255] range.

    Parameters
    ----------
    colors : list
        RGB color values in the [0-1] range.

    Returns
    -------
    colors_rescaled : list
        RGB color values in the [0-255] range.
    """

    colors_rescaled = []
    for rgb in colors:
        rgb = map(lambda x: int(x * 255), rgb)
        colors_rescaled.append(tuple(rgb))

    return colors_rescaled
