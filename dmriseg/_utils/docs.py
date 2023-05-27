"""Functions related to the documentation.
docdict contains the standard documentation entries.
Entries are listed in alphabetical order.
Borrowed from nilearn:
https://github.com/nilearn/nilearn/blob/main/nilearn/_utils/docs.py
"""

import sys

###################################
# Standard documentation entries
#
docdict = dict()

# ToDo
# Use type hints and remove the type from here
docdict[
    "data_dir"
] = """
data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
    Path where data should be downloaded. By default,
    files are downloaded in home directory."""

# mask_img
docdict[
    "mask_img"
] = """
mask_img : Niimg-like object
    Object used for masking the data."""

docdict[
    "title"
] = """
title : :obj:`str`, or None, optional
    The title displayed on the figure.
    Default=None."""

docdict[
    "cmap"
] = """
cmap : :class:`matplotlib.colors.Colormap`, or :obj:`str`, optional
    The colormap to use. Either a string which is a name of
    a matplotlib colormap, or a matplotlib colormap object."""

docdict[
    "cell_width"
] = """
cell_width: int
    Cell width (pixels)."""

docdict[
    "cell_height"
] = """
cell_height : int
    Cell height (pixels)."""

docdict[
    "cols"
] = """
cols : int
    Column count."""

docdict[
    "canvas"
] = """
canvas : PIL.Image
    Canvas (target image)."""

docdict[
    "image"
] = """
image : PIL.Image
    Image."""

docdict[
    "img"
] = """
image : nib.Nifti1Image
    Volume image."""

docdict[
    "left_pos"
] = """
left_pos : int
    Left position (pixels)."""

docdict[
    "mask"
] = """
mask : ndarray, optional
    Transparency mask."""

docdict[
    "overlap_horiz"
] = """
overlap_horiz : int
    Horizontal overlap (pixels)."""

docdict[
    "overlap_vert"
] = """
overlap_vert : int
    Vertical overlap (pixels)."""

docdict[
    "rows"
] = """
rows : int
    Row count."""

docdict[
    "scene"
] = """
scene : ndarray
    Scene data to be drawn."""

docdict[
    "size"
] = """
size : array-like
    Image size (pixels) (width, height)."""

docdict[
    "top_pos"
] = """
top_pos : int
    Top position(pixels)."""


docdict_indented = {}


def _indent_count_lines(lines):
    """Minimum indent for all lines in line list.

    >>> _lines = [" one", "  two", "   three"]
    >>> _indent_count_lines(_lines)
    1
    >>> _lines = []
    >>> _indent_count_lines(_lines)
    0
    >>> _lines = [" one"]
    >>> _indent_count_lines(_lines)
    1
    >>> _indent_count_lines(["    "])
    0
    """

    indent_count = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent_count = min(indent_count, len(line) - len(stripped))
    if indent_count == sys.maxsize:
        return 0
    return indent_count


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function whose docstring is to be filled. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """

    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indent_count_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{str(exp)}")
    return f
