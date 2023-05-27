# -*- coding: utf-8 -*-

import enum
import json
import pathlib
from itertools import compress
from urllib.parse import urljoin

import importlib_resources
import pandas as pd
import requests
from SUITPy import (
    fetch_buckner_2011,
    fetch_diedrichsen_2009,
    fetch_ji_2019,
    fetch_king_2019,
    fetch_xue_2021,
)
from SUITPy.utils import _fetch_files, _get_dataset_dir

from dmriseg._utils import fill_doc
from dmriseg.data.lut.color_utils import rescale_float_colors
from dmriseg.io.file_extensions import (
    DelimitedValuesFileExtension,
    LutFileExtension,
    TextFileExtension,
    asterisk,
)
from dmriseg.io.utils import build_suffix

atlas_cmap_name_sep = "_"


class Atlas(enum.Enum):
    BUCKNER = "buckner"
    DIEDRICHSEN = "diedrichsen"
    DKT = "dkt"
    JI = "ji"
    KING = "king"
    XUE = "xue"


class FsAtlasCmaps(enum.Enum):
    DKT = "fs_dkt"


class SuitAtlasDirnameBase(enum.Enum):
    BUCKNER = "Buckner_2011"
    DIEDRICHSEN = "Diedrichsen_2009"
    JI = "Ji_2019"
    KING = "King_2019"
    XUE = "Xue_2021"


class SuitAtlasBucknerVersion(enum.Enum):
    R7 = "r7"
    R17 = "r17"


class SuitAtlasXueVersion(enum.Enum):
    SUB1 = "Sub1"
    SUB2 = "Sub2"


class ColormapName(enum.Enum):
    BUCKNER_R7 = "buckner_r7"
    BUCKNER_R17 = "buckner_r17"
    DIEDRICHSEN = "diedrichsen"
    DKT_FS = "dkt_fs"
    JI = "ji"
    KING = "king"
    XUE_SUB1 = "xue_sub1"
    XUE_SUB2 = "xue_sub2"


suit_atlas_gh_url_base = (
    "https://raw.githubusercontent.com/DiedrichsenLab/cerebellar_atlases"
)
suit_atlas_gh_revision = "29245e7471485517f8c4dab9711629958b5956d5"
fwd_slash_separator = "/"

buckner_version_arg = "buckner_version"
xue_version_arg = "xue_version"

suit_atlas_version_arg = "suit_atlas_version"
suit_file_fetch_arg = "suit_file_fetch"
lut_extension_arg = "lut_extension"


def get_atlas_from_cmap_name(cmap_name):

    atlas_val, version_val = ColormapName(cmap_name).name.split(
        atlas_cmap_name_sep
    )

    atlas = Atlas(atlas_val.lower())

    if atlas == Atlas.BUCKNER:
        version = SuitAtlasBucknerVersion(version_val.lower())
    elif atlas == Atlas.XUE:
        version = SuitAtlasXueVersion(version_val.lower())
    else:
        version = None

    return atlas, version


def build_atlas_version_kwargs(atlas, version):

    if atlas == Atlas.BUCKNER:
        kwargs = dict({buckner_version_arg: version})
    elif atlas == Atlas.XUE:
        kwargs = dict({xue_version_arg: version})
    else:
        kwargs = dict({})

    return kwargs


# ToDo
# Making this as general as possible for non txt files, or files formatted
# differently will be hard
def read_lut_from_txt(fname):

    lut = dict()
    with fname.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            (key, name, r, g, b, a) = line.split()
            lut[int(key)] = (name, (int(r), int(g), int(b), int(a)))

    return lut


@fill_doc
def get_cmap_from_lut(atlas_lut):

    # Drop the name, and return it as a list
    return list([val[-1][:3] for val in atlas_lut.values()])


def get_local_atlas_cmap_lut_fname(atlas):

    path = importlib_resources.files(__name__).joinpath("anatomy_lut.json")

    with path.open() as f:
        datasets_cmaps = json.load(f)

    return pathlib.Path(str(path)).parent.joinpath(datasets_cmaps[atlas.value])


def fetch_fs_dkt_lut_cmap():

    return get_local_atlas_cmap_lut_fname(FsAtlasCmaps.DKT)


def load_fs_dkt_lut_cmap():

    cmap_fname = fetch_fs_dkt_lut_cmap()

    atlas_lut = read_lut_from_txt(cmap_fname)

    # Convert values to the [0,1] range used by matplotlib colors, and drop
    # the alpha value
    atlas_lut = dict(
        {
            key: tuple([val[0], tuple(map(lambda x: x / 255.0, val[1][:3]))])
            for key, val in atlas_lut.items()
        }
    )

    return atlas_lut


def fetch_suit_atlas(atlas, **kwargs):

    # ToDo
    # This assumes we have Internet access, which may not be the case on a
    # compute node. Think what is the best strategy. We could have done the
    # same for the FS atlas

    data_arg = "data"
    _data = "con"
    if atlas == Atlas.KING:
        _data = kwargs.pop(data_arg, _data)

    if atlas == Atlas.BUCKNER:
        atlas_data = fetch_buckner_2011(**kwargs)
    elif atlas == Atlas.DIEDRICHSEN:
        atlas_data = fetch_diedrichsen_2009(**kwargs)
    elif atlas == Atlas.JI:
        atlas_data = fetch_ji_2019(**kwargs)
    elif atlas == Atlas.KING:
        atlas_data = fetch_king_2019(data=_data, **kwargs)
    elif atlas == Atlas.XUE:
        atlas_data = fetch_xue_2021(**kwargs)
    else:
        raise ValueError(f"Unknown atlas name: {atlas}.")

    return atlas_data


def fetch_fs_dkt_cmap_lut_file():

    return get_local_atlas_cmap_lut_fname(FsAtlasCmaps.DKT)


# ToDo
# Check if these should/can be retrieved with datalad
def fetch_suit_cmap_lut_files(
    atlas_dirname, data_dir=None, resume=True, verbose=1
):
    # ToDo
    # This assumes we have Internet access, which may not be the case on a
    # compute node. Think what is the best strategy. We could have done the
    # same for the FS atlas

    base_url = urljoin(
        suit_atlas_gh_url_base + fwd_slash_separator,
        suit_atlas_gh_revision + fwd_slash_separator + atlas_dirname.value,
    )

    # get maps from "atlas_description.json"
    url = urljoin(base_url + fwd_slash_separator, "atlas_description.json")
    resp = requests.get(url)
    data_dict = json.loads(resp.text)

    # get map names and description
    maps = data_dict["Maps"]
    types = data_dict["Type"]

    suffixes = [
        build_suffix(LutFileExtension.LUT),
        build_suffix(DelimitedValuesFileExtension.TSV),
    ]

    # Get filenames from maps
    maps_full = []
    for _map, _type in zip(maps, types):
        # Get only the dseg-type files
        if _type == "dseg":
            for suffix in suffixes:
                maps_full.append(f"{_map}{suffix}")

    files = []
    for f in maps_full:
        files.append((f, base_url + "/" + f, {}))

    # Upon inspection of the SUITPy code, dataset names have an all lowercase
    # name matching the URL
    dataset_name = atlas_dirname.value.lower()
    data_dir = _get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )

    # Get local fullpath(s) of downloaded file(s)
    fnames = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return [pathlib.Path(val) for val in fnames]


def read_lut_from_tsv(fname):
    df = pd.read_csv(fname, sep="\t")

    # ToDo
    # Add alpha

    # Convert HEX values to the [0,1] range used by matplotlib colors
    import matplotlib.colors as mcolors

    df["color"] = df["color"].apply(lambda x: mcolors.to_rgb(x))

    return df.set_index("index").agg(tuple, axis=1).to_dict()


def read_lut_from_lut(fname):

    lut = dict()
    with open(fname) as f:
        for line in f:
            line = line.strip()

            (key, r, g, b, name) = line.split()
            # ToDo
            # Add alpha and convert to 255 ?
            lut[int(key)] = (name, (float(r), float(g), float(b)))

    return lut


def _filter_suit_cmap_fnames(fnames, lut_extension, **kwargs):

    buckner_version = None
    if buckner_version_arg in kwargs:
        buckner_version = kwargs.pop(buckner_version_arg)

    xue_version = None
    if xue_version_arg in kwargs:
        xue_version = kwargs.pop(xue_version_arg)

    lut_fname = list(
        compress(
            fnames,
            [
                val.match(asterisk + build_suffix(lut_extension))
                for val in fnames
            ],
        )
    )

    if buckner_version:
        lut_fname = list(
            compress(
                lut_fname,
                [
                    val.match(asterisk + buckner_version.value + asterisk)
                    for val in lut_fname
                ],
            )
        )

    if xue_version:
        lut_fname = list(
            compress(
                lut_fname,
                [
                    val.match(asterisk + xue_version.value + asterisk)
                    for val in lut_fname
                ],
            )
        )

    assert len(lut_fname) == 1

    return lut_fname[0]


def fetch_atlas_cmap_lut_file(atlas, **kwargs):

    suit_file_fetch_kwargs = dict({})
    if suit_file_fetch_arg in kwargs:
        suit_file_fetch_kwargs = kwargs.pop(suit_file_fetch_arg)

    suit_atlas_version_kwargs = dict({})
    if suit_atlas_version_arg in kwargs:
        suit_atlas_version_kwargs = kwargs.pop(suit_atlas_version_arg)

    lut_extension = LutFileExtension.LUT
    if lut_extension_arg in kwargs:
        lut_extension = kwargs.pop(lut_extension_arg)

    if atlas == Atlas.DKT:
        atlas_cmap_lut_file = fetch_fs_dkt_cmap_lut_file()
    elif atlas == Atlas.BUCKNER:
        atlas_cmap_files = fetch_suit_cmap_lut_files(
            SuitAtlasDirnameBase.BUCKNER, **suit_file_fetch_kwargs
        )
        atlas_cmap_lut_file = _filter_suit_cmap_fnames(
            atlas_cmap_files, lut_extension, **suit_atlas_version_kwargs
        )
    elif atlas == Atlas.DIEDRICHSEN:
        atlas_cmap_files = fetch_suit_cmap_lut_files(
            SuitAtlasDirnameBase.DIEDRICHSEN, **suit_file_fetch_kwargs
        )
        atlas_cmap_lut_file = _filter_suit_cmap_fnames(
            atlas_cmap_files, lut_extension
        )
    elif atlas == Atlas.JI:
        atlas_cmap_files = fetch_suit_cmap_lut_files(
            SuitAtlasDirnameBase.JI, **suit_file_fetch_kwargs
        )
        atlas_cmap_lut_file = _filter_suit_cmap_fnames(
            atlas_cmap_files, lut_extension
        )
    elif atlas == Atlas.KING:
        atlas_cmap_files = fetch_suit_cmap_lut_files(
            SuitAtlasDirnameBase.KING, **suit_file_fetch_kwargs
        )
        atlas_cmap_lut_file = _filter_suit_cmap_fnames(
            atlas_cmap_files, lut_extension
        )
    elif atlas == Atlas.XUE:
        atlas_cmap_files = fetch_suit_cmap_lut_files(
            SuitAtlasDirnameBase.XUE, **suit_file_fetch_kwargs
        )
        atlas_cmap_lut_file = _filter_suit_cmap_fnames(
            atlas_cmap_files, lut_extension, **suit_atlas_version_kwargs
        )
    else:
        raise NotImplementedError(
            f"Unsupported atlas:\nFound: {atlas}; {list(Atlas.__members__)}"
        )

    return atlas_cmap_lut_file


def read_lut_data(fname):

    ext = pathlib.Path(fname).suffixes[0][1:]

    if ext == TextFileExtension.TXT.value:
        lut = read_lut_from_txt(fname)
    elif ext == LutFileExtension.LUT.value:
        lut = read_lut_from_lut(fname)
    elif ext == DelimitedValuesFileExtension.TSV.value:
        lut = read_lut_from_tsv(fname)
    else:
        raise NotImplementedError(f"{ext} extension reading not implemented.")

    return lut


def get_atlas_cmap(atlas, **kwargs):
    """Use the LUT files as their values are already within the [0,1] range used
    by matplotlib colors."""

    fname = fetch_atlas_cmap_lut_file(atlas, **kwargs)

    lut = read_lut_data(fname)

    return get_cmap_from_lut(lut)


def rescale_lut(lut):
    _vals = list(
        zip(
            *[
                list(map(lambda x: x[0], lut.values())),
                rescale_float_colors([val[1] for val in lut.values()]),
            ]
        )
    )
    return dict(zip(lut.keys(), _vals))


def add_alpha_to_lut(lut, alpha):
    _vals = list(
        zip(
            *[
                list(map(lambda x: x[0], lut.values())),
                [tuple([*val[1], alpha]) for val in lut.values()],
            ]
        )
    )
    return dict(zip(lut.keys(), _vals))


def lut2df(lut):
    from itertools import chain

    # Unzip the dict values to build the dataframe
    _lut = dict(
        {key: list([val[0], *(chain(val[1]))]) for key, val in lut.items()}
    )

    columns = ["LabelName", "R", "G", "B", "A"]
    df = pd.DataFrame.from_dict(_lut, orient="index", columns=columns)
    df.index.name = "ID"

    return df
