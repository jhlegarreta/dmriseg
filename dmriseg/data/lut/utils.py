# -*- coding: utf-8 -*-

import enum
import json
import pathlib
from itertools import compress
from urllib.parse import urljoin

import importlib_resources
import pandas as pd
import requests
from SUITPy.atlas import fetch_atlas
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

class_id_label = "ID"
class_name_label = "LabelName"
r_label = "R"
g_label = "G"
b_label = "B"
a_label = "A"


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


class SuitAtlasDiedrichsenGroups(enum.Enum):
    ALL = "all"
    CRUS = "crus"
    DCN = "dcn"
    DENTATE = "dentate"
    FASTIGIAL = "fastigial"
    INTERPOSED = "interposed"
    LH = "lh"
    LOBULES = "lobules"
    VERMIS = "vermis"
    RH = "rh"
    ANTERIOR_LOBE = "anterior_lobe"
    POSTERIOR_LOBE = "posterior_lobe"


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


def apply_fastsurfer_cmap(img_data):

    from skimage import color

    # ToDo
    # If not all labels are present in the img_data, it may happen that for the
    # next call, colors start where the cycle stopped for the previous call
    return color.label2rgb(img_data, bg_label=0)


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
        atlas_data = fetch_atlas("Buckner_2011", **kwargs)
    elif atlas == Atlas.DIEDRICHSEN:
        atlas_data = fetch_atlas("Diedrichsen_2009", **kwargs)
    elif atlas == Atlas.JI:
        atlas_data = fetch_atlas("Ji_2019", **kwargs)
    elif atlas == Atlas.KING:
        atlas_data = fetch_atlas("King_2019", data=_data, **kwargs)
    elif atlas == Atlas.XUE:
        atlas_data = fetch_atlas("Xue_2021", **kwargs)
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
        dataset_name, atlas_dir=data_dir, verbose=verbose
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


def read_lut_from_tsv2(fname):
    df = pd.read_csv(fname, sep="\t")

    # ToDo
    # Add alpha ??
    return df.set_index("ID")[["R", "G", "B"]].apply(tuple, axis=1).to_dict()


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
            f"Unsupported atlas:\nFound: {atlas}\n"
            f"Available: {list(Atlas.__members__)}"
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


def add_additional_label_to_lut(lut, label, name, color):
    inters_label = list(set(label).intersection(set(lut.keys())))

    if inters_label:
        raise ValueError(f"Labels exist in current LUT: {inters_label}")
    else:
        [
            lut.update(dict({_label: (_name, _color)}))
            for _label, _name, _color in zip(label, name, color)
        ]

    return dict(sorted(lut.items()))


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

    columns = [class_name_label, r_label, g_label, b_label, a_label]
    df = pd.DataFrame.from_dict(_lut, orient="index", columns=columns)
    df.index.name = class_id_label

    return df


def map_diedrichsen2cerebnet():
    diedrichsen2cerebnet_map = dict(
        {
            "Unknown": "Unknown",
            "Left_I_IV": "Left_I_IV",
            "Right_I_IV": "Right_I_IV",
            "Left_V": "Left_V",
            "Right_V": "Right_V",
            "Left_VI": "Left_VI",
            "Vermis_VI": "Vermis_VI",
            "Right_VI": "Right_VI",
            "Left_CrusI": "Left_CrusI",
            "Vermis_CrusI": "Vermis_VII",  # Deduced reading Diedrichsen 2009
            "Right_CrusI": "Right_CrusI",
            "Left_CrusII": "Left_CrusII",
            "Vermis_CrusII": "Vermis_VII",  # Deduced reading Diedrichsen 2009
            "Right_CrusII": "Right_CrusII",
            "Left_VIIb": "Left_VIIb",
            "Vermis_VIIb": "Vermis_VII",
            "Right_VIIb": "Right_VIIb",
            "Left_VIIIa": "Left_VIIIa",
            "Vermis_VIIIa": "Vermis_VIII",
            "Right_VIIIa": "Right_VIIIa",
            "Left_VIIIb": "Left_VIIIb",
            "Vermis_VIIIb": "Vermis_VIII",
            "Right_VIIIb": "Right_VIIIb",
            "Left_IX": "Left_IX",
            "Vermis_IX": "Vermis_IX",
            "Right_IX": "Right_IX",
            "Left_X": "Left_X",
            "Vermis_X": "Vermis_X",
            "Right_X": "Right_X",
            "Left_Dentate": "Left_Corpus_Medullare",
            "Right_Dentate": "Right_Corpus_Medullare",
            "Left_Interposed": "Left_Corpus_Medullare",
            "Right_Interposed": "Right_Corpus_Medullare",
            "Left_Fastigial": "Left_Corpus_Medullare",
            "Right_Fastigial": "Right_Corpus_Medullare",
        }
    )

    assert len(diedrichsen2cerebnet_map.keys()) == 34 + 1
    assert len(set(diedrichsen2cerebnet_map.values())) == 28


# ToDo
# Improve this: download or store the LUT and read the data from there
# identifying the groups
def get_diedrichsen_group_labels(group_name):

    if group_name == SuitAtlasDiedrichsenGroups.DCN.value:
        return [29, 30, 31, 32, 33, 34]
    elif group_name == SuitAtlasDiedrichsenGroups.DENTATE.value:
        return [29, 30]
    elif group_name == SuitAtlasDiedrichsenGroups.INTERPOSED.value:
        return [31, 32]
    elif group_name == SuitAtlasDiedrichsenGroups.FASTIGIAL.value:
        return [33, 34]
    elif group_name == SuitAtlasDiedrichsenGroups.VERMIS.value:
        return [6, 9, 12, 15, 18, 21, 24, 27]
    elif group_name == SuitAtlasDiedrichsenGroups.LOBULES.value:
        return [1, 2, 3, 4, 5, 7, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28]
    elif group_name == SuitAtlasDiedrichsenGroups.CRUS.value:
        return [8, 10, 11, 13]
    elif group_name == SuitAtlasDiedrichsenGroups.ANTERIOR_LOBE.value:
        return [1, 2, 3, 4]
    elif group_name == SuitAtlasDiedrichsenGroups.POSTERIOR_LOBE.value:
        return [5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25]
    elif group_name == SuitAtlasDiedrichsenGroups.ALL.value:
        return list(range(1, 35))
    else:
        raise NotImplementedError(f"{group_name} not implemented")


def rename_suit_atlas_diedrichsen_groups_plot_labels(group_name):

    if group_name == SuitAtlasDiedrichsenGroups.ALL.value:
        return "All"
    elif group_name == SuitAtlasDiedrichsenGroups.CRUS.value:
        return "Crus"
    elif group_name == SuitAtlasDiedrichsenGroups.DCN.value:
        return "DCN"
    elif group_name == SuitAtlasDiedrichsenGroups.DENTATE.value:
        return "Dentate"
    elif group_name == SuitAtlasDiedrichsenGroups.FASTIGIAL.value:
        return "Fastigial"
    elif group_name == SuitAtlasDiedrichsenGroups.INTERPOSED.value:
        return "Interposed"
    elif group_name == SuitAtlasDiedrichsenGroups.LH.value:
        return "Lh"
    elif group_name == SuitAtlasDiedrichsenGroups.LOBULES.value:
        return "Lobules"
    elif group_name == SuitAtlasDiedrichsenGroups.VERMIS.value:
        return "Vermis"
    elif group_name == SuitAtlasDiedrichsenGroups.RH.value:
        return "Rh"
    elif group_name == SuitAtlasDiedrichsenGroups.ANTERIOR_LOBE.value:
        return "anterior lobe"
    elif group_name == SuitAtlasDiedrichsenGroups.POSTERIOR_LOBE.value:
        return "posterior lobe"
    else:
        raise NotImplementedError(f"Group not recognized: {group_name}")
