#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot labelmap volume distribution.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dmriseg.data.lut.utils import SuitAtlasDiedrichsenGroups
from dmriseg.data.lut.utils import class_id_label as lut_class_id_label
from dmriseg.data.lut.utils import (
    class_name_label,
    get_diedrichsen_group_labels,
    read_lut_from_tsv2,
)
from dmriseg.io.file_extensions import FigureFileExtension
from dmriseg.io.utils import build_suffix, group_fname_label, underscore

volume_distribution_fnamal_label = "labelmaps_volumes_distribution"


def legend_properties(group):

    _labels = get_diedrichsen_group_labels(group)

    if group == SuitAtlasDiedrichsenGroups.ALL.value:
        return {"bbox_to_anchor": (0.5, -0.425), "ncols": 7}
    if group == SuitAtlasDiedrichsenGroups.DCN.value:
        return {"bbox_to_anchor": (0.5, -0.175), "ncols": len(_labels)}
    elif group == SuitAtlasDiedrichsenGroups.DENTATE.value:
        return {"bbox_to_anchor": (0.5, -0.175), "ncols": len(_labels)}
    elif group == SuitAtlasDiedrichsenGroups.INTERPOSED.value:
        return {"bbox_to_anchor": (0.5, -0.175), "ncols": len(_labels)}
    elif group == SuitAtlasDiedrichsenGroups.FASTIGIAL.value:
        return {"bbox_to_anchor": (0.5, -0.175), "ncols": len(_labels)}
    elif group == SuitAtlasDiedrichsenGroups.VERMIS.value:
        return {"bbox_to_anchor": (0.5, -0.175), "ncols": len(_labels)}
    elif group == SuitAtlasDiedrichsenGroups.LOBULES.value:
        return {"bbox_to_anchor": (0.5, -0.225), "ncols": 8}
    elif group == SuitAtlasDiedrichsenGroups.CRUS.value:
        return {"bbox_to_anchor": (0.5, -0.175), "ncols": len(_labels)}
    else:
        raise ValueError(f"Unknown group name: {group}")


def plot_distribution(df, label_mapping, group, palette="viridis"):

    # Rename labels
    _df = df.rename(columns=label_mapping, inplace=False)

    # Plot the distribution of each column on a single plot
    fig, ax = plt.subplots(figsize=(12, 7))
    common_norm = False
    # ax = sns.kdeplot(
    #     data=_df,
    #     common_norm=common_norm,
    #     palette=palette,
    #     linewidth=1,
    #     legend=True,
    # )

    # The above does not create legend handles and labels, so plot the
    # distribution of each column on a single plot
    for i, column in enumerate(_df.columns):
        sns.kdeplot(
            _df[column],
            common_norm=common_norm,
            color=palette[i],
            linewidth=1,
            label=column,
        )

    # (handles, labels) = ax.get_legend_handles_labels()
    # Add legend with custom placement
    legend_props = legend_properties(group)
    plt.legend(loc="lower center", **legend_props)

    plt.xlabel("volume ($mm^3$)")
    plt.ylabel("frequency")
    plt.title("Volume distribution")
    plt.tight_layout()

    return fig


def normalize_colors(lut):

    # Need colors to be in [0,1] for VTK
    cmap = {key: tuple(np.array(values) / 255) for key, values in lut.items()}

    return cmap


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_labelmap_volume_fname",
        help="Labelmap volume data filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "in_participants_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "in_labels_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_dirname",
        help="Output dirname",
        type=Path,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    sep = "\t"

    df = pd.read_csv(args.in_labelmap_volume_fname, sep=sep, index_col=0)

    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    # Remove the background
    # labels = df_lut[lut_class_id_label].values[1:]

    keys_column = lut_class_id_label
    values_column = class_name_label
    # Make label IDs be strings
    label_mapping = {
        str(key): value
        for key, value in df_lut.set_index(keys_column)[values_column]
        .to_dict()
        .items()
    }

    lut = read_lut_from_tsv2(args.in_labels_fname)

    # Plot them in groups so that dissimilar volumes do not hinder gaining
    # insight from the plot
    group_names = [
        SuitAtlasDiedrichsenGroups.ALL,
        SuitAtlasDiedrichsenGroups.DCN,
        SuitAtlasDiedrichsenGroups.DENTATE,
        SuitAtlasDiedrichsenGroups.INTERPOSED,
        SuitAtlasDiedrichsenGroups.FASTIGIAL,
        SuitAtlasDiedrichsenGroups.VERMIS,
        SuitAtlasDiedrichsenGroups.LOBULES,
        SuitAtlasDiedrichsenGroups.CRUS,
    ]

    suffix = build_suffix(FigureFileExtension.SVG)

    for group in group_names:
        # Keep only the labels corresponding to the group
        _labels = get_diedrichsen_group_labels(group.value)
        # Removes the background as well
        _lut = {k: v for k, v in lut.items() if k in _labels}
        df_group = df[list(map(str, _labels))]

        # Normalize colors
        normalized_colors = list(normalize_colors(_lut).values())

        fig = plot_distribution(
            df_group, label_mapping, group.value, palette=normalized_colors
        )

        file_basename = (
            volume_distribution_fnamal_label
            + underscore
            + group_fname_label
            + underscore
            + group.value
            + suffix
        )
        fig.savefig(args.out_dirname / file_basename)


if __name__ == "__main__":
    main()
