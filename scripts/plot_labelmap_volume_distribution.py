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

from dmriseg.data.lut.utils import class_id_label as lut_class_id_label
from dmriseg.data.lut.utils import class_name_label, read_lut_from_tsv2


def plot_distribution(df, label_mapping, palette="viridis"):

    # Rename labels
    df.rename(columns=label_mapping, inplace=True)

    # Plot the distribution of each column on a single plot
    fig, ax = plt.subplots()
    common_norm = False
    sns.kdeplot(
        data=df,
        common_norm=common_norm,
        palette=palette,
        linewidth=1,
        legend=True,
    )

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
        help="Labelmap volumde data filename (*.tsv)",
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
        "out_filename",
        help="Output filename (*.tsv)",
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
    # Normalize colors and remove the background
    normalized_colors = list(normalize_colors(lut).values())[1:]

    fig = plot_distribution(df, label_mapping, palette=normalized_colors)
    fig.savefig(args.out_filename)


if __name__ == "__main__":
    main()
