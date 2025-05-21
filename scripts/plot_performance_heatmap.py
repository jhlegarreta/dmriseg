#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot mean performance values as a heatmap. Mark statistically significant
values with an ``X``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dmriseg.analysis.measures import rename_measure_names_plot_labels
from dmriseg.data.lut.utils import SuitAtlasDiedrichsenGroups
from dmriseg.utils.contrast_utils import rename_contrasts_plot_labels

mean_label = "mean"
stat_contrast_a_label = "A"
stat_contrast_b_label = "B"
stat_reject_label = "reject"


def plot_heatmap(mean_df, significance_mask, measure_name):

    fig = plt.figure(figsize=(10, 6))
    ax = sns.heatmap(mean_df, cmap="viridis", annot=False, cbar=True)

    # Overlay crosses for significant results
    for i, row_label in enumerate(mean_df.index):
        for j, col_label in enumerate(mean_df.columns):
            if significance_mask.loc[row_label, col_label]:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "X",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

    plt.title(measure_name)
    plt.ylabel("Contrast")
    plt.xlabel("Label set")
    plt.tight_layout()

    return fig


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_significance_filename",
        help="Input statistical significance filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_filename",
        help="Output filename (*.png)",
        type=Path,
    )
    parser.add_argument(
        "measure_name",
        help="Measure name (e.g. dice)",
        type=str,
    )
    parser.add_argument(
        "--in_performance_filenames",
        help="Performance filenames (*.tsv)",
        type=Path,
        nargs="+",
    )
    parser.add_argument(
        "--contrast_names",
        help="Contrast names (e.g. t1, b0)",
        type=str,
        nargs="+",
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Load mean values from describe TSVs
    mean_vals = {}
    file_labels = []

    assert len(args.in_performance_filenames) == len(args.contrast_names)

    sep = "\t"
    index_col = 0
    for contrast_name, fname in zip(
        args.contrast_names, args.in_performance_filenames
    ):
        df = pd.read_csv(fname, sep=sep, index_col=index_col)
        if mean_label not in df.index:
            raise ValueError(f"'{mean_label}' not found in {fname}")
        mean_vals[contrast_name] = df.loc[mean_label]
        file_labels.append(contrast_name)

    # Create dataframe: rows: contrast names; columns: label groups
    mean_df = pd.DataFrame.from_dict(mean_vals, orient="index")

    # Create significance mask for heatmap
    significance_mask = pd.DataFrame(
        False, index=mean_df.index, columns=mean_df.columns
    )

    # Load significance (reject) information
    df = pd.read_csv(args.in_significance_filename, sep=sep, index_col=[0])

    # Find the reference contrast
    ref_contrast_name = set(df[stat_contrast_a_label]).intersection(
        set(df[stat_contrast_b_label])
    )
    assert len(ref_contrast_name) == 1
    ref_contrast_name = ref_contrast_name.pop()

    for _, row in df.iterrows():
        a, b, reject = (
            row[stat_contrast_a_label],
            row[stat_contrast_b_label],
            row[stat_reject_label],
        )
        # If not significant, continue
        if not reject:
            continue
        # Set the appropriate cell in the significance mask to True
        contrast_name = [
            value for value in (a, b) if value != ref_contrast_name
        ]
        assert len(contrast_name) == 1
        contrast_name = contrast_name[0]
        significance_mask.loc[
            contrast_name, SuitAtlasDiedrichsenGroups.ALL.value
        ] = True

    # Rename contrasts to match the naming of the manuscript
    mapping = {
        contrast_name: rename_contrasts_plot_labels(contrast_name)
        for contrast_name in args.contrast_names
    }
    mean_df = mean_df.rename(index=mapping)
    significance_mask = significance_mask.rename(index=mapping)

    measure_label = rename_measure_names_plot_labels(args.measure_name)

    fig = plot_heatmap(mean_df, significance_mask, measure_label)
    fig.savefig(args.out_filename)


if __name__ == "__main__":
    main()
