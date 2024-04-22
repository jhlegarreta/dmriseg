#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot statistical significance values in boxplots between a reference contrast
method and each given contrast. Each pair is plot into a separate figure.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dmriseg.data.lut.utils import class_id_label as lut_class_id_label
from dmriseg.data.lut.utils import class_name_label as lut_class_name_label
from dmriseg.utils.stat_preparation_utils import (
    class_id_label,
    contrast_label,
    create_df,
    create_pairs,
    pvalue_label,
)
from dmriseg.visualization.stat_annot_utils import add_stat_annotation


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_labels_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "metric_name",
        help="Metric name",
        type=str,
    )
    parser.add_argument(
        "ref_contrast_name",
        help="Reference contrast name",
        type=str,
    )
    parser.add_argument(
        "in_performance_fname_ref",
        type=Path,
        help="Reference performance filename (*.tsv).",
    )
    parser.add_argument(
        "--in_performance_fnames",
        nargs="+",
        type=Path,
        help="Performance filenames (*.tsv).",
    )
    parser.add_argument(
        "--in_significance_fnames",
        nargs="+",
        type=Path,
        help="Significance filenames (*.tsv).",
    )
    parser.add_argument(
        "--out_fnames", nargs="+", type=Path, help="Output filenames (*.png)."
    )
    # parser.add_argument("split_anatomically", help="Split anatomically (hemispheric lobules L/R, vermis, nuclei L/R)", type=bool)
    parser.add_argument(
        "--contrast_names", nargs="+", type=str, help="Contrast names."
    )
    parser.add_argument(
        "--cmap_name", type=str, help="Contrast names.", default="YlGnBu"
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    assert (
        len(args.in_performance_fnames)
        == len(args.in_significance_fnames)
        == len(args.out_fnames)
        == len(args.contrast_names)
    )

    # ToDo
    # Rearrange to have all left/right next to each other

    sep = "\t"

    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    labels = df_lut[lut_class_id_label].values[1:]
    label_names = df_lut[lut_class_name_label].values[1:]

    index_col = 0
    df_metric_ref = pd.read_csv(
        args.in_performance_fname_ref, sep=sep, index_col=index_col
    )

    # ToDo
    # For now, we will only plot pair-wise comparisons. Eventually, we may want
    # To plot triplet, etc. comparisons (e.g. sphm vs b0, T1, T2, etc.). In
    # order to create plots that can actually be read, this would also involve
    # creating separate plots for nuclei L/R, lobules L/R, and vermis.
    n_colors = 2
    # ToDo
    # If no cmap_name is given, then use the the one contained in the LUT
    if args.cmap_name:
        contrast_palette = sns.color_palette("YlGnBu", n_colors=n_colors)
    else:
        raise NotImplementedError(
            "Non-string custom colormap not implemented."
        )

    # Loop over the performance/significance filenames
    for performance_fname, significance_fname, out_fname, contrast_name in zip(
        args.in_performance_fnames,
        args.in_significance_fnames,
        args.out_fnames,
        args.contrast_names,
    ):

        df_metric = pd.read_csv(
            performance_fname, sep=sep, index_col=index_col
        )

        df_significance = pd.read_csv(
            significance_fname, sep=sep, index_col=index_col
        )
        pvalues = df_significance.loc[pvalue_label].values

        contrast_order = [args.ref_contrast_name, contrast_name]

        # ToDo
        # Create pairs depending on the pvalues: set a threshold e.g.
        # p-value annotation legend:
        #    ns: 5.00e-02 < p <= 1.00e+00
        #     *: 1.00e-02 < p <= 5.00e-02
        #    **: 1.00e-03 < p <= 1.00e-02
        #   ***: 1.00e-04 < p <= 1.00e-03
        #  ****: p <= 1.00e-04
        # and show only those that we deem relevant.
        # For now, show all
        box_pairs = create_pairs(args.ref_contrast_name, contrast_name, labels)

        data = create_df(
            df_metric_ref,
            df_metric,
            args.ref_contrast_name,
            contrast_name,
            args.metric_name,
        )

        # Plot with seaborn
        x = class_id_label
        y = args.metric_name
        hue = contrast_label
        hue_order = contrast_order
        perform_stat_test = False
        test_short_name = "mytest"
        order = list(map(str, labels))
        ax = sns.boxplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            palette=contrast_palette,
        )
        add_stat_annotation(
            ax,
            data=data,
            x=x,
            y=y,
            hue=hue,
            box_pairs=box_pairs,
            order=order,
            hue_order=hue_order,
            perform_stat_test=perform_stat_test,
            pvalues=pvalues,
            test_short_name=test_short_name,
            loc="inside",
            verbose=2,
        )
        plt.legend(loc="upper left", bbox_to_anchor=(1.03, 1))

        ymax = 1.1
        ax.set_ylim([0, ymax])
        ax.set_xlabel("Labels")
        ax.set_xticklabels(label_names, ha="right", rotation=45)
        plt.tight_layout()
        plt.show()
        plt.savefig(out_fname)


if __name__ == "__main__":
    main()
