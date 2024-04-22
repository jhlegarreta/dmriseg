#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

from dmriseg.data.lut.utils import class_id_label as lut_class_id_label
from dmriseg.data.lut.utils import class_name_label as lut_class_name_label
from dmriseg.utils.stat_preparation_utils import (
    class_id_label,
    contrast_label,
    create_df,
    create_pairs,
    pvalue_label,
)


def get_log_ax(orient="v"):
    if orient == "v":
        figsize = (12, 6)
        set_scale = "set_yscale"
    else:
        figsize = (10, 8)
        set_scale = "set_xscale"
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_alpha(1)
    getattr(ax, set_scale)
    return ax


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

    contrast_palette = sns.color_palette("YlGnBu", n_colors=2)

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
        significance_values = df_significance.loc[pvalue_label].values

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
        pairs = create_pairs(args.ref_contrast_name, contrast_name, labels)

        data = create_df(
            df_metric_ref,
            df_metric,
            args.ref_contrast_name,
            contrast_name,
            args.metric_name,
        )

        # Create new plot
        ax = get_log_ax()

        hue_plot_params = {
            "data": data,
            "x": class_id_label,
            "y": args.metric_name,
            "order": list(map(str, labels)),
            "hue": contrast_label,
            "hue_order": contrast_order,
            "palette": contrast_palette,
        }

        # Plot with seaborn
        ax = sns.boxplot(ax=ax, **hue_plot_params)

        # Add annotations
        annotator = Annotator(ax, pairs, **hue_plot_params)
        # parameters = dict({
        #    "line_height": 2,
        #    "line_offset": 2,
        #    "text_offset": 2})
        # annotator.configure(**parameters)
        annotator.annotate_custom_annotations(
            list(map(str, significance_values))
        )

        ax.set_ylabel(args.metric_name)
        # May make all lines and annotations overlap. Will need to see how it
        # plays when I have real data
        # ax.set_ylim([0, 1.1])
        ax.set_xlabel("Labels", labelpad=20)
        ax.set_xticklabels(label_names, ha="right", rotation=45)
        ax.legend(loc=(1.05, 0.5))
        plt.tight_layout()
        plt.show()
        plt.savefig(out_fname)


if __name__ == "__main__":
    main()
