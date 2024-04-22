#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dmriseg.data.lut.utils import (
    a_label,
    b_label,
    class_name_label,
    g_label,
    r_label,
)
from dmriseg.dataset.utils import boxplot_channel_metric, rescale_int_colors


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_performance_fname",
        help="Performance filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "in_labels_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_fname",
        help="Plot filename (*.png)",
        type=Path,
    )
    parser.add_argument(
        "metric_name",
        help="Metric name",
        type=str,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    sep = "\t"
    color_labels = [r_label, g_label, b_label, a_label]

    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    label_names = df_lut[class_name_label].values[1:]
    _label_colors = df_lut[color_labels].values[1:]
    _label_colors[:, -1] = 255
    label_colors = np.asarray(rescale_int_colors(_label_colors))

    df_metric = pd.read_csv(args.in_performance_fname, sep=sep, index_col=0)
    metric_values = df_metric.values
    # The below was added to see that the script was working as expected
    # import numpy as np
    # np.random.seed(0)
    # rng = np.random.default_rng()
    # metric_values = rng.normal(size=(10, 34))

    assert len(label_names) == metric_values.shape[1]

    # ToDo
    # Rearrange to have all left/right next to each other

    grid = True
    # label_colors = "viridis"
    _fig = boxplot_channel_metric(
        metric_values,
        args.metric_name,
        label_names,
        cmap=label_colors,
        grid=grid,
    )
    _fig.savefig(args.out_fname)
    plt.close(_fig)


if __name__ == "__main__":
    main()
