#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from dmriseg.visualization.plot_utils import plot_curves


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "out_loss_plot_fname",
        help="Output loss plot filename (*.png)",
        type=Path,
    )
    parser.add_argument(
        "out_metric_plot_fname",
        help="Output metric plot filename (*.png)",
        type=Path,
    )
    parser.add_argument(
        "--model_metadata_fnames",
        nargs="+",
        help="Model metadata filenames (*.pkl)",
        type=Path,
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        help="Model names",
        type=str,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # ToDo
    # Save the model name, metric name and loss name to the metadata file

    # Load/init meta data dict
    meta_data = []
    for fname in args.model_metadata_fnames:
        with open(fname, "rb") as f:
            meta_data.append(pickle.load(f))

    # Plot the loss values
    loss_values = np.asarray([item["loss_values"] for item in meta_data])
    epochs = np.asarray([len(item["loss_values"]) for item in meta_data])

    assert [epoch == epochs[0] for epoch in epochs]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"][: len(args.model_metadata_fnames)]

    # Plot loss values
    names = args.model_names
    xlabel = "Epoch"
    ylabel = "Loss"
    fig = plot_curves(loss_values, colors, names, xlabel, ylabel)
    fig.savefig(args.out_loss_plot_fname)
    plt.close(fig)

    # Plot metric values
    # best_metric = np.asarray([item["best_metric"] for item in meta_data])
    metric_values = np.asarray([item["metric_values"] for item in meta_data])

    names = args.model_names
    ylabel = "Metric"
    fig = plot_curves(metric_values, colors, names, xlabel, ylabel)
    fig.savefig(args.out_metric_plot_fname)
    plt.close(fig)


if __name__ == "__main__":
    main()
