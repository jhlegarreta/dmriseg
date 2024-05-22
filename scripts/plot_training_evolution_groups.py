#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot training evolution: plot loss and metric across epochs grouped by name: for
each group plot their mean values as a solid curve and their standard deviation
values as a shaded strip.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from dmriseg.visualization.plot_utils import plot_shaded_strip


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

    # prop_cycle = plt.rcParams["axes.prop_cycle"]
    # colors = prop_cycle.by_key()["color"][: len(args.model_metadata_fnames)]

    # Identify the groups
    names = args.model_names
    unique_names = sorted(set(names))
    groupd_idx = []
    for name in unique_names:
        idx = [_idx for _idx, value in enumerate(names) if value == name]
        groupd_idx.append(idx)

    # Compute the mean and std dev within each group
    mean_loss = []
    stddev_loss = []
    for _idx in groupd_idx:
        _mean_loss = np.mean(loss_values[_idx], axis=0)
        _stddev_loss = np.std(loss_values[_idx], axis=0)

        mean_loss.append(_mean_loss)
        stddev_loss.append(_stddev_loss)

    # Plot loss values
    xlabel = "Epoch"
    ylabel_training = "Loss"
    ylabel_metric = "Metric"

    fig = plot_shaded_strip(
        mean_loss, stddev_loss, unique_names, xlabel, ylabel_training
    )
    fig.savefig(args.out_loss_plot_fname)
    plt.close(fig)

    # Plot metric values
    # best_metric = np.asarray([item["best_metric"] for item in meta_data])
    metric_values = np.asarray([item["metric_values"] for item in meta_data])

    # Compute the mean and std dev within each group
    mean_metric = []
    stddev_metric = []
    for _idx in groupd_idx:
        _mean_metric = np.mean(metric_values[_idx], axis=0)
        _stddev_metric = np.std(metric_values[_idx], axis=0)

        mean_metric.append(_mean_metric)
        stddev_metric.append(_stddev_metric)

    fig = plot_shaded_strip(
        mean_metric, stddev_metric, unique_names, xlabel, ylabel_metric
    )
    fig.savefig(args.out_metric_plot_fname)
    plt.close(fig)


if __name__ == "__main__":
    main()
