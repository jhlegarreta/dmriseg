# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def get_label_cmap(n_labels):
    """Get matplotlib colour map for a label map."""
    unique_values = np.arange(n_labels)
    colors = plt.cm.turbo(np.linspace(0, 1, len(unique_values)))
    cmap = ListedColormap(colors)
    # Create bin edges by adding half a step to unique values
    bin_edges = np.concatenate(
        [unique_values - 0.5, [unique_values[-1] + 0.5]]
    )
    # Create a BoundaryNorm to map unique values to indices in the colormap
    norm = BoundaryNorm(bin_edges, cmap.N, clip=True)

    return cmap, norm


def barplot_dice(ds, loader, figsize=None):
    """Bar plot dice scores."""
    if figsize is None:
        figsize = [18, 5]

    _, ax = plt.subplots(figsize=figsize)
    idx = [i for i in range(len(ds))]
    ax.bar(idx, ds)
    ax.set_xticks(idx)
    ax.set_xticklabels(idx)
    ax.set_ylim([0, 1])
    plt.grid()
    plt.show()
    print(
        f"avg(Dice)={np.asarray(ds).mean():0.2f} with K={len(ds)} and N={len(loader)}"
    )


def plot_loss_and_metric(axs, loss_values, metric_values, validation_epoch):
    """Plots training loss and metric"""
    x = [i + 1 for i in range(len(loss_values))]
    y = loss_values
    axs[2, 0].plot(x, y, ".", markersize=1)
    axs[2, 0].set_title("Loss")
    axs[2, 0].set_xlabel("epoch")
    x = [validation_epoch * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    axs[2, 1].plot(x, y, ".", markersize=1)
    axs[2, 1].set_title("Metric")
    axs[2, 1].set_xlabel("epoch")


def boxplot_channel_metric(
    metric_values, metric_name, class_names, cmap=None, title=None, grid=False
):
    assert metric_values.shape[1] == len(class_names)

    figsize = (15, 10)
    fig, ax = plt.subplots(figsize=figsize)
    bplot = ax.boxplot(metric_values, patch_artist=True)
    # Set the face colors
    if cmap is not None:
        if isinstance(cmap, np.ndarray):
            for patch, color in zip(bplot["boxes"], cmap):
                patch.set_facecolor(color)
        elif isinstance(cmap, str):
            cm = plt.cm.get_cmap(cmap)
            class_count = len(class_names)
            colors = [cm(val / class_count) for val in range(class_count)]
            for patch, color in zip(bplot["boxes"], colors):
                patch.set_facecolor(color)
        else:
            raise NotImplementedError(f"{cmap} not implemented.")

    ticks = [i + 1 for i in range(len(class_names))]
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    ax.set_ylim([0, 1])
    plt.xlabel("Class")
    plt.ylabel(f"{metric_name}")
    if title is not None:
        plt.title(title)
    plt.grid(grid)
    fig.tight_layout()
    return fig


def plot_curves(y, colors, names, xlabel, ylabel):

    fig, ax = plt.subplots()

    for idx, (vals, color, name) in enumerate(zip(y, colors, names)):
        x = np.asarray(range(len(vals)))
        ax.plot(x, vals, color=color, label=f"{name}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()

    return fig


def plot_shaded_strip(
    mean_features,
    std_dev_features,
    strip_labels,
    xlabel,
    ylabel,
    colormap_name="YlOrRd",
    alpha=0.3,
    figsize=(12, 10),
):

    strip_count = len(strip_labels)

    # Get colormap from colormap name
    cmap = plt.cm.get_cmap(colormap_name)
    colors = cmap(np.linspace(0.1, 1.0, strip_count))

    fig, ax = plt.subplots(figsize=figsize)
    for mean, std_dev, label, color in zip(
        mean_features, std_dev_features, strip_labels, colors
    ):

        x = np.asarray(range(len(mean)))
        ax.plot(x, mean, label=label, c=color)
        ax.fill_between(
            x, mean - std_dev, mean + std_dev, alpha=alpha, facecolor=color
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()

    return fig


def show_mplt_colormap(colormap, figsize=(15, 3)):

    fig = plt.figure(figsize=figsize)

    # ToDo
    # Should we use len(colormap._lut) instead of colormap.N?

    # Specifying a step that is too small generates a black line plot due to
    # the density of values/a large width/height aspect ratio, although
    # applying the colormap to another arbitrary image shows that the colors
    # are effectively there, e.g.
    #
    # data = np.random.rand(10, 10) * 2 - 1
    # plt.pcolor(data, cmap=colormap)
    # plt.colorbar()
    # plt.show()
    #
    # Repeating the rows to reduce the width/height aspect ratio makes the
    # trick.

    # ToDo
    # When having a lot of colors the below does not work well; the colorbar
    # black border overlaps with the colors and we see nothing
    # step = 1  # 0.01
    # height = int(1 / step)
    # plt.imshow(
    #    np.asarray([list(np.arange(0, colormap.N, step))] * height),
    #    interpolation="nearest",
    #    origin="lower",
    #    cmap=colormap,
    # )
    # plt.imshow([list(np.arange(0, colormap.N, 0.1))],
    #           interpolation='nearest', origin='lower', cmap=colormap)

    # Alternatively
    from matplotlib.cm import ScalarMappable

    sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(0, colormap.N))
    sm.set_array([])
    plt.colorbar(sm)

    plt.xticks([])
    plt.yticks([])

    return fig
