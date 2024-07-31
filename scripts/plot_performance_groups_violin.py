#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot the performance across different contrasts as grouped violin plots. The
groups are determined by the group names: e.g. for the dentate, three groups
will be created: the overall performance across its labels, and one for each the
left and right dentate; for the dcn, 7 groups will be created, the overall
performance across its labels, and one for each left and right dentate/
interposed/fastigial pairs. In each group there will be as many violin plots
as contrast data files are provided.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dmriseg.analysis.measures import Measure, rename_measure_names_plot_labels
from dmriseg.data.lut.utils import SuitAtlasDiedrichsenGroups
from dmriseg.data.lut.utils import class_id_label as lut_class_id_label
from dmriseg.data.lut.utils import (
    class_name_label,
    get_diedrichsen_group_labels,
    rename_suit_atlas_diedrichsen_groups_plot_labels,
)
from dmriseg.io.file_extensions import (
    DelimitedValuesFileExtension,
    FigureFileExtension,
)
from dmriseg.io.utils import (
    append_label_to_fname,
    build_suffix,
    contrast_label,
    fold_label,
    legend_label,
    participant_label_id,
    underscore,
)
from dmriseg.utils.contrast_utils import (
    ContrastNames,
    get_contrast_from_dir_base,
    rename_contrasts_plot_labels,
)
from dmriseg.utils.stat_preparation_utils import arg_label, significance_label
from dmriseg.visualization.plot_utils import (
    create_mpl_fig_from_legend_props,
    get_plot_ylim,
)

# from statannotations.Annotator import Annotator


label_label = "label"
score_label = "score"
medians_label = "medians"
means_label = "means"
q25_label = "q25"
q75_label = "q75"

pval_thres_label = "pval_thres"


def get_contrast_order(df):
    contrast_names = list(
        map(
            rename_contrasts_plot_labels,
            list(map(lambda x: x.value, list(ContrastNames))),
        )
    )
    _contrasts = set(df[contrast_label])
    return [x for x in contrast_names if x in _contrasts]


def get_label_order(df, label_names):
    _labels = set(df[label_label])
    labels = [x for x in label_names if x in _labels]
    # Prepend the overall label
    overall_label = list(_labels.difference(labels))
    assert len(overall_label) == 1
    return [overall_label[0], *labels]


def get_group_label_names(group_name, label_mapping):
    labels = list(map(str, get_diedrichsen_group_labels(group_name.value)))
    return [label_mapping[k] if k in labels else None for k in labels]


def filter_labels(df, group_name, label_mapping):
    label_names = get_group_label_names(group_name, label_mapping)
    group_df = df[df[label_label].isin(label_names)]

    return group_df.reset_index(drop=True)


def get_indices(source_list, elements_list):
    return [
        i for i, element in enumerate(source_list) if element in elements_list
    ]


def filter_pvalue_labels(
    pvalue_labels, stat_signif_pairs, group_name, label_mapping
):
    label_names = get_group_label_names(group_name, label_mapping)
    pairs_label_names = [pair[0][0] for pair in stat_signif_pairs]
    idx = get_indices(pairs_label_names, label_names)

    pvalue_labels_interest = list(map(lambda x: pvalue_labels[x], idx))
    stat_signif_pairs_interest = list(map(lambda x: stat_signif_pairs[x], idx))

    return pvalue_labels_interest, stat_signif_pairs_interest


def plot_grouped_violin(
    df,
    label_mapping,
    measure_name,
    stat_signif_pairs,
    pvalue_labels,
    palette_name="husl",
    xlabel=False,
    rotate_xlabels=False,
    separate_legend_from_fig=False,
    bottom_legend=False,
    use_scatter_legend=False,
):

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(14, 7))

    hue_order = get_contrast_order(df)
    order = get_label_order(df, list(label_mapping.values()))

    hue_count = len(hue_order)

    data_params = {"x": label_label, "y": measure_name, "data": df}
    hue_plot_params = {
        "order": order,
        "hue": contrast_label,
        "hue_order": hue_order,
        "dodge": True,
    }
    group_data_params = {**data_params, **hue_plot_params}

    # Create the violin plot
    sns.violinplot(
        **group_data_params, cut=0, color="#DDDDDD", inner=None
    )  # , scale='width')

    # Draw the data points
    custom_palette = sns.color_palette(
        palette_name, df[contrast_label].nunique()
    )
    data_params = {
        "palette": custom_palette,
        "jitter": True,
        "linewidth": 1,
        "size": 6,
    }
    sns.stripplot(
        x=label_label,
        y=measure_name,
        data=df,
        **hue_plot_params,
        **data_params,
    )

    # Compute the mean and 25 and 75 quantiles
    groupby = [label_label, contrast_label]
    # medians = df.groupby(groupby)[measure_name].median(numeric_only=True).reset_index()
    # medians.rename(columns={measure_name: medians_label}, inplace=True)
    means = (
        df.groupby(groupby)[measure_name].mean(numeric_only=True).reset_index()
    )
    means.rename(columns={measure_name: means_label}, inplace=True)
    q25 = df.groupby(groupby).quantile(q=0.25, numeric_only=True).reset_index()
    q25.rename(columns={measure_name: q25_label}, inplace=True)
    q75 = df.groupby(groupby).quantile(q=0.75, numeric_only=True).reset_index()
    q75.rename(columns={measure_name: q75_label}, inplace=True)

    # Draw the median
    medians_params = {
        "palette": ["white"] * hue_count,
        "jitter": False,
        "edgecolor": "black",
        "linewidth": 1,
        "size": 6,
    }
    sns.stripplot(
        x=label_label,
        y=means_label,
        data=means,
        **hue_plot_params,
        **medians_params,
    )

    # Draw the 25 and 75 quantiles
    quantiles_params = {
        "palette": ["red"] * hue_count,
        "jitter": False,
        "edgecolor": "black",
        "linewidth": 1,
        "size": 4,
    }
    sns.stripplot(
        x=label_label,
        y=q25_label,
        data=q25,
        **hue_plot_params,
        **quantiles_params,
    )
    sns.stripplot(
        x=label_label,
        y=q75_label,
        data=q75,
        **hue_plot_params,
        **quantiles_params,
    )

    # Add statistical significance annotations
    # annotator = Annotator(ax, stat_signif_pairs, **group_data_params)
    # annotator.configure(loc="outside")
    # annotator.annotate_custom_annotations(pvalue_labels)

    sns.despine(top=True, left=True)
    ax.grid(axis="y")
    ax.set(ylabel=measure_name)
    if not xlabel:
        plt.xlabel(None)
    if rotate_xlabels:
        plt.xticks(rotation=45, ha="right")

    plt.ylabel(rename_measure_names_plot_labels(measure_name))

    ylim = get_plot_ylim(measure_name)
    ax.set_ylim(ylim)

    # Get the handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Use the first N elements to create the legend and thus avoid multiple
    # legends from the multiple calls to violinplot and stripplot. The first
    # elements correspond to the legend created by the first plot call
    # (violinplot). Alternatively, use the scatter data legend.
    start_idx = 0
    stop_idx = hue_count
    if use_scatter_legend:
        start_idx = 3
        stop_idx = 6

    loc = "upper left"
    bbox_to_anchor = (1.05, 1)
    ncols = 1
    borderaxespad = 0.0
    title = None
    frameon = False

    if bottom_legend:
        loc = "upper center"
        bbox_to_anchor = (0.5, -0.1)
        ncols = hue_count

    # Set a black edgecolor
    for ha in handles:
        ha.set_edgecolor("black")

    legend_props = {
        "handles": handles[start_idx:stop_idx],
        "labels": labels[start_idx:stop_idx],
        "bbox_to_anchor": bbox_to_anchor,
        "loc": loc,
        "ncols": ncols,
        "borderaxespad": borderaxespad,
        "title": title,
        "frameon": frameon,
    }
    plt.legend(**legend_props)

    if separate_legend_from_fig:
        ax.legend().set_visible(False)

    plt.tight_layout()

    # bbox_inches = "tight"
    return fig, legend_props


def prepare_df(dfs, contrasts, measure_name, label_mapping):

    # Add the contrast column to each df
    [
        df.insert(2, contrast_label, contrast)
        for df, contrast in zip(dfs, contrasts)
    ]

    # Set the participant label id and contrast as indices for each df
    # [df.set_index(contrast_label, append=True, inplace=True) for df in dfs]

    # Stack the dfs
    df = pd.concat(dfs)

    # Rename labels to the corresponding names
    df.rename(columns=label_mapping, inplace=True)

    # Put the label scores into a single column and make their column names
    # become values in an additional column
    melted_df = pd.melt(
        df,
        id_vars=[participant_label_id, fold_label, contrast_label],
        value_vars=list(label_mapping.values()),
        var_name=label_label,
        value_name=measure_name,
    )

    return melted_df


def aggregate_overall_performance(df, group_name):

    # Compute the mean across all labels for each participant/contrast
    groupby = [participant_label_id, contrast_label, fold_label]
    means_df = df.groupby(groupby).mean(numeric_only=True).reset_index()
    # Add it as an additional record for each participant/contrast with the label group_name.value
    means_df[label_label] = rename_suit_atlas_diedrichsen_groups_plot_labels(
        group_name.value
    )
    extended_group_df = pd.concat([df, means_df]).reset_index(drop=True)

    return extended_group_df


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--in_performance_dirnames",
        help="Input dirname where performance data files dwell (*.tsv)",
        type=Path,
        nargs="+",
    )
    parser.add_argument(
        "--out_dirname",
        help="Output dirname (*.png)",
        type=Path,
    )
    parser.add_argument(
        "--measure_name",
        help="Measure name",
        type=str,
    )
    parser.add_argument(
        "--in_labels_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "--in_significance_fnames",
        help="Significance filenames (*.tsv)",
        type=Path,
        nargs="+",
    )
    parser.add_argument(
        "--in_description_fnames",
        help="Description filenames (*.tsv)",
        type=Path,
        nargs="+",
    )
    parser.add_argument(
        "--in_measurement_fnames",
        help="Measurement filenames (*.tsv)",
        type=Path,
        nargs="+",
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Ensure input argument data lengths match
    assert len(args.in_performance_dirnames) == len(
        args.in_significance_fnames
    )
    assert len(args.in_performance_dirnames) == len(args.in_description_fnames)
    assert len(args.in_performance_dirnames) == len(args.in_measurement_fnames)

    alpha = 0.05

    seed = 1234
    np.random.seed(seed=seed)

    dvf_ext = DelimitedValuesFileExtension.TSV
    sep = "\t"

    measure = Measure(args.measure_name).value

    data_suffix = build_suffix(dvf_ext)
    file_basename = measure + data_suffix
    fnames = [
        dirname / file_basename for dirname in args.in_performance_dirnames
    ]

    # dfs = [pd.read_csv(fname, sep=sep, index_col=participant_label_id) for fname in fnames]
    dfs = [pd.read_csv(fname, sep=sep) for fname in fnames]

    parent_dirnames = [
        dirname.parent.name for dirname in args.in_performance_dirnames
    ]
    _contrasts = [
        get_contrast_from_dir_base(parent_dirname)
        for parent_dirname in parent_dirnames
    ]

    # Rename the contrast labels to stick to the naming chosen for the paper
    contrasts = list(map(rename_contrasts_plot_labels, _contrasts))

    # Read the labels to create the id to name mapping
    # from dmriseg.dataset.utils import suit_lut  # would be another possibility
    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    keys_column = lut_class_id_label
    values_column = class_name_label
    # Make label IDs be strings
    label_mapping = {
        str(key): value
        for key, value in df_lut.set_index(keys_column)[values_column]
        .to_dict()
        .items()
    }
    # Drop the background
    del label_mapping["0"]

    # Prepare the data: compose a dataframe where we add the contrast column
    df = prepare_df(dfs, contrasts, args.measure_name, label_mapping)

    from dmriseg.stats.annotation_utils import (
        create_pval_thres_annot_label_df,
        format_stat_annotations,
    )

    dfs_stats = [
        pd.read_csv(fname, sep=sep, index_col=[significance_label])
        for fname in args.in_significance_fnames
    ]
    dfs_measurements = [
        pd.read_csv(fname, sep=sep, index_col=[arg_label])
        for fname in args.in_measurement_fnames
    ]
    dfs_stat_descr = [
        pd.read_csv(fname, sep=sep) for fname in args.in_description_fnames
    ]
    # Ensure that all df_stat_descr are the same
    [
        pd.testing.assert_frame_equal(dfs_stat_descr[0], df)
        for df in dfs_stat_descr
    ]

    pvalue_labels, stat_signif_pairs, pval_thres = format_stat_annotations(
        dfs_stats, dfs_measurements, dfs_stat_descr[0], label_mapping, alpha
    )

    # Save pval_thres to a TSV to be able to tell the thresholds used for the
    # annotations
    pval_thres_df = create_pval_thres_annot_label_df(pval_thres)
    _file_basename = measure + underscore + pval_thres_label + data_suffix
    fname = args.out_dirname / _file_basename
    pval_thres_df.to_csv(fname, sep="\t")

    # ToDo
    # If the statistical significance between groups is to be displayed, that
    # would require to provide the corresponding three TSV files (stats, meas,
    # and descr) required to generate the annotation labels for each group
    # and contrast pair, so in that case, it would be easier to make the group
    # name an argument to the script.

    # Generate plots: one containing each labels as a separate entity, then
    # averaging the performances across labels in each group
    group_names = [
        SuitAtlasDiedrichsenGroups.DCN,
        SuitAtlasDiedrichsenGroups.DENTATE,
        SuitAtlasDiedrichsenGroups.INTERPOSED,
        SuitAtlasDiedrichsenGroups.FASTIGIAL,
        SuitAtlasDiedrichsenGroups.VERMIS,
        SuitAtlasDiedrichsenGroups.LOBULES,
        SuitAtlasDiedrichsenGroups.CRUS,
    ]

    palette_name = "husl"
    rotate_xlabels = False
    xlabel = False
    use_scatter_legend = True
    bottom_legend = True
    separate_legend_from_fig = True

    plot_ext = FigureFileExtension.SVG
    plot_suffix = build_suffix(plot_ext)
    # figsize = (1920, 1080)
    # dpi = 300

    for group_name in group_names:

        group_df = filter_labels(df, group_name, label_mapping)
        extended_group_df = aggregate_overall_performance(group_df, group_name)

        # Keep only the pvalue_labels and pairs of interest
        _pvalue_labels, _stat_signif_pairs = filter_pvalue_labels(
            pvalue_labels, stat_signif_pairs, group_name, label_mapping
        )

        # ToDo
        # Prepend the group significance labels OR mark only the group
        # significance in the plot if their significance is to be displayed

        # Prepare colors according to the SUIT colors
        # ToDo
        # We should add another color for the overall label
        # labels = list(map(str, get_diedrichsen_group_labels(group_name.value)))
        # label_names = [
        #    label_mapping[k] if k in labels else None for k in labels
        # ]
        # group_lut_df = df_lut[df_lut["LabelName"].isin(label_names)]
        # palette = np.stack([group_lut_df["R"], group_lut_df["G"], group_lut_df["B"]]).T / 255

        fig, legend_props = plot_grouped_violin(
            extended_group_df,
            label_mapping,
            args.measure_name,
            _stat_signif_pairs,
            _pvalue_labels,
            palette_name=palette_name,
            xlabel=xlabel,
            rotate_xlabels=rotate_xlabels,
            separate_legend_from_fig=separate_legend_from_fig,
            bottom_legend=bottom_legend,
            use_scatter_legend=use_scatter_legend,
        )

        file_basename = measure + underscore + group_name.value + plot_suffix
        fname = args.out_dirname / file_basename
        fig.savefig(fname)
        # Convert figure to pillow to save it with the desired size
        # image = mplfig2img(fig)
        # img = rescale_image_keep_aspect(image, figsize)
        # img.save(fname, dpi=(dpi, dpi))

        # Save the legend
        if separate_legend_from_fig:
            # Create a new figure for the legend
            fig_legend = create_mpl_fig_from_legend_props(**legend_props)
            legend_fname = append_label_to_fname(fname, legend_label)
            fig_legend.savefig(legend_fname, transparent=True)


if __name__ == "__main__":
    main()
