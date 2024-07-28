# -*- coding: utf-8 -*-

import itertools

import pandas as pd
from statannotations.PValueFormat import PValueFormat
from statannotations.stats.StatResult import StatResult
from statannotations.utils import DEFAULT as STAT_DEFAULT

from dmriseg.utils.contrast_utils import rename_contrasts_plot_labels
from dmriseg.utils.stat_preparation_utils import (
    contrast_label,
    pvalue_label,
    stat_str_label,
    statistic_label,
    test_description_label,
    test_short_name_label,
)

annot_label = "annot_label"


def format_pvalues(
    df_stats, test_descr, test_short_name, stat_str, alpha=0.05
):

    formatter = PValueFormat()
    stat_res = [
        StatResult(
            test_descr,
            test_short_name,
            stat_str,
            df_stats[column][statistic_label],
            df_stats[column][pvalue_label],
            alpha=alpha,
        )
        for column in df_stats
    ]
    pval_thres = formatter._get_pvalue_thresholds(STAT_DEFAULT)

    return [formatter.format_data(elem) for elem in stat_res], pval_thres


# ToDo
# Check the overlap with utils.stat_preparation_utils.create_pairs
def create_pairs(labels, contrasts, label_mapping):

    label_names = [label_mapping[k] if k in labels else None for k in labels]
    all_combinations = list(itertools.product(label_names, contrasts))
    # Group the combinations by the label names
    _pairs = [
        [
            (item1, item2)
            for item1, item2 in all_combinations
            if item1 == label_name
        ]
        for label_name in label_names
    ]

    return _pairs


def format_stat_annotations(
    dfs_stats, dfs_measurements, df_stat_descr, label_mapping, alpha
):

    annots = []
    pairs = []
    pval_thres = None
    for df_stats, df_measurements in zip(dfs_stats, dfs_measurements):
        labels = df_stats.columns.values
        contrasts = df_measurements[contrast_label].values
        contrasts = list(map(rename_contrasts_plot_labels, contrasts))
        _pairs = create_pairs(labels, contrasts, label_mapping)
        annot, pval_thres = format_pvalues(
            df_stats,
            df_stat_descr[test_description_label].values.item(),
            df_stat_descr[test_short_name_label].values.item(),
            df_stat_descr[stat_str_label].values.item(),
            alpha=alpha,
        )

        annots.extend(annot)
        pairs.extend(_pairs)

    # All pval_thres values are the same, so take the last
    return annots, pairs, pval_thres


def create_pval_thres_annot_label_df(pval_thres):
    return pd.DataFrame(pval_thres, columns=[pvalue_label, annot_label])
