#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from dmriseg.data.lut.utils import class_name_label
from dmriseg.io.utils import participant_label_id as _participant_label_id

participant_label_id = "participant_id"
class_id_label = "label_id"
contrast_label = "contrast"
pvalue_label = "pvalue"

significance_label = "significance"
statistic_label = "statistic"


def create_pairs(ref_contrast_name, contrast_name, labels):
    """Create all possible pairs between the labels and the contrast names."""

    return [
        [(str(label), ref_contrast_name), (str(label), contrast_name)]
        for label in labels
    ]


def create_df(
    df_metric_ref, df_metric, ref_contrast_name, contrast_name, metric_name
):
    """Create a DataFrame that will be used to plot the statistical significance
    values between contrast pairs. Reformats the input dataframes so that each
    metric value (corresponding to a different label) is contained in a separate
    record, and thus, adds the class name label and metric names as column.
    Appends the appropriate contrast name to each record. Finally, concatenates
    the reformatted dataframes."""

    # Melt the DataFrame to convert it to long format
    df_metric_ref_melted = df_metric_ref.reset_index().melt(
        id_vars=_participant_label_id,
        var_name=class_name_label,
        value_name=metric_name,
    )

    # Sort the melted DataFrame by ID and reset index
    df_metric_ref_melted = df_metric_ref_melted.sort_values(
        by=[_participant_label_id, class_name_label]
    ).reset_index(drop=True)

    # Add the contrast label
    df_metric_ref_melted[contrast_label] = ref_contrast_name

    # Rename participant and label id columns so that they have appropriate names
    df_metric_ref_melted = df_metric_ref_melted.rename(
        columns={
            _participant_label_id: participant_label_id,
            class_name_label: class_id_label,
        }
    )

    # Do the same for the other metric
    # Melt the DataFrame to convert it to long format
    df_metric_melted = df_metric.reset_index().melt(
        id_vars=_participant_label_id,
        var_name=class_name_label,
        value_name=metric_name,
    )

    # Sort the melted DataFrame by ID and reset index
    df_metric_melted = df_metric_melted.sort_values(
        by=[_participant_label_id, class_name_label]
    ).reset_index(drop=True)

    # Add the contrast label
    df_metric_melted[contrast_label] = contrast_name

    # Rename participant and label id columns so that they have appropriate names
    df_metric_melted = df_metric_melted.rename(
        columns={
            _participant_label_id: participant_label_id,
            class_name_label: class_id_label,
        }
    )

    # Concat the dfs
    return pd.concat(
        [df_metric_ref_melted, df_metric_melted], axis=0
    ).reset_index(drop=True)
