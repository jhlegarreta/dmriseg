#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from dmriseg.data.lut.utils import (
    SuitAtlasDiedrichsenGroups,
    class_name_label,
    get_diedrichsen_group_labels,
)
from dmriseg.io.utils import fold_label
from dmriseg.io.utils import participant_label_id as _participant_label_id
from dmriseg.utils.contrast_utils import get_contrast_names_lut

participant_label_id = "participant_id"
class_id_label = "label_id"
contrast_label = "contrast"
pvalue_label = "pvalue"

significance_label = "significance"
statistic_label = "statistic"

arg_label = "arg"
x_label = "x"
y_label = "y"

test_description_label = "test_description"
test_short_name_label = "test_short_name"
stat_str_label = "stat_str"
alternative_label = "alternative"

subject_label = "subject"


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


def prepare_data_for_anova(dfs, measure, contrast_names):

    contrast_names_lut = get_contrast_names_lut()

    suit_labels = get_diedrichsen_group_labels(
        SuitAtlasDiedrichsenGroups.ALL.value
    )

    # Compute the mean across all labels for each participant/contrast
    columns_of_interest = list(map(str, suit_labels))

    measure_prtcpnt_mean = np.hstack(
        [df[columns_of_interest].mean(axis=1).values for df in dfs]
    )

    # Create the values for the participant (subject for AnovaRM) and contrast
    # (within for AnovaRM) columns
    participant_ids = np.hstack([df.index.to_numpy() for df in dfs])
    contrast = np.hstack(
        [
            len(df.index) * [contrast_names_lut[contrast_name]]
            for df, contrast_name in zip(dfs, contrast_names)
        ]
    )

    depvar_label = measure
    _subject_label = subject_label
    within_label = [contrast_label]

    df_anova = pd.DataFrame(
        {
            _subject_label: participant_ids,
            contrast_label: contrast,
            depvar_label: measure_prtcpnt_mean,
        }
    )

    return df_anova, depvar_label, _subject_label, within_label


# ToDo
# This is exactly the same as above, but I am not using the contrast_names_lut
# labels, but the actual contrast names
def prepare_data_for_pairwise_test(dfs, measure, contrast_names):

    suit_labels = get_diedrichsen_group_labels(
        SuitAtlasDiedrichsenGroups.ALL.value
    )

    # Compute the mean across all labels for each participant/contrast
    columns_of_interest = list(map(str, suit_labels))

    # Drop the fold column
    [df.drop(labels=[fold_label], axis=1, inplace=True) for df in dfs]

    measure_prtcpnt_mean = np.hstack(
        [df[columns_of_interest].mean(axis=1).values for df in dfs]
    )

    # Create the values for the participant and contrast columns
    # For the pairwise t-test, we are interested in knowing the significance
    # between contrasts (i.e. the fixed effect), whereas participants are the
    # measurements (i.e. the random effect)
    participant_ids = np.hstack([df.index.to_numpy() for df in dfs])
    contrast = np.hstack(
        [
            len(df.index) * [contrast_name]
            for df, contrast_name in zip(dfs, contrast_names)
        ]
    )

    depvar_label = measure
    _subject_label = subject_label
    between_label = contrast_label

    df_stat_test = pd.DataFrame(
        {
            _subject_label: participant_ids,
            between_label: contrast,
            depvar_label: measure_prtcpnt_mean,
        }
    )

    return df_stat_test, depvar_label, _subject_label, between_label


def describe_wilcoxon_ranksum(alternative=None):
    return pd.DataFrame(
        [["Wilcoxon rank sum", "wilco", "pvalue", alternative]],
        columns=[
            test_description_label,
            test_short_name_label,
            stat_str_label,
            alternative_label,
        ],
    )


def describe_measurements(contrast_names):

    df = pd.DataFrame(
        contrast_names, columns=[contrast_label], index=[x_label, y_label]
    )
    df.index.name = arg_label
    return df
