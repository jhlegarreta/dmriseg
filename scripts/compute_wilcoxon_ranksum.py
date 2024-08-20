#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute statistical significance between the performance of two contrast methods
using the Wilcoxon rank-sum hypothesis test. Assumes the first given contrast is
the reference against which the test is performed (x for scipy.stats.ranksums).
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ranksums

from dmriseg.data.lut.utils import SuitAtlasDiedrichsenGroups
from dmriseg.data.lut.utils import class_id_label as lut_class_id_label
from dmriseg.data.lut.utils import get_diedrichsen_group_labels
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import append_label_to_fname, build_suffix, underscore
from dmriseg.stats.utils import StatisticalTest
from dmriseg.utils.stat_preparation_utils import (
    arg_label,
    describe_measurements,
    describe_wilcoxon_ranksum,
    pvalue_label,
    significance_label,
    statistic_label,
)

description_label = "description"
measurement_label = "measurement"


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
        "out_dirname",
        help="Output dirname",
        type=Path,
    )
    parser.add_argument(
        "--in_performance_fnames",
        nargs="+",
        help="Performance filenames (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "--in_contrast_names",
        nargs="+",
        help="Contrast names",
        type=str,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    assert len(args.in_performance_fnames) == len(args.in_contrast_names)

    sep = "\t"
    dvf_ext = DelimitedValuesFileExtension.TSV

    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    labels = df_lut[lut_class_id_label].values[1:]

    index_col = 0
    df_metric_ref = pd.read_csv(
        args.in_performance_fnames[0], sep=sep, index_col=index_col
    )
    metric_values_ref = df_metric_ref.values
    contrast_name_ref = args.in_contrast_names[0]
    contrast_name = args.in_contrast_names[1]

    stat_test = StatisticalTest.WILCOXON_RANKSUM.value
    file_label = (
        stat_test
        + underscore
        + args.metric_name
        + underscore
        + contrast_name_ref
        + underscore
        + contrast_name
    )
    suffix = build_suffix(dvf_ext)

    index = [statistic_label, pvalue_label]
    alternative = "two-sided"

    df_metric = pd.read_csv(
        args.in_performance_fnames[1], sep=sep, index_col=index_col
    )
    metric_values = df_metric.values

    res = ranksums(metric_values_ref, metric_values, alternative=alternative)

    df = pd.DataFrame(
        data=[res.statistic, res.pvalue], columns=labels, index=index
    )

    file_basename = file_label + suffix
    fname = args.out_dirname / file_basename
    df.to_csv(fname, sep=sep, index=True, index_label=significance_label)

    # Create additional files to describe the statistical test performed
    descr_df = describe_wilcoxon_ranksum(alternative)
    descr_fname = append_label_to_fname(fname, description_label)
    descr_df.to_csv(descr_fname, sep=sep)
    measurement_df = describe_measurements([contrast_name_ref, contrast_name])
    measurement_fname = append_label_to_fname(fname, measurement_label)
    measurement_df.to_csv(
        measurement_fname, sep=sep, index=True, index_label=arg_label
    )

    # Compute overall significance and save the results to a file
    group_name = SuitAtlasDiedrichsenGroups.ALL

    labels = list(map(str, get_diedrichsen_group_labels(group_name.value)))
    # Compute the mean across labels and use the distribution across
    # participants to perform the statistical test:
    # - We are interested in how the overall measurements differ between
    # participants
    # - We want to understand if the differences between participants are
    # consistent when averaged over labels
    # ToDo
    # Not sure the above is OK: we are interested in measuring the difference
    # across CONTRASTS (our fixed effect); whereas PARTICIPANTS are our random
    # effect
    axis = 1
    particip_metric_values_ref = df_metric_ref[labels].mean(axis=axis)
    particip_metric_values = df_metric[labels].mean(axis=axis)
    particip_res = ranksums(
        particip_metric_values_ref,
        particip_metric_values,
        alternative=alternative,
    )

    particip_fname = append_label_to_fname(fname, group_name.value)
    particip_df = pd.DataFrame(
        data=[particip_res.statistic, particip_res.pvalue],
        columns=[group_name.value],
        index=index,
    )
    particip_df.to_csv(
        particip_fname, sep=sep, index=True, index_label=significance_label
    )

    # Create additional files to describe the statistical test performed
    _descr_df = describe_wilcoxon_ranksum(alternative)
    _fname = append_label_to_fname(
        particip_fname, description_label + underscore + group_name.value
    )
    _descr_df.to_csv(_fname, sep=sep)
    _measurement_df = describe_measurements(
        [contrast_name_ref, contrast_name + underscore + group_name.value]
    )
    _fname = append_label_to_fname(
        particip_fname, measurement_label + underscore + group_name.value
    )
    _measurement_df.to_csv(_fname, sep=sep, index=True, index_label=arg_label)


if __name__ == "__main__":
    main()
