#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Correct for multiple comparisons the results of a pairwise test.

When using `compute_pairwise_test.py` the corrected p-values (`p-corr`) are
corrected for all comparisons (e.g. A, B, C, etc.: A-B, A-C, B-C, etc.), not
only the ones we might be interested in (i.e. the pairwise comparisons between
the highest ranked contrast (A) and the rest (i.e. A-B, A-C, etc.), excluding
the pairwise comparisons between the rest of the pairs (i.e. B-C, etc.).

To be called on the results of `compute_pairwise_test.py`.
"""

import argparse
from pathlib import Path

import pandas as pd
import pingouin as pg

contrast_a_label = "A"
contrast_b_label = "B"
p_adjust_label = "p-adjust"
p_corr_label = "p-corr"
p_unc_label = "p-unc"
reject_label = "reject"


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_pairwise_results_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_fname",
        help="Output fname (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "interest_contrast_name",
        help="Interest contrast name",
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
    df = pd.read_csv(args.in_pairwise_results_fname, sep=sep)

    # Select the p-values of interest
    df_interest = df[
        (df[contrast_a_label] == args.interest_contrast_name)
        | (df[contrast_b_label] == args.interest_contrast_name)
    ]
    pvals = df_interest[p_unc_label].values

    # Perform the correction
    method = "fdr_bh"
    reject, pvals_corr = pg.multicomp(pvals, method=method)

    # Build the df
    df_p_corr = pd.DataFrame(
        {
            contrast_a_label: df_interest[contrast_a_label].values,
            contrast_b_label: df_interest[contrast_b_label].values,
            reject_label: reject,
            p_corr_label: pvals_corr,
            p_adjust_label: [method] * len(pvals),
        }
    )

    # Save the corrected values
    df_p_corr.to_csv(args.out_fname, sep=sep)


if __name__ == "__main__":
    main()
