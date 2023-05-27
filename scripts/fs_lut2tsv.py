#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FreeSurfer color LUT txt to tsv."""

import argparse
from pathlib import Path

from dmriseg.data.lut.utils import lut2df, read_lut_from_txt


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_txt", help="Input LUT filename (*.txt).", type=Path
    )
    parser.add_argument(
        "out_tsv", help="Output TSV filename (*tsv).", type=Path
    )

    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    lut = read_lut_from_txt(args.in_txt)
    df = lut2df(lut)
    df.to_csv(args.out_tsv, sep="\t")


if __name__ == "__main__":
    main()
