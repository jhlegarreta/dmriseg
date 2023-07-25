#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert SUIT lut file to tsv."""

import argparse
from pathlib import Path

from dmriseg.data.lut.utils import (
    Atlas,
    SuitAtlasBucknerVersion,
    SuitAtlasXueVersion,
    add_additional_label_to_lut,
    add_alpha_to_lut,
    fetch_atlas_cmap_lut_file,
    lut2df,
    read_lut_data,
    rescale_lut,
)
from dmriseg.utils.parsing_utils import (
    parse_atlas_version_kwargs,
    rgb_color,
    verify_background_label_data,
)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("atlas", help="Atlas name.", type=Atlas)
    parser.add_argument(
        "out_tsv", help="Output TSV filename (*tsv).", type=Path
    )
    parser.add_argument(
        "--rescale", action="store_true", help="Rescale to [0, 255]."
    )
    parser.add_argument("--alpha", help="Alpha value.", type=int)
    parser.add_argument(
        "--buckner_version",
        help="SUIT atlas Buckner version.",
        type=SuitAtlasBucknerVersion,
    )
    parser.add_argument(
        "--xue_version",
        help="SUIT atlas Buckner version.",
        type=SuitAtlasXueVersion,
    )
    parser.add_argument(
        "--background_label",
        help="Background label.",
        nargs="+",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--background_name",
        help="Background name.",
        nargs="+",
        type=str,
        default=[],
    )
    parser.add_argument(
        "--background_color",
        help="Background color.",
        nargs="+",
        type=rgb_color,
        default=[],
    )

    return parser


def _parse_args(parser):

    args = parser.parse_args()

    kwargs = parse_atlas_version_kwargs(parser, args)
    verify_background_label_data(parser, args)

    return (
        args.atlas,
        args.out_tsv,
        args.rescale,
        args.alpha,
        args.background_label,
        args.background_name,
        args.background_color,
        kwargs,
    )


def main():

    parser = _build_arg_parser()
    (
        atlas,
        out_tsv,
        rescale,
        alpha,
        background_label,
        background_name,
        background_color,
        kwargs,
    ) = _parse_args(parser)

    fname = fetch_atlas_cmap_lut_file(atlas, **kwargs)

    lut = read_lut_data(fname)

    if rescale:
        lut = rescale_lut(lut)

    if background_label is not None:
        lut = add_additional_label_to_lut(
            lut, background_label, background_name, background_color
        )

    if alpha is not None:
        lut = add_alpha_to_lut(lut, alpha)

    df = lut2df(lut)
    df.to_csv(out_tsv, sep="\t", columns=df.columns, index_label=df.index.name)


if __name__ == "__main__":
    main()
