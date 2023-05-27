#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert SUIT lut file to tsv."""

import argparse
from pathlib import Path

from dmriseg.data.lut.utils import (
    Atlas,
    SuitAtlasBucknerVersion,
    SuitAtlasXueVersion,
    add_alpha_to_lut,
    build_atlas_version_kwargs,
    fetch_atlas_cmap_lut_file,
    lut2df,
    read_lut_data,
    rescale_lut,
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

    return parser


def _parse_args(parser):

    args = parser.parse_args()

    kwargs = dict({})
    if args.atlas == Atlas.BUCKNER:
        if args.buckner_version:
            kwargs = build_atlas_version_kwargs(
                args.atlas, args.buckner_version
            )
        else:
            raise ValueError(
                f"Version must be provided for atlas: f{args.atlas}. Options: {SuitAtlasBucknerVersion}"
            )
    elif args.atlas == Atlas.XUE:
        if args.xue_version:
            kwargs = build_atlas_version_kwargs(args.atlas, args.xue_version)
        else:
            raise ValueError(
                f"Version must be provided for atlas: f{args.atlas}. Options: {SuitAtlasXueVersion}"
            )

    return args.atlas, args.out_tsv, args.rescale, args.alpha, kwargs


def main():

    parser = _build_arg_parser()
    atlas, out_tsv, rescale, alpha, kwargs = _parse_args(parser)

    fname = fetch_atlas_cmap_lut_file(atlas, **kwargs)

    lut = read_lut_data(fname)

    if rescale:
        lut = rescale_lut(lut)

    if alpha is not None:
        lut = add_alpha_to_lut(lut, alpha)

    df = lut2df(lut)
    df.to_csv(out_tsv, sep="\t", columns=df.columns, index_label=df.index.name)


if __name__ == "__main__":
    main()
