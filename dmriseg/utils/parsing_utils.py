# -*- coding: utf-8 -*-

import argparse

from dmriseg.data.lut.utils import (
    Atlas,
    SuitAtlasBucknerVersion,
    SuitAtlasXueVersion,
    build_atlas_version_kwargs,
)


def rgb_color(sequence):
    try:
        r, g, b = map(int, sequence.split(","))
        return r, g, b
    except TypeError:
        raise argparse.ArgumentTypeError("RGB color must be R,G,B")


def parse_atlas_version_kwargs(parser, args):

    kwargs = dict({})
    if args.atlas == Atlas.BUCKNER:
        if args.buckner_version:
            kwargs = build_atlas_version_kwargs(
                args.atlas, args.buckner_version
            )
        else:
            parser.error(
                f"Version must be provided for atlas: f{args.atlas}. Options: {SuitAtlasBucknerVersion}"
            )
    elif args.atlas == Atlas.XUE:
        if args.xue_version:
            kwargs = build_atlas_version_kwargs(args.atlas, args.xue_version)
        else:
            parser.error(
                f"Version must be provided for atlas: f{args.atlas}. Options: {SuitAtlasXueVersion}"
            )

    return kwargs


def verify_background_label_data(parser, args):

    background_label_count = 0
    background_name_count = 0
    background_color_count = 0
    if args.background_label:
        background_label_count = len(args.background_label)
    if args.background_name:
        background_name_count = len(args.background_name)
    if args.background_color:
        background_color_count = len(args.background_color)
    if (
        background_label_count
        != background_name_count
        != background_color_count
    ):
        parser.error(
            "Background label, name and color should have the same number of elements."
        )
