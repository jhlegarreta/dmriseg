#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dmriseg.data.lut.utils import (
    Atlas,
    SuitAtlasBucknerVersion,
    SuitAtlasXueVersion,
    buckner_version_arg,
    build_atlas_version_kwargs,
    get_atlas_cmap,
    suit_atlas_version_arg,
)


def test_fetch_atlas_cmap():

    atlas = Atlas.__members__.values()

    buckner_version = SuitAtlasBucknerVersion.__members__.values()
    xue_version = SuitAtlasXueVersion.__members__.values()

    args = []
    for _atlas in atlas:
        if _atlas == Atlas.BUCKNER:
            for version in buckner_version:
                version_kwargs = build_atlas_version_kwargs(_atlas, version)
                kwargs = dict({suit_atlas_version_arg: version_kwargs})
        elif _atlas == Atlas.XUE:
            for version in xue_version:
                version_kwargs = build_atlas_version_kwargs(_atlas, version)
                kwargs = dict({suit_atlas_version_arg: version_kwargs})
        else:
            kwargs = dict({})

        args.append((_atlas, kwargs))

    for _atlas, kwargs in args:

        cmap = get_atlas_cmap(_atlas, **kwargs)

        if _atlas == Atlas.DKT:
            assert len(cmap) == 1374
        elif _atlas == Atlas.BUCKNER:
            if (
                kwargs[suit_atlas_version_arg][buckner_version_arg]
                == SuitAtlasBucknerVersion.R7
            ):
                assert len(cmap) == 7
            elif (
                kwargs[suit_atlas_version_arg][buckner_version_arg]
                == SuitAtlasBucknerVersion.R17
            ):
                assert len(cmap) == 17
        elif _atlas == Atlas.DIEDRICHSEN:
            assert len(cmap) == 34
        elif _atlas == Atlas.JI:
            assert len(cmap) == 10
        elif _atlas == Atlas.KING:
            assert len(cmap) == 10
        elif _atlas == Atlas.XUE:
            assert len(cmap) == 10

        assert [len(values) == 3 for values in cmap]
