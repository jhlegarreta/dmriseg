# -*- coding: utf-8 -*-

from dmriseg.models.model_pool import get_model


def build_model(model_name, **model_kwargs):

    return get_model(model_name, **model_kwargs)
