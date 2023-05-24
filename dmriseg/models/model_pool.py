# -*- coding: utf-8 -*-

import enum

# from dmriseg.models.nucleusnet import NucleusNet
from dmriseg.models.unet import UNet


class ModelNames(enum.Enum):
    NUCLEUSNET = "nucleusnet"
    UNET = "unet"


def get_model(model_name, **kwargs):

    # if model_name == ModelNames.NUCLEUSNET.value:
    #    model = NucleusNet(**kwargs)
    if model_name == ModelNames.UNET.value:
        model = UNet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}.")

    return model
