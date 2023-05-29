#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from dmriseg.models.unet import UNet


def test_unet():
    # A full forward pass
    x = torch.randn(1, 32)
    model = UNet()
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Model: {model}")
    print(f"Output shape: {y.shape}")
