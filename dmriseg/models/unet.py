# -*- coding: utf-8 -*-

import torch.nn as nn


class UNet(nn.Module):
    """U-Net model."""

    def __init__(self, init_features=32):
        super(UNet, self).__init__()

        self._features = init_features

    def forward(self, x):
        return self._features * x
