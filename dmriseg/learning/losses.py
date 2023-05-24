# -*- coding: utf-8 -*-

import enum

import torch.nn as nn


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1.0 - dsc


class LossFunction(enum.Enum):
    DICE = Dice.__name__.lower()


def get_loss_func(func_name):
    if func_name == LossFunction.DICE.value:
        return Dice()
    else:
        raise NotImplementedError(
            f"{func_name} loss function not implemented."
        )


def is_metric_improved(prev_metric, curr_metric, func_name):

    if func_name == LossFunction.DICE.value:
        return True if prev_metric > curr_metric else False
    else:
        raise ValueError(f"Unknown loss function name: {func_name}.")
