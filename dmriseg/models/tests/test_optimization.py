#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import decimal

import torch
from monai.losses import DiceCELoss, HausdorffDTLoss

from dmriseg.dataset.utils import get_model
from dmriseg.models.optimization import OptimizationScheduler


def test_optimization_scheduler():

    to_steal = 0.01
    scheduler = OptimizationScheduler(to_steal)

    epochs = 20
    model_name = "SegResNet16"
    out_channels = 3  # background + 2 foreground labels, e.g. fastigial nuclei
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_2d = False
    model_pth = None
    model = get_model(
        model_name, out_channels, device, test_2d=test_2d, model_pth=model_pth
    )
    lr = 1.0e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fns = [DiceCELoss, HausdorffDTLoss]
    loss_weights = [1, 0.01]

    delta_weights = epochs * to_steal

    dec_places = abs(decimal.Decimal(str(to_steal)).as_tuple().exponent)
    _places = decimal.Decimal(f"1e-{dec_places}")

    _optimizer = copy.deepcopy(optimizer)
    _loss_fns = copy.deepcopy(loss_fns)
    _loss_weights = copy.deepcopy(loss_weights)
    _loss_weights_prec = None
    for epoch in range(epochs):
        _optimizer, _loss_fns, _loss_weights = scheduler(
            epoch, _optimizer, _loss_fns, _loss_weights
        )
        # The sum of the loss weights does not change
        _loss_weights_prec = list(
            map(
                lambda x: float(
                    decimal.Decimal(x).quantize(
                        _places, rounding=decimal.ROUND_HALF_UP
                    )
                ),
                _loss_weights,
            )
        )
        assert sum(_loss_weights_prec) == sum(loss_weights)
        assert _loss_weights_prec[1] == round(
            loss_weights[1] + (epoch + 1) * to_steal, ndigits=dec_places
        )
        assert _loss_weights_prec[0] == round(
            loss_weights[0] - (epoch + 1) * to_steal, ndigits=dec_places
        )

    assert _loss_fns == loss_fns
    assert _loss_weights_prec[1] == round(
        loss_weights[1] + delta_weights, ndigits=dec_places
    )
    assert _loss_weights_prec[0] == round(
        loss_weights[0] - delta_weights, ndigits=dec_places
    )

    # Check that the lower bound of the first term is scheduler.first_lower_bound
    to_steal = 0.5
    scheduler = OptimizationScheduler(to_steal)

    epochs = 2

    delta_weights = epochs * to_steal

    _optimizer = copy.deepcopy(optimizer)
    _loss_fns = copy.deepcopy(loss_fns)
    _loss_weights = copy.deepcopy(loss_weights)
    _loss_weights_prec = None
    for epoch in range(epochs):
        _optimizer, _loss_fns, _loss_weights = scheduler(
            epoch, _optimizer, _loss_fns, _loss_weights
        )
        _loss_weights_prec = list(
            map(
                lambda x: float(
                    decimal.Decimal(x).quantize(
                        _places, rounding=decimal.ROUND_HALF_UP
                    )
                ),
                _loss_weights,
            )
        )
        if _loss_weights_prec[0] > scheduler.first_lower_bound:
            # The sum of the loss weights does not change
            assert sum(_loss_weights_prec) == sum(loss_weights)

    assert _loss_fns == loss_fns
    assert _loss_weights_prec[1] == loss_weights[1] + delta_weights
    assert _loss_weights_prec[0] == scheduler.first_lower_bound
