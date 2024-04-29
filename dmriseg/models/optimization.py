# -*- coding: utf-8 -*-

import logging
from typing import Any, Callable, List, Tuple

logger = logging.getLogger("root")


# Named StealWeight; taken from
# https://github.com/rosanajurdi/Prior-based-Losses-for-Medical-Image-Segmentation/blob/eab8b9c27b776b70a1491ae44e0e2e52d0f1638f/scheduler.py
# In reality, the only thing it changes are the weights, so it should not take
# the optimizer and the loss functions as parameters
class OptimizationScheduler:
    """Re-weights the loss weights according to the ``to_steal`` value: subtract
    that value to the weight corresponding to the first term (e.g. Dice loss),
    and increases the weight corresponding to the second term (e.g. a boundary
    loss) by that value.

    The first term's weight lower bound is set to ``first_lower_bound``.

    Assumes two loss terms only.
    """

    def __init__(self, to_steal: float, first_lower_bound: float = 0.1):
        self._to_steal: float = to_steal
        self._first_lower_bound: float = first_lower_bound

    def __call__(
        self,
        epoch: int,
        optimizer: Any,
        loss_fns: List[Callable],
        loss_weights: List[float],
    ) -> Tuple[float, List[Callable], List[float]]:

        a, b = loss_weights
        # Set a lower bound for the first loss term weight
        new_weights: List[float] = [
            max(self._first_lower_bound, a - self._to_steal),
            b + self._to_steal,
        ]

        # print(f"Loss weights went from {loss_weights} to {new_weights}")
        logger.info(f"Loss weights went from {loss_weights} to {new_weights}")

        return optimizer, loss_fns, new_weights

    @property
    def first_lower_bound(self):
        return self._first_lower_bound
