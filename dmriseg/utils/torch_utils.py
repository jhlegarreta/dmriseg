# -*- coding: utf-8 -*-

import enum

import numpy as np
import torch


class CheckpointStateInfo(enum.Enum):
    BEST_METRIC = "best_metric"
    EPOCH = "epoch"
    MODEL_STATE = "model_state"
    OPTIMIZER_STATE = "optimizer_state"
    LR_SCHEDULER_STATE = "lr_scheduler_state"


def prepare_device():

    cuda = torch.cuda.is_available()

    seed = 1234

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

    return device, kwargs


def save_checkpoint(state, fname):
    torch.save(state, fname)


def build_checkpoint_state(
    epoch, best_metric, model, optimizer, lr_scheduler=None
):

    state = dict(
        {
            CheckpointStateInfo.BEST_METRIC.value: best_metric,
            CheckpointStateInfo.EPOCH.value: epoch,
            CheckpointStateInfo.MODEL_STATE.value: model.state_dict(),
            CheckpointStateInfo.OPTIMIZER_STATE.value: optimizer.state_dict(),
        }
    )

    if lr_scheduler is not None:
        state[
            CheckpointStateInfo.LR_SCHEDULER_STATE.value
        ] = lr_scheduler.state_dict()

    return state


def load_checkpoint_state(fname, device, model, optimizer, lr_scheduler=None):

    state = torch.load(fname, map_location=device)

    model.load_state_dict(state[CheckpointStateInfo.MODEL_STATE.value])
    optimizer.load_state_dict(state[CheckpointStateInfo.OPTIMIZER_STATE.value])

    if (
        lr_scheduler is not None
        and state[CheckpointStateInfo.LR_SCHEDULER_STATE.value] in state.keys()
    ):
        lr_scheduler.load_state_dict(
            state[CheckpointStateInfo.LR_SCHEDULER_STATE.value]
        )

    return (
        state[CheckpointStateInfo.BEST_METRIC.value],
        state[CheckpointStateInfo.EPOCH.value],
    )


def get_optimizer(model, method, **kwargs):

    if method == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif method == "adam":
        return torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.999),
            **kwargs,
        )
    elif method == "adamW":
        return torch.optim.AdamW(
            model.parameters(),
            betas=(0.9, 0.999),
            eps=1e-8,
            **kwargs,
        )
    elif method == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            momentum=0.6,  # 0.95,  # momentum factor
            alpha=0.90,  # smoothing constant (Discounting factor for the history/coming gradient)
            eps=1e-10,  # term added to the denominator to improve numerical stability
            weight_decay=1e-4,  # 0,  # weight decay (L2 penalty)
            centered=False,  # if True, compute the centered RMSProp (gradient normalized by estimation of its variance)
            **kwargs,
        )
    else:
        raise NotImplementedError(f"{method} optimizer not implemented.")


def get_lr_scheduler(optimizer, scheduler_type, **kwargs):

    if scheduler_type == "step_lr":
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, **kwargs)
    elif scheduler_type == "cosineWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, **kwargs
        )
    elif scheduler_type == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=0.001,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0,
            **kwargs,  # total_steps=None, epochs=self._epochs, steps_per_epoch=self.experiment_dict["num_steps_per_train_epoch"],
        )
    elif scheduler_type is None:
        return None
    else:
        raise ValueError(f"{scheduler_type} lr scheduler not supported.")
