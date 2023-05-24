# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm

from dmriseg.dataset.dmri_data_management import DMRIDataManager, LearningSplit
from dmriseg.dataset.dmri_dataset import LearningDataParts
from dmriseg.io.utils import build_checkpoint_fname
from dmriseg.learning import learning_keys
from dmriseg.learning.losses import get_loss_func, is_metric_improved
from dmriseg.models.utils import build_model
from dmriseg.utils.torch_utils import (
    build_checkpoint_state,
    get_lr_scheduler,
    get_optimizer,
    load_checkpoint_state,
    save_checkpoint,
)


# ToDo
# Rename altogether to training_manager, TrainManager ??
class DMRISegLearner(object):
    """Learns to segment tissues from dMRI data."""

    def __init__(
        self,
        dataset_fname,
        experiment_dirname,
        model_name,
        device,
        loss_func_name,
        epochs,
        optimizer_method,
        lr_scheduler_type,
        **kwargs,
    ):

        model_kwargs = kwargs.pop(learning_keys.MODEL_KWARGS, {})
        optimizer_kwargs = kwargs.pop(learning_keys.OPTIMIZER_KWARGS, {})
        lr_scheduler_kwargs = kwargs.pop(learning_keys.LR_SCHEDULER_KWARGS, {})

        self._train_kwargs = kwargs.pop(learning_keys.TRAIN_KWARGS, {})

        self._dam = DMRIDataManager(dataset_fname)

        self._device = device
        self._epochs = epochs
        self._experiment_dirname = experiment_dirname
        self._model = build_model(model_name, **model_kwargs)
        self._loss_func = get_loss_func(loss_func_name)
        self._optimizer = get_optimizer(
            self._model, optimizer_method, **optimizer_kwargs
        )
        self._lr_scheduler = get_lr_scheduler(
            self._optimizer, lr_scheduler_type, **lr_scheduler_kwargs
        )

        # self._best_metric, self._start_epoch = self.load_model_state(checkpoint_fname, optimizer, scheduler=scheduler)

    def train(self, data_loader, epoch):

        self._model.train()
        train_loss = 0

        for batch_idx, batch in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):

            self._optimizer.zero_grad()

            # Get the images and label maps from the batch
            # Need to cast to torch.float32 to avoid
            # RuntimeError: Input type (double) and bias type (float) should be the same
            images = batch[LearningDataParts.IMAGE.value].to(
                self._device, dtype=torch.float32
            )
            labelmaps = batch[LearningDataParts.LABELMAP.value].to(
                self._device
            )

            # ToDo
            # Speed up training using mixed precision
            # Casts operations to mixed precision
            # with torch.cuda.amp.autocast():
            #    loss = model(data)

            pred = self._model(images).type(torch.int64)
            loss = self._loss_func(pred, labelmaps)
            loss.requires_grad = True
            loss.backward()
            train_loss += loss.item()

            self._optimizer.step()
            self._lr_scheduler.step()

            # ToDo
            # Report progress: epoch and loss

        # ToDo
        # Record/log the loss to comet

    @torch.no_grad()
    def eval(self, loader, epoch):
        self._model.eval()

        metric = 0
        return metric

    def save_model_state(self, epoch, best_metric, optimizer):
        state = build_checkpoint_state(
            epoch, best_metric, self._model, optimizer
        )
        fname = build_checkpoint_fname(self._experiment_dirname)
        save_checkpoint(state, fname)

    def load_model_state(self, fname, optimizer):
        return load_checkpoint_state(
            fname, self._device, self._model, optimizer
        )

    def run(self):

        # ToDo
        # Resume if a previous checkpoint has been provided
        # self._best_metric, self._start_epoch
        start_epoch = 0
        best_metric = 0

        # Transfer the model to device(s)
        self._model = self._model.to(self._device)

        train_loader = self._dam.get_loader(
            LearningSplit.TRAIN, **self._train_kwargs
        )
        valid_loader = self._dam.get_loader(
            LearningSplit.VALID, **self._train_kwargs
        )

        for epoch in range(start_epoch, self._epochs):

            self.train(train_loader, epoch)

            metric = self.eval(valid_loader, epoch)

            # ToDo
            # Save every N epochs

            if is_metric_improved(
                best_metric, metric, type(self._loss_func).__name__.lower()
            ):
                self.save_model_state(epoch, best_metric, self._optimizer)
