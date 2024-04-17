#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from monai.data import (
    CacheDataset,
    DataLoader,
    decollate_batch,
    partition_dataset,
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism  # first
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from dmriseg.dataset.utils import (
    boxplot_channel_metric,
    extract_slice,
    get_datasets_cerebparc,
    get_label_cmap,
    get_model,
    get_suit_classnames,
    get_timestamp,
    get_transforms,
    inference,
    plot_loss_and_metric,
)

# from dmriseg.utils import logging_setup

set_determinism(seed=0)
torch.backends.cudnn.benchmark = True

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

logger = logging.getLogger("root")
logger_file_basename = "experiment_logfile.log"


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    # parser.add_argument(
    #    "in_train_fa_dirname",
    #    help="Input training FA image dirname (*.nii.gz)",
    #    type=Path,
    # )
    # parser.add_argument(
    #    "in_train_md_dirname",
    #    help="Input training MD image dirname (*.nii.gz)",
    #    type=Path,
    # )
    parser.add_argument(
        "in_train_img_dirname",
        help="Input training image dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "in_train_labelmap_dirname",
        help="Input training data dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "in_valid_img_dirname",
        help="Input validation image data dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "in_valid_labelmap_dirname",
        help="Input validation labelmap data dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "out_dirname",
        help="Output dirname",
        type=Path,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Set up logger
    # logger_fname = Path(args.out_dirname).joinpath(logger_file_basename)
    # _set_up_logger(logger_fname)

    # logger.addHandler(logging.StreamHandler())

    # Parameters
    dataset = "cerebellum_cerebparc"
    dout = str(args.out_dirname)
    meta_pth = os.path.join(dout, "meta_data.pkl")
    model_pth = os.path.join(dout, "model_latest.pth")
    test_2d = False
    test_overfit = False
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # steps = 2*int(1e4)  # 2*int(1e6)
    use_amp = False  # Does not seem stable, best to disable
    # validation_steps = 100  # 5000
    spatial_size = (192,) * 3
    patch_size = (192,) * 3
    lr = 1.0e-4
    # Make output folder
    if not os.path.isdir(dout):
        os.makedirs(dout, exist_ok=True)
    elif os.path.isdir(dout) and test_overfit:
        shutil.rmtree(dout)
    print(dout)
    model_pth = None if not os.path.exists(model_pth) else model_pth
    log_dir = dout + "/runs"
    writer = SummaryWriter(log_dir)

    # DDP
    if "LOCAL_RANK" in os.environ:
        print("Setting up DDP...", end="")
        ddp = True
        local_rank = int(os.environ["LOCAL_RANK"])
        # initialize the distributed training process, every GPU runs in a process
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        num_gpus = dist.get_world_size()
        print("done!")
    else:
        ddp = False
        num_gpus = 1
    torch.cuda.set_device(device)

    # Remove background
    classnames = get_suit_classnames()[1:]

    # Get files
    datasets = get_datasets_cerebparc(
        str(args.in_train_img_dirname),
        str(args.in_train_labelmap_dirname),
        str(args.in_valid_img_dirname),
        str(args.in_valid_labelmap_dirname),
        test_overfit=test_overfit,
        test_2d=test_2d,
        batch_size=batch_size,
    )
    train_files = datasets[dataset]["train"]
    val_files = datasets[dataset]["val"]
    # train_files = train_files[:8]
    # val_files = val_files[:8]
    num_train = len(train_files)
    num_val = len(val_files)
    # Set when to run validation
    # steps_per_epoch = num_train
    validation_interval = 1  # round(validation_steps/steps_per_epoch)

    # Partition per device
    if ddp:
        train_files = partition_dataset(
            data=train_files,
            num_partitions=num_gpus,
            shuffle=False,
            seed=0,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]
        val_files = partition_dataset(
            data=val_files,
            num_partitions=num_gpus,
            shuffle=False,
            seed=0,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]

    # Get transforms
    transforms = get_transforms(
        dataset,
        target_labels=datasets[dataset]["target_labels"],
        device=device,
        n_labels=datasets[dataset]["n_labels"],
        test_2d=test_2d,
        spatial_size=spatial_size,
        test_overfit=test_overfit,
        patch_size=patch_size,
    )

    # Get data loaders
    train_loader = DataLoader(
        CacheDataset(
            data=train_files,
            transform=transforms[dataset]["train"],
            cache_rate=1.0,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CacheDataset(
            data=val_files,
            transform=transforms[dataset]["val"],
            cache_rate=1.0,
        ),
        batch_size=1,
        shuffle=False,
    )
    max_epochs = 200  # steps // num_train

    # Get model
    model_name = "SegResNet16"
    # model_name = "UNet"
    out_channels = datasets[dataset]["n_labels"] + 1
    print(f"Number of output channels: {out_channels}")
    model = get_model(
        model_name, out_channels, device, test_2d=test_2d, model_pth=model_pth
    )

    if ddp:
        model = DistributedDataParallel(
            model,
            device_ids=[device],
            find_unused_parameters=False,
            #         broadcast_buffers=False,
        )

    # Get loss, optimiser and metrics
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        squared_pred=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=math.sqrt(batch_size * num_gpus) * lr
    )
    # ToDo
    # Will probably want reduction="mean_channel" and then do the mean myself
    # so that I know which are the best/worst labels at each epoch. Or "none" to
    # have a data point per label, per image so that I can plot a boxplot. Have
    # two instances if necessary
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_c = DiceMetric(include_background=False, reduction="none")
    dice_metric_batch = DiceMetric(
        include_background=False, reduction="mean_batch"
    )

    # Add LR scheduler
    factor = 0.5
    patience = 10
    min_lr = 1e-8
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
    )

    # Various
    post_pred = transforms["post"]
    scaler = torch.cuda.amp.GradScaler()
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    cmap, norm = get_label_cmap(n_labels=out_channels)

    dim = 0  # sagittal; dim = 1  # coronal; dim = 2  # axial

    # Load/init meta data dict
    if os.path.isfile(meta_pth) and model_pth is not None:
        with open(meta_pth, "rb") as f:
            meta_data = pickle.load(f)
        if "epoch" not in meta_data:
            meta_data["epoch"] = 0
        if "best_metric" not in meta_data:
            meta_data["best_metric"] = -1
        if "loss_values" not in meta_data:
            meta_data["loss_values"] = []
        if "metric_values" not in meta_data:
            meta_data["metric_values"] = []
    else:
        meta_data = dict(
            epoch=0,
            best_metric=-1,
            loss_values=[],
            metric_values=[],
        )

    # Train/Eval
    print("-" * 64)
    print("Starting training from epoch {}".format(meta_data["epoch"]))
    print("-" * 64)

    best_metric = meta_data["best_metric"]
    loss_values = meta_data["loss_values"]
    metric_values = meta_data["metric_values"]

    for epoch in range(meta_data["epoch"], max_epochs):

        # Save latest meta data
        meta_data["epoch"] = epoch
        meta_data["best_metric"] = best_metric
        meta_data["loss_values"] = loss_values
        meta_data["metric_values"] = metric_values
        with open(meta_pth, "wb") as f:
            pickle.dump(meta_data, f)

        # Train
        # ----------------------
        model.train()
        epoch_loss = 0
        # For printing a random debug figure
        save_fig_ix = np.random.randint(0, len(train_loader))

        step_epoch = 0
        for ix, batch_data in enumerate(train_loader):

            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            step_epoch += 1

            batch_ix = np.random.randint(0, inputs.shape[0])

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (epoch % validation_interval == 0) and (save_fig_ix == ix):
                with torch.no_grad():
                    post_outputs = [
                        post_pred(i) for i in decollate_batch(outputs)
                    ]
                axs[0, 0].imshow(
                    extract_slice(inputs[batch_ix][0], dim=dim).cpu().numpy(),
                    cmap="gray",
                )
                axs[0, 0].set_title("train image")
                axs[0, 0].axis("off")
                axs[0, 1].imshow(
                    extract_slice(labels[batch_ix][0], dim=dim).cpu().numpy(),
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                )
                axs[0, 1].set_title("train label")
                axs[0, 1].axis("off")
                axs[0, 2].imshow(
                    extract_slice(
                        post_outputs[batch_ix].argmax(dim=0), dim=dim
                    )
                    .cpu()
                    .numpy(),
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                )
                axs[0, 2].set_title("train prediction")
                axs[0, 2].axis("off")

        epoch_loss /= step_epoch
        print(
            f"EPOCH={epoch + 1:{' '}{len(str(max_epochs))}}/{max_epochs} |__LOSS (N={num_train})={epoch_loss:.4f}  |  {get_timestamp()}"
        )
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.flush()

        loss_values.append(epoch_loss)

        if epoch % validation_interval == 0 or epoch == max_epochs - 1:
            # Eval
            # ----------------------
            model.eval()
            save_fig_ix = np.random.randint(0, len(val_loader))

            valid_loss = 0
            with torch.no_grad():

                for ix, batch_data in enumerate(val_loader):

                    inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )
                    batch_ix = np.random.randint(0, inputs.shape[0])

                    outputs = inference(inputs, model, use_amp)
                    loss = loss_function(outputs, labels)
                    valid_loss += loss.item()

                    post_outputs = [
                        post_pred(i) for i in decollate_batch(outputs)
                    ]
                    dice_metric(y_pred=post_outputs, y=labels)
                    dice_metric_c(y_pred=post_outputs, y=labels)
                    dice_metric_batch(y_pred=post_outputs, y=labels)

                    if save_fig_ix == ix:
                        axs[1, 0].imshow(
                            extract_slice(inputs[batch_ix][0], dim=dim)
                            .cpu()
                            .numpy(),
                            cmap="gray",
                        )
                        axs[1, 0].set_title("val image")
                        axs[1, 0].axis("off")
                        axs[1, 1].imshow(
                            extract_slice(labels[batch_ix][0], dim=dim)
                            .cpu()
                            .numpy(),
                            cmap=cmap,
                            norm=norm,
                            interpolation="nearest",
                        )
                        axs[1, 1].set_title("val label")
                        axs[1, 1].axis("off")
                        axs[1, 2].imshow(
                            extract_slice(
                                post_outputs[batch_ix].argmax(dim=0), dim=dim
                            )
                            .cpu()
                            .numpy(),
                            cmap=cmap,
                            norm=norm,
                            interpolation="nearest",
                        )
                        axs[1, 2].set_title("val prediction")
                        axs[1, 2].axis("off")

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                metric_c = dice_metric_c.aggregate()
                metric_batch = dice_metric_batch.aggregate()
                # reset the status for next validation round
                dice_metric.reset()
                dice_metric_c.reset()
                dice_metric_batch.reset()

                metric_values.append(metric)

                print(
                    f"EPOCH={epoch + 1:{' '}{len(str(max_epochs))}}/{max_epochs} |____METRIC (N={num_val})={metric:.4f}"
                )

                writer.add_scalar("Metric/eval", metric, epoch)
                writer.flush()
                for i in range(0, len(metric_batch), 10):
                    print(
                        " " * (13 + len(str(max_epochs)))
                        + "|______"
                        + ", ".join(
                            [
                                f"{k + i:3.0f}={v:0.3f}".format(k, v)
                                for k, v in enumerate(metric_batch[i : i + 10])
                            ]
                        )
                    )

                if metric > best_metric:
                    best_metric = metric
                    torch.save(
                        model.state_dict(),
                        os.path.join(dout, "model_best.pth"),
                    )

                plot_loss_and_metric(
                    axs, loss_values, metric_values, validation_interval
                )
                fig.suptitle(
                    f"EPOCH={epoch}, LOSS={epoch_loss:.4f}, METRIC={metric:.4f}"
                )
                fig.tight_layout()
                fig.savefig(os.path.join(dout, "outputs.png"))
                plt.close(fig)

                _fig = boxplot_channel_metric(
                    metric_c.cpu().numpy(), "Dice", classnames, epoch
                )
                _fig.savefig(os.path.join(dout, "dice_channel_boxplot.png"))
                plt.close(_fig)

            scheduler.step(valid_loss)
            _lr = scheduler._last_lr
            print(f"Learning rate is: {_lr}")

        torch.save(model.state_dict(), os.path.join(dout, "model_latest.pth"))
        writer.close()


if __name__ == "__main__":
    main()
