#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from monai.data import (
    CacheDataset,
    DataLoader,
    decollate_batch,
    partition_dataset,
)
from monai.utils import set_determinism  # , first

from dmriseg.dataset.utils import (
    get_model,
    get_test_dataset,
    inference,
    write_dat,
)
from dmriseg.utils import logging_setup

set_determinism(seed=0)
torch.backends.cudnn.benchmark = True

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

logger = logging.getLogger("root")
logger_file_basename = "experiment_logfile.log"


def _set_up_logger(log_fname):

    logging_setup.set_up(log_fname)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_test_img_dirname",
        help="Input training image dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "model_weights_fname",
        help="Model weights filename (*.pth)",
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
    logger_fname = Path(args.out_dirname).joinpath(logger_file_basename)
    _set_up_logger(logger_fname)

    logger.addHandler(logging.StreamHandler())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_2d = False
    use_amp = False  # Does not seem stable, best to disable
    batch_size = 1

    # DDP
    if "LOCAL_RANK" in os.environ:
        logger.info("Setting up DDP...", end="")
        ddp = True
        local_rank = int(os.environ["LOCAL_RANK"])
        # initialize the distributed training process, every GPU runs in a process
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        num_gpus = dist.get_world_size()
        logger.info("done!")
    else:
        ddp = False
        num_gpus = 1
    torch.cuda.set_device(device)

    # Dataset
    dataset = "cerebellum"
    datasets = get_test_dataset(
        str(args.in_test_img_dirname), test_2d=test_2d, batch_size=batch_size
    )
    test_files = datasets[dataset]["test"]

    # Partition per device
    if ddp:
        test_files = partition_dataset(
            data=test_files,
            num_partitions=num_gpus,
            shuffle=False,
            seed=0,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]

    # Get data loaders
    test_loader = DataLoader(
        CacheDataset(
            data=test_files,
            cache_rate=1.0,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    out_channels = datasets[dataset]["n_labels"] + 1

    # Get model
    model_name = "SegResNet16"
    model = get_model(
        model_name,
        out_channels,
        device,
        test_2d=test_2d,
        model_pth=args.model_weights_fname,
    )
    model.eval()

    with torch.no_grad():

        for ix, batch_data in enumerate(test_loader):
            inputs = batch_data["image"].to(device)

            outputs = inference(inputs, model, use_amp)
            # No test time augmentation
            # post_outputs = [post_pred(i) for i in decollate_batch(outputs)]
            post_outputs = decollate_batch(outputs)
            for img_data in post_outputs:
                ref_img_fname = batch_data["image"]
                out_fname = Path(args.out_dirname)
                write_dat(img_data, out_fname, ref_img_fname)


if __name__ == "__main__":
    main()
