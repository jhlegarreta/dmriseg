#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityd,
)
from monai.utils import set_determinism  # , first

from dmriseg.dataset.utils import (
    get_model,
    get_test_dataset_cerebparc,
    inference,
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
    num_workers = 4

    # Parameters for the sliding window inference
    patch_size = (192,) * 3
    sw_batch_size = 4

    # Dataset
    dataset = "cerebellum_cerebparc"
    datasets = get_test_dataset_cerebparc(
        str(args.in_test_img_dirname), test_2d=test_2d, batch_size=batch_size
    )
    test_files = datasets[dataset]["test"]

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

    # Get transforms
    test_img_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityd(keys="image"),
        ]
    )

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_img_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=args.out_dirname,
                output_postfix="pred",
                resample=False,
                separate_folder=False,
            ),
        ]
    )

    # Get data loaders
    test_ds = Dataset(data=test_files, transform=test_img_transforms)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    with torch.no_grad():

        for ix, batch_data in enumerate(test_loader):
            inputs = batch_data["image"].to(device)
            batch_data["pred"] = inference(
                inputs,
                model,
                use_amp,
                patch_size=patch_size,
                sw_batch_size=sw_batch_size,
            )
            # batch_data["pred"] = sliding_window_inference(
            #    inputs, patch_size, sw_batch_size, model)
            # ToDo
            # test time augmentation
            [post_transforms(i) for i in decollate_batch(batch_data)]


if __name__ == "__main__":
    main()
