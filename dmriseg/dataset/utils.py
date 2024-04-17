# -*- coding: utf-8 -*-

from datetime import datetime
from pathlib import Path

# import cornucopia as cc
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

# from monai.data import DataLoader, Dataset
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference

# from monai.networks.layers import Act, Norm
from monai.networks.nets import SegResNet, UNet
from monai.transforms import (  # RandomizableTransform,; RandSpatialCropd,; RandZoomd,; Resized,; SpatialPadd,
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandFlipd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
)

# from monai.utils import set_determinism


def get_timestamp():
    """Get a time stamp."""
    now = datetime.now()
    return now.strftime("%H:%M:%S %d/%m/%Y")


def filter_labels(inSegArray, verbose=False):
    """Filter labels in a segmentation array, ensuring that the label values are
    contigious.
    """
    unqLabels = torch.unique(inSegArray)
    outSegArray = torch.zeros_like(inSegArray)
    keepCnt = 1
    for unqLabel in unqLabels:
        if unqLabel == 0:
            continue
        outSegArray[inSegArray == unqLabel] = keepCnt
        keepCnt += 1

    return outSegArray


def extract_slice(dat, dim=2, mid_ix=None):
    """Extract slice from volume, along some dimension."""
    if len(dat.shape) == 2:
        return dat
    dat = dat.as_tensor()
    mid = (0.5 * torch.as_tensor(dat.shape[-3:])).round().type(torch.int)
    if isinstance(mid_ix, int):
        mid = mid_ix
    if dim == 0:
        dat = dat[..., mid[0], :, :]
    if dim == 1:
        dat = dat[..., :, mid[1], :]
    if dim == 2:
        dat = dat[..., :, :, mid[2]]
    return dat


class MapLabelsCerebellum(MapTransform):
    """Transform to map Cerebellum data target label identifiers (i.e. those
    that are of interest for prediction purposes) to contiguous values starting
    at one."""

    @staticmethod
    def label_mapping():
        mapping = {1: 1}
        return mapping

    def __init__(self, key_label="label"):
        self.key_label = key_label
        self.lm = MapLabelsCerebellum.label_mapping()

    def __call__(self, data):
        d = dict(data)

        label = d[self.key_label]
        meta = label.meta
        label = label.as_tensor()

        # Make contiguous
        label_contiguous = torch.zeros_like(label)
        for val in self.lm:
            label_contiguous[label == val] = self.lm[val]

        d[self.key_label] = MetaTensor(label_contiguous)
        d[self.key_label].copy_meta_from(meta)

        return d


class MapLabelsCerebellumParc(MapTransform):
    """Transform to map Cerebellum data target label identifiers (i.e. those
    that are of interest for prediction purposes) to contiguous values starting
    at one."""

    @staticmethod
    def label_mapping():
        l0 = list(range(1, 35))
        mapping = {}
        for i0 in l0:
            mapping[i0] = i0
        return mapping

    def __init__(self, key_label="label"):
        self.key_label = key_label
        self.lm = MapLabelsCerebellumParc.label_mapping()

    def __call__(self, data):
        d = dict(data)

        label = d[self.key_label]
        meta = label.meta
        label = label.as_tensor()

        # Make contiguous
        label_contiguous = torch.zeros_like(label)
        for val in self.lm:
            label_contiguous[label == val] = self.lm[val]

        d[self.key_label] = MetaTensor(label_contiguous)
        d[self.key_label].copy_meta_from(meta)

        return d


class SliceFromVolume(MapTransform):
    """Transform to get slice from volume."""

    def __init__(self, do=False, key_image="image", key_label="label"):
        self.key_label = key_label
        self.key_image = key_image
        self.do = do

    def __call__(self, data):
        d = dict(data)
        if not self.do:
            return d

        if self.key_image in d:
            image = d[self.key_image]
            meta = image.meta
            image = extract_slice(image)

            d[self.key_image] = MetaTensor(image)
            d[self.key_image].copy_meta_from(meta)

        if self.key_label in d:
            label = d[self.key_label]
            meta = label.meta
            label = extract_slice(label)
            label = filter_labels(label)

            d[self.key_label] = MetaTensor(label)
            d[self.key_label].copy_meta_from(meta)

        return d


def get_label_cmap(n_labels):
    """Get matplotlib colour map for a label map."""
    unique_values = np.arange(n_labels)
    colors = plt.cm.turbo(np.linspace(0, 1, len(unique_values)))
    cmap = ListedColormap(colors)
    # Create bin edges by adding half a step to unique values
    bin_edges = np.concatenate(
        [unique_values - 0.5, [unique_values[-1] + 0.5]]
    )
    # Create a BoundaryNorm to map unique values to indices in the colormap
    norm = BoundaryNorm(bin_edges, cmap.N, clip=True)

    return cmap, norm


def barplot_dice(ds, loader, figsize=None):
    """Bar plot dice scores."""
    if figsize is None:
        figsize = [18, 5]

    _, ax = plt.subplots(figsize=figsize)
    idx = [i for i in range(len(ds))]
    ax.bar(idx, ds)
    ax.set_xticks(idx)
    ax.set_xticklabels(idx)
    ax.set_ylim([0, 1])
    plt.grid()
    plt.show()
    print(
        f"avg(Dice)={np.asarray(ds).mean():0.2f} with K={len(ds)} and N={len(loader)}"
    )


def write_dat(dat, pth_dat, pth_affine):
    """Write nifti to disk with nibabel."""
    affine = nib.load(pth_affine).affine
    nib.save(nib.Nifti1Image(dat.detach().cpu().numpy(), affine), pth_dat)


def inference(input, model, use_amp, patch_size=None):
    """Run pytorch inference, either full image of sliding window."""

    def _compute(input, model, patch_size):
        if patch_size is not None:
            return sliding_window_inference(
                inputs=input,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )
        else:
            return model(input)

    if use_amp:
        with torch.cuda.amp.autocast():
            return _compute(input, model, patch_size)
    else:
        return _compute(input, model, patch_size)


def get_datasets(
    train_img_dirname,
    train_lmap_dirname,
    valid_img_dirname,
    valid_lmap_dirname,
    test_overfit=False,
    test_2d=False,
    batch_size=1,
):
    """Get a dataset."""
    datasets = {
        "cerebellum": {
            "train": None,
            "val": None,
            "n_labels": None,  # excluding background and GMM classes
            "target_labels": None,
        },
    }

    dataset = "cerebellum"
    train_images = [
        str(f) for f in sorted(Path(train_img_dirname).rglob("*.nii.gz"))
    ]
    train_labels = [
        str(f) for f in sorted(Path(train_lmap_dirname).rglob("*.nii.gz"))
    ]
    train_data = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    valid_images = [
        str(f) for f in sorted(Path(valid_img_dirname).rglob("*.nii.gz"))
    ]
    valid_labels = [
        str(f) for f in sorted(Path(valid_lmap_dirname).rglob("*.nii.gz"))
    ]
    valid_data = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(valid_images, valid_labels)
    ]
    if test_overfit:
        datasets[dataset]["train"] = train_data[:batch_size]
        datasets[dataset]["val"] = train_data[:batch_size]
    else:
        datasets[dataset]["train"] = train_data[0:]
        datasets[dataset]["val"] = valid_data[0:]
    datasets[dataset]["target_labels"] = list(
        MapLabelsCerebellum.label_mapping().values()
    )
    datasets[dataset]["n_labels"] = len(datasets[dataset]["target_labels"])

    return datasets


def get_datasets_cerebparc(
    train_img_dirname,
    train_lmap_dirname,
    valid_img_dirname,
    valid_lmap_dirname,
    test_overfit=False,
    test_2d=False,
    batch_size=1,
):
    """Get a dataset."""
    datasets = {
        "cerebellum_cerebparc": {
            "train": None,
            "val": None,
            "n_labels": None,  # excluding background and GMM classes
            "target_labels": None,
        },
    }

    dataset = "cerebellum_cerebparc"
    train_images = [
        str(f) for f in sorted(Path(train_img_dirname).rglob("*.nii.gz"))
    ]
    train_labels = [
        str(f) for f in sorted(Path(train_lmap_dirname).rglob("*.nii.gz"))
    ]
    train_data = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    valid_images = [
        str(f) for f in sorted(Path(valid_img_dirname).rglob("*.nii.gz"))
    ]
    valid_labels = [
        str(f) for f in sorted(Path(valid_lmap_dirname).rglob("*.nii.gz"))
    ]
    valid_data = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(valid_images, valid_labels)
    ]
    if test_overfit:
        datasets[dataset]["train"] = train_data[:batch_size]
        datasets[dataset]["val"] = train_data[:batch_size]
    else:
        datasets[dataset]["train"] = train_data[0:]
        datasets[dataset]["val"] = valid_data[0:]
    datasets[dataset]["target_labels"] = list(
        MapLabelsCerebellumParc.label_mapping().values()
    )
    datasets[dataset]["n_labels"] = len(datasets[dataset]["target_labels"])

    return datasets


def get_test_dataset(test_img_dirname, test_2d=False, batch_size=1):
    """Get a dataset."""
    datasets = {
        "cerebellum": {
            "test": None,
            "n_labels": None,  # excluding background and GMM classes
            "target_labels": None,
        },
    }

    dataset = "cerebellum"
    test_images = [
        str(f) for f in sorted(Path(test_img_dirname).rglob("*.nii.gz"))
    ]
    test_data = [{"image": image_name} for image_name in test_images]

    datasets[dataset]["test"] = test_data[0:]

    datasets[dataset]["target_labels"] = list(
        MapLabelsCerebellum.label_mapping().values()
    )
    datasets[dataset]["n_labels"] = len(datasets[dataset]["target_labels"])

    return datasets


def get_transforms(
    name,
    target_labels=None,
    num_gmm_classes=10,
    label_mapping=None,
    fov_min_size=None,
    spatial_size=None,
    device="cuda:0",
    patch_size=None,
    n_labels=None,
    test_2d=False,
    synth_params=None,
    test_overfit=False,
):
    """Get transforms for a particular dataset."""
    if synth_params is None:
        if not test_overfit:
            synth_params = {
                "target_labels": target_labels,
                "elastic_steps": 8,
                "rotation": 30,
                "shears": 0.012,
                "zooms": 0.15,
                "elastic": 0.075,
                "elastic_nodes": 10,
                "gmm_fwhm": 10,
                "bias": 7,
                "gamma": 0.6,
                "motion_fwhm": 3,
                "resolution": 8,
                "snr": 10,
                "gfactor": 5,
                "bound": "zeros",
            }
        else:
            synth_params = {
                "target_labels": target_labels,
                "elastic_steps": 8,
                "translations": 0,
                "rotation": 0,
                "shears": 0.0,
                "zooms": 0.0,
                "elastic": 0.0,
                "elastic_nodes": 10,
                "gmm_fwhm": 10,
                "bias": 2,
                "gamma": 0.1,
                "motion_fwhm": 1,
                "resolution": 1,
                "snr": 100,
                "gfactor": 2,
                "bound": "zeros",
            }

    if name == "cerebellum":
        t = {
            "cerebellum": {
                "train": None,
                "val": None,
            },
        }

        t["post"] = Compose(
            [
                Activations(softmax=True),
                AsDiscrete(threshold=0.5),
                # KeepLargestConnectedComponent(),
            ]
        )

        # cerebellum
        # ===========================
        t["cerebellum"]["train"] = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                MapLabelsCerebellum(),
                SliceFromVolume(do=test_2d),
                ResizeWithPadOrCropd(
                    keys=["image", "label"],
                    spatial_size=spatial_size[:2] if test_2d else spatial_size,
                ),
                EnsureTyped(keys="label", dtype=torch.int16, device=device),
                ScaleIntensityd(keys="image"),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            ]
        )
        t["cerebellum"]["val"] = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                MapLabelsCerebellum(),
                SliceFromVolume(do=test_2d),
                ResizeWithPadOrCropd(
                    keys=["label"],
                    spatial_size=spatial_size[:2] if test_2d else spatial_size,
                ),
                EnsureTyped(keys="label", dtype=torch.int16, device=device),
                ScaleIntensityd(keys="image"),
            ]
        )
    elif name == "cerebellum_cerebparc":
        t = {
            "cerebellum_cerebparc": {
                "train": None,
                "val": None,
            },
        }

        t["post"] = Compose(
            [
                Activations(softmax=True),
                AsDiscrete(threshold=0.5),
                # KeepLargestConnectedComponent(),
            ]
        )

        # cerebellum
        # ===========================
        t["cerebellum_cerebparc"]["train"] = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                MapLabelsCerebellumParc(),
                SliceFromVolume(do=test_2d),
                ResizeWithPadOrCropd(
                    keys=["image", "label"],
                    spatial_size=spatial_size[:2] if test_2d else spatial_size,
                ),
                EnsureTyped(keys="label", dtype=torch.int16, device=device),
                ScaleIntensityd(keys="image"),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            ]
        )
        t["cerebellum_cerebparc"]["val"] = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                MapLabelsCerebellumParc(),
                SliceFromVolume(do=test_2d),
                ResizeWithPadOrCropd(
                    keys=["label"],
                    spatial_size=spatial_size[:2] if test_2d else spatial_size,
                ),
                EnsureTyped(keys="label", dtype=torch.int16, device=device),
                ScaleIntensityd(keys="image"),
            ]
        )

    return t


def get_model(name, out_channels, device, test_2d=False, model_pth=None):
    """Get MONAI segmentation model."""

    if name == "SegResNet16":
        net = SegResNet(
            spatial_dims=2 if test_2d else 3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=1,
            out_channels=out_channels,
            dropout_prob=None,
        )
    elif name == "SegResNet32":
        net = SegResNet(
            spatial_dims=2 if test_2d else 3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            in_channels=1,
            out_channels=out_channels,
            dropout_prob=None,
        )
    elif name == "UNet":
        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2,
        )
    else:
        raise NotImplementedError(f"Unknown model name: {name}")

    net = net.to(device)
    if model_pth is not None:
        checkpoint = torch.load(model_pth)
        # Not sure why, but finds unknown keys in state dict if the below is
        # done for UNet
        if name != "UNet":
            checkpoint = {
                k.replace("module.", ""): v for k, v in checkpoint.items()
            }
        net.load_state_dict(checkpoint)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model has {n_parameters:,} trainable parameters")
    return net


def plot_loss_and_metric(axs, loss_values, metric_values, validation_epoch):
    """Plots training loss and metric"""
    x = [i + 1 for i in range(len(loss_values))]
    y = loss_values
    axs[2, 0].plot(x, y, ".", markersize=1)
    axs[2, 0].set_title("Loss")
    axs[2, 0].set_xlabel("epoch")
    x = [validation_epoch * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    axs[2, 1].plot(x, y, ".", markersize=1)
    axs[2, 1].set_title("Metric")
    axs[2, 1].set_xlabel("epoch")


def get_sw_prediction(image, model, patch_size, overlap, blend_mode, tta):
    from torch.nn.functional import softmax

    def get_flip_axes():
        return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

    # Get model prediction
    # Predict on original image
    prediction = sliding_window_inference(
        inputs=image,
        roi_size=patch_size,
        sw_batch_size=1,
        predictor=model,
        overlap=overlap,
        mode=blend_mode,
        device=torch.device("cuda"),
    )
    prediction = softmax(prediction, dim=1)

    # Test time augmentation
    if tta:
        flip_axes = get_flip_axes()
        for i in range(len(flip_axes)):
            axes = flip_axes[i]
            flipped_img = torch.flip(image, dims=axes)
            flipped_pred = sliding_window_inference(
                inputs=flipped_img,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode=blend_mode,
                device=torch.device("cuda"),
            )
            flipped_pred = softmax(flipped_pred, dim=1)
            prediction += torch.flip(flipped_pred, dims=axes)

        prediction /= len(flip_axes) + 1.0

    return prediction


def boxplot_channel_metric(metric_values, metric_name, class_names, epoch):
    assert metric_values.shape[1] == len(class_names)

    figsize = (15, 10)
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(metric_values)
    ticks = [i + 1 for i in range(len(class_names))]
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    ax.set_ylim([0, 1])
    plt.xlabel("Class")
    plt.ylabel(f"{metric_name}")
    plt.title(f"EPOCH={epoch}")
    plt.grid(True)

    return fig


suit_lut = dict(
    {
        0: "Background",
        1: "Left_I_IV",
        2: "Right_I_IV",
        3: "Left_V",
        4: "Right_V",
        5: "Left_VI",
        6: "Vermis_VI",
        7: "Right_VI",
        8: "Left_CrusI",
        9: "Vermis_CrusI",
        10: "Right_CrusI",
        11: "Left_CrusII",
        12: "Vermis_CrusII",
        13: "Right_CrusII",
        14: "Left_VIIb",
        15: "Vermis_VIIb",
        16: "Right_VIIb",
        17: "Left_VIIIa",
        18: "Vermis_VIIIa",
        19: "Right_VIIIa",
        20: "Left_VIIIb",
        21: "Vermis_VIIIb",
        22: "Right_VIIIb",
        23: "Left_IX",
        24: "Vermis_IX",
        25: "Right_IX",
        26: "Left_X",
        27: "Vermis_X",
        28: "Right_X",
        29: "Left_Dentate",
        30: "Right_Dentate",
        31: "Left_Interposed",
        32: "Right_Interposed",
        33: "Left_Fastigial",
        34: "Right_Fastigial",
    }
)


def get_suit_classnames():

    return list(suit_lut.values())
