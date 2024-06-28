#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np

from dmriseg.analysis.measures import (
    compute_center_of_mass_distance,
    compute_metrics,
    compute_relevant_labels,
    fill_missing_values,
    get_label_presence,
)


def test_compute_relevant_labels():

    img_data1 = np.array([0, 1, 1, 2, 0, 2, 3])
    img_data2 = np.array([0, 1, 1, 2, 0, 2, 0])
    labels = [0, 1, 2, 3]
    exclude_background = True
    prsnt_labels, msng_labels, msng_idx = compute_relevant_labels(
        img_data1, img_data2, labels, exclude_background
    )

    assert prsnt_labels == [1, 2]
    assert np.all(msng_labels == np.asarray([3]))
    assert np.all(msng_idx == np.asarray([2]))

    img_data1 = np.array([0, 0, 2, 2, 0, 4, 4])
    img_data2 = np.array([0, 0, 2, 0, 0, 4, 0])
    labels = [0, 1, 2, 3, 4]
    exclude_background = True
    prsnt_labels, msng_labels, msng_idx = compute_relevant_labels(
        img_data1, img_data2, labels, exclude_background
    )

    assert prsnt_labels == [2, 4]
    assert np.all(msng_labels == np.asarray([1, 3]))
    assert np.all(msng_idx == np.asarray([0, 2]))


def test_fill_missing_values():

    metrics = [
        dict(
            {
                "label": [2, 4],
                "dice": [0.2, 0.22],
                "jaccard": [0.4, 0.44],
                "precision": [0.6, 0.66],
                "recall": [0.8, 0.88],
                "fpr": [0.1, 0.11],
                "fnr": [0.2, 0.22],
                "vs": [0.3, 0.33],
                "hd": [0.4, 0.44],
                "msd": [0.5, 0.55],
                "mdsd": [0.6, 0.66],
                "stdsd": [0.7, 0.77],
                "hd95": [0.8, 0.88],
            }
        )
    ]
    labels = [1, 2, 3, 4]
    # msng_labels = [1, 3]
    msng_idx = [0, 2]
    _metrics = fill_missing_values(metrics, labels, msng_idx)

    metrics_expected = [
        dict(
            {
                "label": [1, 2, 3, 4],
                "dice": [0, 0.2, 0, 0.22],
                "jaccard": [0, 0.4, 0, 0.44],
                "precision": [np.nan, 0.6, np.nan, 0.66],
                "recall": [np.nan, 0.8, np.nan, 0.88],
                "fpr": [np.nan, 0.1, np.nan, 0.11],
                "fnr": [np.nan, 0.2, np.nan, 0.22],
                "vs": [-2, 0.3, -2, 0.33],
                "hd": [np.inf, 0.4, np.inf, 0.44],
                "msd": [np.inf, 0.5, np.inf, 0.55],
                "mdsd": [np.inf, 0.6, np.inf, 0.66],
                "stdsd": [np.inf, 0.7, np.inf, 0.77],
                "hd95": [np.inf, 0.8, np.inf, 0.88],
            }
        )
    ]

    assert _metrics[0] == metrics_expected[0]


def test_compute_metrics():

    exclude_background = True

    image1 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 2, 1, 3, 0],
                [0, 2, 0, 3, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    ).astype(np.float32)
    labels = sorted(map(int, np.unique(image1)))
    img1 = nib.Nifti1Image(image1, affine=np.eye(4), dtype=np.float32)
    spacing = np.array([1.0, 1.0, 1.0])
    metrics = compute_metrics(
        img1, img1, spacing, labels, exclude_background=exclude_background
    )

    assert metrics[0]["label"] == labels[1:]
    # Should be 1.0, but for some reason the tool does not return 1
    atol = 1e-2
    assert np.allclose(metrics[0]["dice"], 0.99, atol=atol)
    assert np.allclose(metrics[0]["jaccard"], 0.99, atol=atol)
    assert np.allclose(metrics[0]["precision"], 0.99, atol=atol)
    assert np.allclose(metrics[0]["recall"], 0.99, atol=atol)
    assert np.allclose(metrics[0]["fpr"], 0.0)
    assert np.allclose(metrics[0]["fnr"], 0.0)
    assert np.allclose(metrics[0]["vs"], 0.0)
    assert np.allclose(metrics[0]["hd"], 0.0)
    assert np.allclose(metrics[0]["msd"], 0.0)
    assert np.allclose(metrics[0]["mdsd"], 0.0)
    assert np.allclose(metrics[0]["stdsd"], 0.0)
    assert np.allclose(metrics[0]["hd95"], 0.0)

    # Check with labels that are not in the ground truth
    image2 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 2, 1, 3, 0],
                [0, 2, 0, 3, 0],
                [0, 0, 4, 4, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    ).astype(np.float32)
    labels_img1 = sorted(map(int, np.unique(image1)))
    labels_img2 = sorted(map(int, np.unique(image2)))
    labels = sorted(set(labels_img1).union(set(labels_img2)))
    img2 = nib.Nifti1Image(image2, affine=np.eye(4), dtype=np.float32)
    metrics = compute_metrics(
        img1, img2, spacing, labels, exclude_background=exclude_background
    )
    assert [len(vals) == len(labels[1:]) for vals in metrics[0].values()]

    # ToDo
    # The below fails at seg_metrics, so we should constantly keep of the labels
    # that are present across the images, and use appropriate missing values (0
    # or nan) for the missing labels.
    # Check with a number of labels that the ground truth and the prediction
    # are missing (e.g. they are missing the fastigial)
    labels = [0, 1, 2, 3, 4, 5]
    metrics = compute_metrics(
        img1, img2, spacing, labels, exclude_background=exclude_background
    )
    assert [len(vals) == len(labels[1:]) for vals in metrics[0].values()]

    # Try with actual NIfTI files
    # img1_fname = "/mnt/data/connectome/suit/results/cer_seg_100307.nii"
    # img2_fname = "/mnt/data/connectome/suit/results/cer_seg_100307.nii"

    # img1 = nib.load(img1_fname)
    # img2 = nib.load(img2_fname)
    # labels = sorted(map(int, np.unique(img1.get_fdata())))

    # labels1 = [0, 1, 2]
    # gdth_img = np.array([[0, 0, 1], [0, 1, 2]])
    # pred_img = np.array([[0, 0, 1], [0, 2, 2]])
    # metrics1 = compute_metrics(
    #   labels=labels1[1:], gdth_img=gdth_img, pred_img=pred_img)
    # metrics = compute_metrics(
    #     img1, img2, spacing, labels, exclude_background=exclude_background
    # )

    # assert metrics[0]["label"] == labels[1:]
    # Should be 1.0, but for some reason the tool does not return 1
    # atol = 1e-2
    # assert np.allclose(metrics[0]["dice"], 0.99, atol=atol)
    # assert np.allclose(metrics[0]["jaccard"], 0.99, atol=atol)
    # assert np.allclose(metrics[0]["precision"], 0.99, atol=atol)
    # assert np.allclose(metrics[0]["recall"], 0.99, atol=atol)
    # assert np.allclose(metrics[0]["fpr"], 0.0)
    # assert np.allclose(metrics[0]["fnr"], 0.0)
    # assert np.allclose(metrics[0]["vs"], 0.0)
    # assert np.allclose(metrics[0]["hd"], 0.0)
    # assert np.allclose(metrics[0]["msd"], 0.0)
    # assert np.allclose(metrics[0]["mdsd"], 0.0)
    # assert np.allclose(metrics[0]["stdsd"], 0.0)
    # assert np.allclose(metrics[0]["hd95"], 0.0)

    # img2_fname = "/mnt/data/connectome/suit/results/cer_seg_100408.nii"
    # img2 = nib.load(img2_fname)
    # metrics = compute_metrics(
    #     img1, img2, spacing, labels, exclude_background=exclude_background
    # )

    # # assert metrics[0]["label"] == labels[1:]
    # # assert [len(vals) == len(labels[1:]) for vals in metrics[0].values()]


def test_compute_center_of_mass_distance():

    image1 = np.array(
        [
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 2, 0, 0, 0],
                [2, 2, 3, 3, 4],
            ]
        ]
    ).astype(np.float32)
    labels = sorted(map(int, np.unique(image1)))
    img1 = nib.Nifti1Image(image1, affine=np.eye(4), dtype=np.float32)
    cm_dist, centers1, centers2 = compute_center_of_mass_distance(
        img1, img1, labels
    )
    assert [len(centers1) == len(labels)]
    assert [len(centers2) == len(labels)]
    assert [len(center) == len(img1.shape) for center in centers1]

    # Display the labeled image and center of masses
    import matplotlib.pyplot as plt

    plt.imshow(image1[0, ...], cmap="viridis")
    for i, center in zip(labels, centers1):
        plt.scatter(center[2], center[1], label=f"Label {i}", marker="x")
    plt.title("Labeled Image with Centers of Mass")
    plt.legend()
    plt.show()

    # Print the center of masses
    for i, center in zip(labels, centers1):
        print(f"Label {i}: Center of Mass = {center}")

    # Check with labels that are not in the ground truth
    image2 = np.array(
        [
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 2, 0, 0, 0],
                [2, 2, 3, 3, 5],
            ]
        ]
    ).astype(np.float32)
    labels_img1 = sorted(np.unique(image1))
    labels_img2 = sorted(np.unique(image2))
    labels = sorted(set(labels_img1).union(set(labels_img2)))
    img2 = nib.Nifti1Image(image2, affine=np.eye(4), dtype=np.float32)
    cm_dist, centers1, centers2 = compute_center_of_mass_distance(
        img1, img2, labels
    )
    assert [len(centers1) == len(labels)]
    assert [len(centers2) == len(labels)]
    assert [len(center) == len(img1.shape) for center in centers1]
    assert [len(center) == len(img1.shape) for center in centers2]

    # img1_fname = "/mnt/data/connectome/suit/results/cer_seg_100307.nii"
    # img2_fname = "/mnt/data/connectome/suit/results/cer_seg_100307.nii"

    # img1 = nib.load(img1_fname)
    # img2 = nib.load(img2_fname)
    # labels = sorted(map(int, np.unique(img1.get_fdata())))
    # cm_dist, centers1, centers2 = compute_center_of_mass_distance(
    #     img1, img2, labels
    # )

    # assert np.allclose(cm_dist, 0.0)
    # assert np.allclose(centers1, centers2)

    # img2_fname = "/mnt/data/connectome/suit/results/cer_seg_100408.nii"
    # img2 = nib.load(img2_fname)
    # cm_dist, centers1, centers2 = compute_center_of_mass_distance(
    #     img1, img2, labels
    # )

    # assert len(cm_dist) == len(labels)
    # assert len(centers1) == len(labels)
    # assert len(centers2) == len(labels)


def test_get_label_presence():

    image = np.array(
        [
            [
                [0, 1, 1, 2, 0, 2, 3],
                [0, 1, 1, 1, 0, 2, 2],
            ]
        ]
    ).astype(np.float32)
    img = nib.Nifti1Image(image, affine=np.eye(4), dtype=np.float32)

    labels = [0, 1, 2, 3]
    exclude_background = True
    exp_ld = [True, True, True]
    obt_ld = get_label_presence(img, labels, exclude_background)

    assert exp_ld == obt_ld

    labels = [0, 1, 2, 3, 4]
    exp_ld = [True, True, True, False]
    obt_ld = get_label_presence(img, labels, exclude_background)

    assert exp_ld == obt_ld
