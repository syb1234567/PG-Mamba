"""
Evaluation metrics for PG-Mamba retinal vessel segmentation.

Implemented metrics:
- Soft Dice
- Dice / IoU
- clDice topology metric
- Brier score
- MAE
- ECE calibration metrics
- Caliber-stratified vessel analysis

Inputs:
- Prediction maps: probability values in [0, 1]
- Ground truth: soft segmentation labels

This module is designed for evaluation of soft-label
ultra-widefield SS-OCTA vessel segmentation.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def soft_dice(prob, gt, eps=1e-6):
    """
    Compute soft Dice score using probability maps and soft labels.
    """
    p = prob.ravel()
    g = gt.ravel()
    intersection = 2.0 * (p * g).sum()
    union = p.sum() + g.sum()

    return float((intersection + eps) / (union + eps))


def brier_score(prob, gt):
    """
    Compute Brier score for probability calibration.
    """
    return float(np.mean((prob.ravel() - gt.ravel()) ** 2))


def mae_score(prob, gt):
    """
    Compute mean absolute error between prediction and label.
    """
    return float(np.mean(np.abs(prob.ravel() - gt.ravel())))


def binary_dice_iou(prob, gt, thr=0.5, eps=1e-6):
    """
    Compute binary Dice and IoU after thresholding.
    """
    pred = (prob > thr).astype(np.float64)
    g = (gt > thr).astype(np.float64)
    tp = float((pred * g).sum())
    fp = float((pred * (1 - g)).sum())
    fn = float(((1 - pred) * g).sum())
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return float(dice), float(iou)


def _cl_score(v, s):
    """
    Compute skeleton overlap score.
    """
    if s.sum() == 0:
        return 0.0

    return float((v * s).sum() / s.sum())

def cl_dice(prob, gt, thr=0.5, eps=1e-6):
    """
    Compute topology-aware clDice metric.
    """
    pred = (prob > thr).astype(np.uint8)
    g = (gt > thr).astype(np.uint8)
    if pred.sum() == 0 or g.sum() == 0:
        return 0.0
    s_pred = skeletonize(pred).astype(np.float64)
    s_gt = skeletonize(g).astype(np.float64)
    tprec = _cl_score(g.astype(np.float64), s_pred)
    tsens = _cl_score(pred.astype(np.float64), s_gt)
    if tprec + tsens < eps:
        return 0.0

    return float(2 * tprec * tsens / (tprec + tsens))

def caliber_stratified_metrics(
    prob,
    gt,
    thr=0.5,
    small_max=1.6,
    mid_max=3.2,
    eps=1e-6,
):
    """
    Evaluate vessel segmentation performance according to local vessel radius.
    Vessel caliber is estimated using distance transform on the ground truth mask.
    Args:
        small_max (float):
            Upper radius threshold for small vessels.
        mid_max (float):
            Upper radius threshold for medium vessels.
    Returns:
        Dictionary containing recall and clDice for each caliber group.
    """
    pred_bin = (prob > thr).astype(np.uint8)
    gt_bin = (gt > thr).astype(np.uint8)
    names = ("small", "mid", "large")
    if gt_bin.sum() == 0:
        result = {
            f"recall_{name}": float("nan")
            for name in names
        }
        result.update({
            f"cldice_{name}": float("nan")
            for name in names
        })
        return result
    # Local vessel radius estimation.
    dist = ndimage.distance_transform_edt(gt_bin)
    masks = {
        "small": (gt_bin == 1)
        & (dist > 0)
        & (dist < small_max),

        "mid": (gt_bin == 1)
        & (dist >= small_max)
        & (dist < mid_max),

        "large": (gt_bin == 1)
        & (dist >= mid_max),
    }
    result = {}
    for name, gmask in masks.items():
        gmask_u8 = gmask.astype(np.uint8)
        if gmask_u8.sum() == 0:
            result[f"recall_{name}"] = float("nan")
            result[f"cldice_{name}"] = float("nan")
            continue
        # Pixel recall within each vessel caliber group.
        tp = float((pred_bin * gmask_u8).sum())
        result[f"recall_{name}"] = float(
            tp / (gmask_u8.sum() + eps)
        )
        # Skeleton-based topology recall.
        g_skel = skeletonize(gmask_u8).astype(np.float64)
        if g_skel.sum() == 0:
            result[f"cldice_{name}"] = float("nan")
        else:
            result[f"cldice_{name}"] = float(
                (pred_bin.astype(np.float64) * g_skel).sum()
                / g_skel.sum()
            )
    return result
def vessel_radius_histogram(gt, thr=0.5, bins=None):
    """
    Compute vessel radius distribution from ground-truth segmentation.
    Args:
        gt (np.ndarray): Soft segmentation label.
        thr (float): Threshold for binary conversion.
        bins (array, optional): Histogram bins.
    Returns:
        tuple:
            counts,
            bin_edges,
            percentile statistics
    """
    gt_bin = (gt > thr).astype(np.uint8)

    if gt_bin.sum() == 0:
        return None
    dist = ndimage.distance_transform_edt(gt_bin)
    radii = dist[gt_bin == 1].ravel()
    if bins is None:
        bins = np.arange(
            0,
            max(radii.max() + 1, 2),
            0.5,
        )
    counts, edges = np.histogram(
        radii,
        bins=bins,
    )
    percentile = {
        f"p{q}": float(np.percentile(radii, q))
        for q in (25, 50, 75, 90, 95)
    }
    percentile["mean"] = float(radii.mean())
    percentile["max"] = float(radii.max())
    return counts, edges, percentile

def _ece_in_region(prob, hard_gt, region_mask, n_bins=15):
    """
    Compute expected calibration error within a specific region.
    """
    p = prob[region_mask].ravel()
    y = hard_gt[region_mask].ravel()

    if p.size == 0:
        return float("nan")
    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )
    ece = 0.0
    n_samples = p.size
    for i in range(n_bins):
        low = bins[i]
        high = bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        count = mask.sum()
        if count == 0:
            continue
        ece += (
            count / n_samples
        ) * abs(
            y[mask].mean()
            - p[mask].mean()
        )

    return float(ece)

def boundary_band(hard_gt, width=5):
    """
    Generate boundary region around binary segmentation mask.
    """
    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(
        hard_gt,
        structure=structure,
        iterations=width,
    )
    eroded = ndimage.binary_erosion(
        hard_gt,
        structure=structure,
        iterations=width,
    )
    return dilated & (~eroded)

def ece_scores(
    prob,
    gt,
    thr=0.5,
    n_bins=15,
    band_width=5,
):
    """
    Compute boundary and full-image calibration error.
    """
    hard_gt = (gt > thr).astype(np.uint8)
    ece_full = _ece_in_region(
        prob,
        hard_gt,
        np.ones_like(
            hard_gt,
            dtype=bool,
        ),
        n_bins,
    )
    boundary = boundary_band(
        hard_gt,
        width=band_width,
    )
    if boundary.sum() == 0:
        ece_boundary = float("nan")
    else:
        ece_boundary = _ece_in_region(
            prob,
            hard_gt,
            boundary,
            n_bins,
        )
    return float(ece_boundary), float(ece_full)


def calc_metrics(
    prob,
    gt,
    thr=0.5,
    n_bins=15,
    band_width=5,
    skip_cldice=False,
    caliber=False,
    small_max=1.6,
    mid_max=3.2,
):
    """
    Calculate all segmentation evaluation metrics.
    Args:
        prob:
            Prediction probability map in [0,1].
        gt:
            Soft segmentation label.
        thr:
            Threshold for binary metrics.
        caliber:
            Whether to compute caliber-stratified metrics
    Returns:
        Dictionary containing segmentation metrics.
    """
    prob = np.asarray(
        prob,
        dtype=np.float64,
    )
    gt = np.asarray(
        gt,
        dtype=np.float64,
    )
    if prob.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: {prob.shape} vs {gt.shape}"
        )
    if prob.min() < -1e-6 or prob.max() > 1 + 1e-6:
        raise ValueError(
            "Prediction values must be within [0,1]. "
            "Apply sigmoid before metric calculation."
        )
    dice, iou = binary_dice_iou(
        prob,
        gt,
        thr,
    )
    ece_boundary, ece_full = ece_scores(
        prob,
        gt,
        thr,
        n_bins,
        band_width,
    )
    result = {
        "brier": brier_score(prob, gt),
        "sdice": soft_dice(prob, gt),
        "ece": ece_boundary,
        "ece_full": ece_full,
        "cldice": (
            float("nan")
            if skip_cldice
            else cl_dice(prob, gt, thr)
        ),
        "dice": dice,
        "iou": iou,
        "mae": mae_score(prob, gt),
    }

    if caliber:
        result.update(
            caliber_stratified_metrics(
                prob,
                gt,
                thr,
                small_max,
                mid_max,
            ))
    return result

def avg_metrics(list_of_dict):
    """
    Average metric values from multiple samples.
    NaN values are ignored during averaging.
    """
    if not list_of_dict:
        return {}

    keys = list_of_dict[0].keys()

    return {
        key: float(
            np.nanmean(
                [
                    item[key]
                    for item in list_of_dict
                ]
            )
        )
        for key in keys
    }

def selection_score(m, lam=1.0):
    """
    Compute validation score for model selection.
    Higher values indicate better performance.
    Score:
        S-Dice - lambda * Brier score
    """

    return float(
        m["sdice"]
        - lam * m["brier"]
    )