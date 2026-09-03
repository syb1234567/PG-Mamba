"""
Evaluation utilities for retinal vessel segmentation.

This module provides binary segmentation metric calculation
and result aggregation functions used during model evaluation.
"""

import math
import cv2
import numpy as np


def calc_result(np_pred: np.ndarray, np_label: np.ndarray, thresh_value=None):
    """
    Calculate segmentation metrics from prediction and label maps.

    Args:
        np_pred (np.ndarray): Predicted probability/logit map.
        np_label (np.ndarray): Ground-truth segmentation map.
        thresh_value (float, optional): Threshold for binary conversion.
            If None, Otsu thresholding is applied.

    Returns:
        dict: Segmentation evaluation metrics.
    """

    # Normalize prediction map for OpenCV thresholding.
    if np_pred.max() != np_pred.min():
        temp = cv2.normalize(np_pred, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    else:
        temp = np.zeros_like(np_pred, dtype="uint8")

    # Convert prediction into binary mask.
    if thresh_value is None:
        _, np_pred_bin = cv2.threshold(
            temp,
            0.0,
            1.0,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    else:
        _, np_pred_bin = cv2.threshold(
            temp,
            thresh_value,
            1.0,
            cv2.THRESH_BINARY,
        )

    # Flatten arrays for pixel-level metric computation.
    np_pred = np_pred_bin.flatten()
    np_label = np_label.flatten()

    # Convert label format to binary mask when necessary.
    if np_label.max() > 1:
        np_label = (np_label > 0).astype(float)

    # Compute confusion matrix.
    FP = np.sum(np.logical_and(np_pred == 1, np_label == 0)).astype(float)
    FN = np.sum(np.logical_and(np_pred == 0, np_label == 1)).astype(float)
    TP = np.sum(np.logical_and(np_pred == 1, np_label == 1)).astype(float)
    TN = np.sum(np.logical_and(np_pred == 0, np_label == 0)).astype(float)

    result = {}
    smooth = 1e-6

    # Classification-based segmentation metrics.
    result["acc"] = (TP + TN) / (FP + FN + TP + TN + smooth)
    result["fdr"] = (FP + smooth) / (FP + TP + smooth)

    sen = (TP + smooth) / (FN + TP + smooth)
    spe = (TN + smooth) / (FP + TN + smooth)

    result["sen"] = sen
    result["spe"] = spe
    result["gmean"] = math.sqrt(sen * spe)

    # Overlap metrics.
    result["iou"] = (TP + smooth) / (FP + FN + TP + smooth)
    result["dice"] = (2.0 * TP + smooth) / (FP + FN + 2.0 * TP + smooth)

    # Compatibility value for training/evaluation logging.
    result["loss"] = 1.0 - result["dice"]

    return result


def avg_result(ls_result):
    """
    Average metric values from multiple evaluation results.

    Args:
        ls_result (list): List of metric dictionaries.

    Returns:
        dict: Mean value of each metric.
    """

    if not ls_result:
        return {}

    total_result = {}

    for result in ls_result:
        for key, value in result.items():
            if key not in total_result:
                total_result[key] = []
            total_result[key].append(value)

    for key, values in total_result.items():
        total_result[key] = float(np.mean(np.array(values)))

    return total_result