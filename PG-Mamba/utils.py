"""
Utility functions for PG-Mamba training and evaluation.

Includes sliding-window inference, dataset traversal, metric aggregation,
and optional per-image prediction export.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics import avg_metrics, calc_metrics


def predict_sliding_window(model, image, tile_size=512, overlap=0.5, num_classes=1):
    """Perform sliding-window inference for large images."""
    b, c, h, w = image.shape
    stride = int(tile_size * (1 - overlap))

    probs = torch.zeros((b, num_classes, h, w), device=image.device)
    counts = torch.zeros((b, num_classes, h, w), device=image.device)

    rows = list(range(0, h - tile_size + 1, stride))
    cols = list(range(0, w - tile_size + 1, stride))

    if rows[-1] + tile_size < h:
        rows.append(h - tile_size)
    if cols[-1] + tile_size < w:
        cols.append(w - tile_size)

    model.eval()
    with torch.no_grad():
        for y in rows:
            for x in cols:
                patch = image[:, :, y:y + tile_size, x:x + tile_size]
                prob = model(patch)
                if isinstance(prob, list):
                    prob = prob[0]

                probs[:, :, y:y + tile_size, x:x + tile_size] += prob
                counts[:, :, y:y + tile_size, x:x + tile_size] += 1

    return probs / counts


def traverseDataset(
    model: nn.Module, loader: DataLoader, description, device, funcLoss,
    log_writer: SummaryWriter, log_section, optimizer=None, show_result=False,
    thresh_value=None, skip_cldice=False, caliber=False, return_per_image=False
):
    """Run training, validation, or testing over a dataset."""
    is_training = optimizer is not None
    time_start = time.time()
    per_image = []

    with tqdm(loader, unit="batch") as tepoch:
        total_loss = 0
        ls_eval_result = []
        model.train(is_training)

        for i, batch_data in enumerate(tepoch):
            name = batch_data[0]
            data = batch_data[1].to(device)
            label = batch_data[2].to(device)
            original_size = None

            if not is_training and len(batch_data) > 3:
                try:
                    size_tensor = batch_data[3]
                    if size_tensor.dim() == 2:
                        org_h = size_tensor[0, 0].item()
                        org_w = size_tensor[0, 1].item()
                    else:
                        org_h = size_tensor[0].item()
                        org_w = size_tensor[1].item()
                    original_size = (int(org_h), int(org_w))
                except Exception:
                    original_size = None

            tepoch.set_description(description)
            eval_result = {}

            if is_training:
                pred = model(data)
                if isinstance(pred, list):
                    pred = pred[0]

                loss = funcLoss(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()

            else:
                with torch.no_grad():
                    _, _, h, w = data.shape

                    if h > 512 or w > 512:
                        out = predict_sliding_window(model, data, tile_size=512, overlap=0.5)
                        loss_val = 0.0
                    else:
                        pred = model(data)
                        if isinstance(pred, list):
                            pred = pred[0]

                        loss = funcLoss(pred, label)
                        loss_val = loss.item()
                        out = pred

                    batch_size = out.shape[0]
                    for index in range(batch_size):
                        prob_np = out[index][0].detach().cpu().numpy()
                        gt_np = label[index][0].detach().cpu().numpy()

                        if original_size is not None:
                            real_h, real_w = original_size
                            prob_np = prob_np[:real_h, :real_w]
                            gt_np = gt_np[:real_h, :real_w]

                        eval_result = calc_metrics(
                            prob_np, gt_np, skip_cldice=skip_cldice, caliber=caliber
                        )

                        if return_per_image:
                            img_name = name[index] if isinstance(name, (list, tuple)) else name
                            per_image.append({
                                "name": str(img_name),
                                "metrics": dict(eval_result),
                                "prob": prob_np.astype(np.float32),
                                "gt": gt_np.astype(np.float32),
                            })

            eval_result["loss"] = float(loss_val)
            ls_eval_result.append(eval_result)
            total_loss += loss_val

            avg_loss = total_loss / (i + 1)
            current_lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0
            tepoch.set_postfix(
                lr=f"{current_lr:.2e}",
                loss=f"{avg_loss:.3f}",
                curr=f"{loss_val:.3f}",
            )

    time_end = time.time()
    avg_ms = (time_end - time_start) * 1000 / len(loader) / max(loader.batch_size, 1)
    num_params = sum(param.nelement() for param in model.parameters())

    result = avg_metrics(ls_eval_result)
    result["avg_ms"] = avg_ms
    result["num_params"] = num_params

    if return_per_image:
        return result, per_image
    return result