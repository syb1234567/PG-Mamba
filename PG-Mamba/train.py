"""
Training script for PG-Mamba with 5-fold subject-level cross-validation.

Each subject is assigned to a single fold so that paired eyes from the same
participant cannot appear in both training and test sets. The best checkpoint
is selected by the lowest validation Brier score among epochs satisfying the
Dice gate.
"""

import json
import os
import random
import re
import shutil

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SegmentationDataset
from loss import CombinedLoss
from utils import traverseDataset


class Config:
    IMAGE_DIR = "image"
    LABEL_DIR = "label"
    LABEL_PREFIX = "mask_"
    ROOT_RESULT = "result"
    N_SPLITS = 5
    CROP_SIZE = 512
    INPUT_CHANNELS = 1
    BATCH_SIZE = 2
    NUM_WORKERS = 3
    MAX_EPOCH = 400
    EARLY_STOP_PATIENCE = 40
    WEIGHT_DECAY = 1e-2
    DICE_GATE = 0.95
    SEED = 0


MODEL_LR = {
    "PGMamba": 3e-4,
}
DEFAULT_LR = 3e-4


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"seed set to {seed}")


def build_models():
    """Return the models included in the public training release."""
    from our_model.PGMamba import PGMamba

    ic = Config.INPUT_CHANNELS
    return {"PGMamba": lambda: PGMamba(input_channels=ic, num_classes=1)}


def pick_best(res_val, state):
    """Select the lowest-Brier checkpoint that satisfies the Dice gate."""
    dice = res_val["dice"]
    brier = res_val["brier"]
    state["running_best_dice"] = max(state["running_best_dice"], dice)
    gate = Config.DICE_GATE * state["running_best_dice"]

    if dice < gate:
        return False
    if brier < state["best_brier"]:
        state["best_brier"] = brier
        return True
    return False


def run():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, input_channels: {Config.INPUT_CHANNELS}")

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    names = sorted([f for f in os.listdir(Config.IMAGE_DIR) if f.lower().endswith(exts)])
    subjects = [re.match(r"(\d+)_[LR]", os.path.splitext(n)[0]).group(1) for n in names]
    names = np.array(names)
    subjects = np.array(subjects)
    print(f"samples: {len(names)}, subjects: {len(set(subjects))}")

    gkf = GroupKFold(n_splits=Config.N_SPLITS)
    folds = list(gkf.split(names, groups=subjects))

    models = build_models()
    print(f"models: {list(models.keys())}")
    os.makedirs(Config.ROOT_RESULT, exist_ok=True)

    for name_model in models:
        lr = MODEL_LR.get(name_model, DEFAULT_LR)
        for fold, (trainval_idx, test_idx) in enumerate(folds):
            run_dir = os.path.join(Config.ROOT_RESULT, name_model, f"fold{fold}")
            flag = os.path.join(run_dir, "finished.flag")

            if os.path.exists(flag):
                print(f"[skip] {name_model} fold{fold} done.")
                continue
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

            os.makedirs(run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=run_dir)

            tv_names = names[trainval_idx]
            tv_subj = subjects[trainval_idx]
            uniq_subj = list(dict.fromkeys(tv_subj.tolist()))
            n_val_subj = max(1, len(uniq_subj) // 6)
            val_subj = set(uniq_subj[-n_val_subj:])
            val_mask = np.array([s in val_subj for s in tv_subj])

            train_names = tv_names[~val_mask].tolist()
            val_names = tv_names[val_mask].tolist()
            test_names = names[test_idx].tolist()

            print(
                f"\n=== {name_model} | fold{fold} | lr={lr:.0e} | "
                f"train={len(train_names)} val={len(val_names)} test={len(test_names)} ==="
            )

            ds_tr = SegmentationDataset(Config.IMAGE_DIR, Config.LABEL_DIR, train_names, Config.LABEL_PREFIX, mode="train", crop_size=Config.CROP_SIZE)
            ds_va = SegmentationDataset(Config.IMAGE_DIR, Config.LABEL_DIR, val_names, Config.LABEL_PREFIX, mode="val")
            ds_te = SegmentationDataset(Config.IMAGE_DIR, Config.LABEL_DIR, test_names, Config.LABEL_PREFIX, mode="test")

            tr_loader = DataLoader(ds_tr, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=Config.NUM_WORKERS)
            va_loader = DataLoader(ds_va, batch_size=1, num_workers=1)
            te_loader = DataLoader(ds_te, batch_size=1, num_workers=1)

            model = models[name_model]().to(device)
            criterion = CombinedLoss(weight_dice=0.5, weight_mse=0.5)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=Config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=Config.MAX_EPOCH, eta_min=1e-6)

            config = {
                "model": name_model,
                "fold": fold,
                "lr": lr,
                "input_channels": Config.INPUT_CHANNELS,
                "selection": f"Brier-min within Dice>={Config.DICE_GATE:.2f}*best_dice",
            }
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            state = {"running_best_dice": -1, "best_brier": float("inf"), "best_epoch": -1}
            for epoch in range(Config.MAX_EPOCH):
                torch.cuda.empty_cache()
                res_tr = traverseDataset(model, tr_loader, f"{name_model} f{fold} E{epoch} train", device, criterion, None, None, optimizer=optimizer)
                res_va = traverseDataset(model, va_loader, f"{name_model} f{fold} E{epoch} val", device, criterion, None, None, optimizer=None)
                scheduler.step()

                writer.add_scalar("train/loss", res_tr["loss"], epoch)
                for key in ("dice", "brier", "sdice", "cldice"):
                    writer.add_scalar(f"val/{key}", res_va.get(key, 0), epoch)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

                if pick_best(res_va, state):
                    state["best_epoch"] = epoch
                    torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pth"))
                    best_val = {"epoch": epoch, **{k: float(v) for k, v in res_va.items() if isinstance(v, (int, float))}}
                    with open(os.path.join(run_dir, "best_val.json"), "w") as f:
                        json.dump(best_val, f, indent=2)
                    print(f"  [best] E{epoch} dice={res_va['dice']:.4f} brier={res_va['brier']:.5f}")

                if epoch - state["best_epoch"] >= Config.EARLY_STOP_PATIENCE:
                    print(f"  early stop at E{epoch} (best E{state['best_epoch']})")
                    break

            best_path = os.path.join(run_dir, "model_best.pth")
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device))

            res_te = traverseDataset(model, te_loader, f"{name_model} f{fold} TEST", device, criterion, None, None, optimizer=None)
            test_metrics = {k: float(v) for k, v in res_te.items() if isinstance(v, (int, float))}
            with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)

            print(
                f"  TEST: dice={res_te.get('dice', 0):.4f} "
                f"brier={res_te.get('brier', 0):.5f} "
                f"cldice={res_te.get('cldice', 0):.4f}"
            )

            writer.close()
            with open(flag, "w") as f:
                f.write("done")

    print("\nAll folds finished. Aggregate result/{model}/fold*/test_metrics.json for mean and standard deviation.")


if __name__ == "__main__":
    run()