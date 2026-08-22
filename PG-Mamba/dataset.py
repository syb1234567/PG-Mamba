import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Filename-list driven dataset for OCTA vessel segmentation.

    Input:
        grayscale OCTA image and probabilistic soft label.

    Output:
        name, image tensor [1,H,W], label tensor [1,H,W], original size.
    """

    def __init__(self, image_dir, label_dir, names,
                 label_prefix="mask_", mode="train",
                 crop_size=512, disable_rotation=True):
        super().__init__()

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_prefix = label_prefix
        self.mode = mode
        self.crop_size = crop_size
        self.disable_rotation = disable_rotation

        self.ls_item = []
        miss = 0

        for name in names:
            path_image = os.path.join(image_dir, name)
            label_name = label_prefix + name if label_prefix else name
            path_label = os.path.join(label_dir, label_name)

            if os.path.exists(path_image) and os.path.exists(path_label):
                self.ls_item.append({
                    "name": name,
                    "path_image": path_image,
                    "path_label": path_label
                })
            else:
                miss += 1

        msg = f"[{mode}] loaded {len(self.ls_item)} pairs"
        if miss:
            msg += f" ({miss} missing)"
        print(msg)


    def __len__(self):
        return len(self.ls_item)


    def __getitem__(self, index):

        index = index % len(self)

        item = self.ls_item[index]
        name = item["name"]

        # Load grayscale image and soft label
        image = cv2.imread(item["path_image"], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(item["path_label"], cv2.IMREAD_GRAYSCALE)

        if image is None or label is None:
            raise ValueError(f"Failed to read data: {name}")

        image = image.astype("float32") / 255.0
        label = label.astype("float32") / 255.0

        # Field-of-view mask
        blur = cv2.GaussianBlur(image, (25, 25), 0)
        _, fov_mask = cv2.threshold(blur, 0.05, 1.0, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, kernel)

        image *= fov_mask
        label *= fov_mask

        # CLAHE + gamma correction
        img_u8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        image = clahe.apply(img_u8).astype("float32") / 255.0

        gamma = random.uniform(1.2, 1.6) if self.mode == "train" else 1.5
        image = np.power(image + 1e-6, gamma) * fov_mask

        H_orig, W_orig = image.shape
        original_size = np.array([H_orig, W_orig])


        # Augmentation and crop
        if self.mode == "train":

            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)

            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)

            if not self.disable_rotation:
                k = random.randint(0, 3)
                image = np.rot90(image, k)
                label = np.rot90(label, k)

            H, W = image.shape
            cH, cW = self.crop_size, self.crop_size

            pad_h = max(0, cH - H)
            pad_w = max(0, cW - W)

            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(
                    image, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=0
                )
                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=0
                )

                H, W = image.shape

            x_start = random.randint(0, W - cW)
            y_start = random.randint(0, H - cH)

            image = image[y_start:y_start+cH, x_start:x_start+cW]
            label = label[y_start:y_start+cH, x_start:x_start+cW]

        else:
            # Pad validation/test images for encoder-decoder networks
            H, W = image.shape

            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32

            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(
                    image, 0, pad_h, 0, pad_w,
                    cv2.BORDER_REFLECT
                )

                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w,
                    cv2.BORDER_REFLECT
                )


        # Convert to tensors
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label).unsqueeze(0).float()

        return name, image_tensor, label_tensor, original_size