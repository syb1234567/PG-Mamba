import cv2
import os
import numpy as np
import random
import torch
from torch.utils.data.dataset import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, ls_path_dataset, start=0, end=1, 
                 image_dir_name="image", label_dir_name="label", 
                 label_prefix="", mode="train", crop_size=512,
                 disable_rotation=True) -> None:  # ← 新增参数
        super().__init__()

        if not isinstance(ls_path_dataset, list):
            ls_path_dataset = [ls_path_dataset]

        self.label_prefix = label_prefix
        self.mode = mode
        self.crop_size = crop_size
        self.disable_rotation = disable_rotation  # ← 新增
        self.ls_item = []
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        for path_dataset in ls_path_dataset:
            path_dir_image = os.path.join(path_dataset, image_dir_name)
            path_dir_label = os.path.join(path_dataset, label_dir_name)
            
            print(f"[{mode}] Scanning Image Dir: {path_dir_image}")
            
            if not os.path.exists(path_dir_image):
                continue
            
            ls_file = os.listdir(path_dir_image)
            count_matched = 0
            
            for name in ls_file:
                if not name.lower().endswith(valid_extensions): 
                    continue
                
                path_image = os.path.join(path_dir_image, name)
                label_name = (self.label_prefix + name) if self.label_prefix else name
                path_label = os.path.join(path_dir_label, label_name)
                
                if os.path.exists(path_label):
                    self.ls_item.append({
                        "name": name,
                        "path_image": path_image,
                        "path_label": path_label,
                    })
                    count_matched += 1

            print(f"  -> Found {count_matched} pairs.")

        if len(self.ls_item) > 0:
            random.seed(0)
            random.shuffle(self.ls_item)
            start = int(start * len(self.ls_item))
            end = int(end * len(self.ls_item))
            if end == start and len(self.ls_item) > 0:
                end = len(self.ls_item)
            self.ls_item = self.ls_item[start:end]
        
        print(f"Total Loaded {len(self.ls_item)} samples (Mode: {self.mode})")

    def __len__(self):
        return len(self.ls_item)

    def __getitem__(self, index):
        index = index % len(self)
        item = self.ls_item[index]
        name = item['name']

        # 1. 读取数据
        image = cv2.imread(item['path_image'], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(item['path_label'], cv2.IMREAD_GRAYSCALE)
        
        if image is None or label is None:
            raise ValueError(f"Failed to read data: {name}")

        image = image.astype("float32") / 255.0
        label = label.astype("float32") / 255.0
        
        # 2. FOV Masking
        blur = cv2.GaussianBlur(image, (25, 25), 0)
        _, fov_mask = cv2.threshold(blur, 0.05, 1.0, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, kernel)
        
        image = image * fov_mask
        label = label * fov_mask

        # 3. CLAHE + Gamma
        img_u8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_u8)
        image = img_clahe.astype("float32") / 255.0
        
        if self.mode == "train":
            gamma = random.uniform(1.2, 1.6)
        else:
            gamma = 1.5
            
        image = np.power(image + 1e-6, gamma)
        image = image * fov_mask

        # 4. ✅ 改进：生成全局坐标
        H_orig, W_orig = image.shape
        original_size = np.array([H_orig, W_orig])

        x_range = np.linspace(-1, 1, W_orig).astype(np.float32)
        y_range = np.linspace(-1, 1, H_orig).astype(np.float32)
        x_map, y_map = np.meshgrid(x_range, y_range)

        # 5. 数据增强
        if self.mode == "train":
            # (A) Random Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)
                x_map = cv2.flip(x_map, 1)
                y_map = cv2.flip(y_map, 1)
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)
                x_map = cv2.flip(x_map, 0)
                y_map = cv2.flip(y_map, 0)
            
            # (B) ✅ 可选的旋转增强
            if not self.disable_rotation:
                k = random.randint(0, 3)
                image = np.rot90(image, k)
                label = np.rot90(label, k)
                x_map = np.rot90(x_map, k)
                y_map = np.rot90(y_map, k)

            H, W = image.shape

            # (C) Padding
            cH, cW = self.crop_size, self.crop_size
            pad_h = max(0, cH - H)
            pad_w = max(0, cW - W)
            
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                x_map = cv2.copyMakeBorder(x_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                y_map = cv2.copyMakeBorder(y_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                H, W = image.shape

            # (D) Random Crop
            x_start = random.randint(0, W - cW)
            y_start = random.randint(0, H - cH)
            image = image[y_start:y_start+cH, x_start:x_start+cW]
            label = label[y_start:y_start+cH, x_start:x_start+cW]
            x_map = x_map[y_start:y_start+cH, x_start:x_start+cW]
            y_map = y_map[y_start:y_start+cH, x_start:x_start+cW]
            
        else:
            # Val/Test Padding
            H, W = image.shape
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                x_map = cv2.copyMakeBorder(x_map, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                y_map = cv2.copyMakeBorder(y_map, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        # 6. ✅ 堆叠通道（保持全局坐标）
        input_stack = np.stack([image, x_map, y_map], axis=-1)
        
        input_stack = np.ascontiguousarray(input_stack)
        input_tensor = torch.from_numpy(input_stack).permute(2, 0, 1).float()
        
        label = np.ascontiguousarray(label)
        label_tensor = torch.from_numpy(label).unsqueeze(0).float()

        return name, input_tensor, label_tensor, original_size

def prepareDatasets():
    path_OCTA_Custom = "/root/autodl-tmp/OCTAMamba/data" 
    from loss import CombinedLoss 
    
    all_datasets = {}
    all_datasets['OCTA_Custom'] = {
        "train": SegmentationDataset(
            os.path.join(path_OCTA_Custom, "train"),
            image_dir_name="images", 
            label_dir_name="labels", 
            label_prefix="mask_",
            mode="train", 
            crop_size=512,
            disable_rotation=True  # ✅ 禁用旋转
        ),
        "val": SegmentationDataset(
            os.path.join(path_OCTA_Custom, "val"),
            image_dir_name="images", 
            label_dir_name="labels", 
            label_prefix="mask_",
            mode="val", 
            crop_size=512,
            disable_rotation=True
        ),
        "test": SegmentationDataset(
            os.path.join(path_OCTA_Custom, "test"),
            image_dir_name="images", 
            label_dir_name="labels", 
            label_prefix="mask_",
            mode="test", 
            crop_size=512,
            disable_rotation=True
        ),
        "loss": CombinedLoss()
    }
    return all_datasets