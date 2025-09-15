import os
from typing import Callable, Optional, List, Tuple, Dict
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        mask_suffix: str = "_mask.png",
        image_exts: List[str] = [".png", ".jpg", ".jpeg"],
        num_classes: Optional[int] = None,
        rgb_to_class_map: Optional[Dict[Tuple[int, int, int], int]] = None,
        cache: bool = False
    ):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.image_exts = image_exts
        self.num_classes = num_classes
        self.rgb_to_class_map = rgb_to_class_map
        self.cache = cache

        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir):
            raise RuntimeError(f"Image or mask directory not found: {self.img_dir}, {self.mask_dir}")

        self.images = []
        for fname in os.listdir(self.img_dir):
            base, ext = os.path.splitext(fname)
            if ext.lower() in self.image_exts:
                mask_name = base + self.mask_suffix
                if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                    self.images.append(base)

        if not self.images:
            raise RuntimeError("No valid image-mask pairs found!")

        self.cache_data = {} if cache else None

    def __len__(self):
        return len(self.images)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    def _load_mask(self, path: str) -> np.ndarray:
        # Try to load as grayscale first
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path}")

        # Case 1: RGB mask (3 channels)
        if mask.ndim == 3 and mask.shape[2] == 3:
            if self.rgb_to_class_map:
                h, w, _ = mask.shape
                new_mask = np.zeros((h, w), dtype=np.int64)
                for rgb, cls_id in self.rgb_to_class_map.items():
                    new_mask[(mask == rgb).all(axis=-1)] = cls_id
                mask = new_mask
            else:
                # Auto-convert to grayscale if no mapping provided
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.int64)

        # Case 2: Grayscale mask (already 1 channel)
        else:
            mask = mask.astype(np.int64)

        # Validate classes (useful for binary masks 0/1)
        if self.num_classes is not None and mask.max() >= self.num_classes:
            raise ValueError(
                f"Mask contains invalid class IDs. Found max ID {mask.max()} "
                f"but num_classes={self.num_classes}"
            )

        return mask


    # def _load_mask(self, path: str) -> np.ndarray:
    #     mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #     if mask is None:
    #         raise FileNotFoundError(f"Mask not found: {path}")

    #     # Case 1: RGB mask (3 channels)
    #     if mask.ndim == 3 and mask.shape[2] == 3:
    #         if self.rgb_to_class_map:
    #             h, w, _ = mask.shape
    #             new_mask = np.zeros((h, w), dtype=np.int64)
    #             for rgb, cls_id in self.rgb_to_class_map.items():
    #                 new_mask[(mask == rgb).all(axis=-1)] = cls_id
    #             mask = new_mask
    #         else:
    #             raise ValueError(
    #                 "RGB mask detected but no rgb_to_class_map provided. "
    #                 "Provide a mapping like {(0,0,0):0, (128,0,0):1, ...}"
    #             )
    #     else:
    #         # Case 2: Grayscale mask
    #         mask = mask.astype(np.int64)

    #     # Validate classes
    #     if self.num_classes is not None and mask.max() >= self.num_classes:
    #         raise ValueError(
    #             f"Mask contains invalid class IDs. Found max ID {mask.max()} but num_classes={self.num_classes}"
    #         )

    #     return mask

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        base = self.images[idx]
        img_path = None
        for ext in self.image_exts:
            candidate = os.path.join(self.img_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(f"No valid image found for base name: {base}")

        mask_path = os.path.join(self.mask_dir, base + self.mask_suffix)

        # Load from cache or disk
        if self.cache and base in self.cache_data:
            image, mask = self.cache_data[base]
        else:
            image = self._load_image(img_path)
            mask = self._load_mask(mask_path)
            if self.cache:
                self.cache_data[base] = (image, mask)

        # Apply Albumentations transforms if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask
