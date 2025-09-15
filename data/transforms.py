import torch
from torchvision.transforms import functional as TF
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(backbone):
    cfg = resolve_data_config({}, model=backbone)
    transform = create_transform(**cfg)
    # For segmentation masks we will only resize/center-crop and convert to tensor w/o normalization.
    # We'll derive sizes from cfg.
    size = cfg.get('input_size', (3, 224, 224))[1:]
    return transform, size, cfg


def get_albumentations_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
