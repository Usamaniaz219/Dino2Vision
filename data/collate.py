import torch
from torchvision.transforms import functional as TF
from typing import Tuple, Callable, Any
import functools
from PIL import Image
import numpy as np


class CollateWithTransform:
    """Wrapper class to make collate functions pickleable"""
    
    def __init__(self, transform, image_size=None, task="classification"):
        self.transform = transform
        self.image_size = image_size
        self.task = task
    
    def __call__(self, batch):
        if self.task == "classification":
            return self._collate_classification(batch)
        else:
            return self._collate_segmentation(batch)
    
    def _collate_classification(self, batch):
        images, labels = zip(*batch)
        processed_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                # Convert tensor to PIL image and apply transform
                pil_img = TF.to_pil_image((img * 255).byte())
                processed_images.append(self.transform(pil_img))
            else:
                # Assume it's already a PIL image or compatible
                processed_images.append(self.transform(img))
        return torch.stack(processed_images, 0), torch.tensor(labels, dtype=torch.long)
    
    # def _collate_segmentation(self, batch):
    #     ims, masks = zip(*batch)
    #     proc_ims, proc_masks = [], []
        
    #     for img, mask in zip(ims, masks):
    #         # Process image
    #         if isinstance(img, torch.Tensor):
    #             pil_img = TF.to_pil_image((img * 255).byte())
    #         else:
    #             pil_img = img
    #             pil_img = Image.fromarray(pil_img)
    #         timg = self.transform(pil_img)
            
    #         # Process mask
    #         if self.image_size:
    #             # Resize mask to match model input spatial size
    #             tmask = TF.resize(
    #                 mask.unsqueeze(0).float(), 
    #                 size=[timg.shape[1], timg.shape[2]], 
    #                 interpolation=TF.InterpolationMode.NEAREST
    #             ).squeeze(0).long()
    #         else:
    #             tmask = mask
            
    #         proc_ims.append(timg)
    #         proc_masks.append(tmask)
        
    #     return torch.stack(proc_ims, 0), torch.stack(proc_masks, 0)

    def _collate_segmentation(self, batch): 
        ims, masks = zip(*batch)
        proc_ims, proc_masks = [], []
        
        for img, mask in zip(ims, masks):
            # --- Process image ---
            if isinstance(img, torch.Tensor):
                pil_img = TF.to_pil_image((img * 255).byte())
            else:
                pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            timg = self.transform(pil_img)

            # --- Process mask ---
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)   # ✅ Convert NumPy → Tensor
            if mask.dtype != torch.long:
                mask = mask.long()
            
            if self.image_size:
                # Resize mask to match model input spatial size
                tmask = TF.resize(
                    mask.unsqueeze(0).float(), 
                    size=[timg.shape[1], timg.shape[2]], 
                    interpolation=TF.InterpolationMode.NEAREST
                ).squeeze(0).long()
            else:
                tmask = mask

            proc_ims.append(timg)
            proc_masks.append(tmask)
    
        return torch.stack(proc_ims, 0), torch.stack(proc_masks, 0)


def create_collate_fn(transform, image_size=None, task="classification"):
    """
    Create a pickleable collate function for DataLoader
    
    Args:
        transform: The transform to apply to images
        image_size: Tuple (height, width) for segmentation tasks
        task: Either "classification" or "segmentation"
    """
    return CollateWithTransform(transform, image_size, task)





















# import torch
# from torchvision.transforms import functional as TF
# from typing import Tuple


# def collate_with_transform_classification(transform):
#     def _collate(batch):
#         images, labels = zip(*batch)
#         images = [transform(TF.to_pil_image((img*255).byte())) if isinstance(img, torch.Tensor) else transform(img) for img in images]
#         return torch.stack(images, 0), torch.tensor(labels, dtype=torch.long)
#     return _collate


# def collate_with_transform_segmentation(transform, image_size: Tuple[int,int]):
#     Ih, Iw = image_size
#     def _collate(batch):
#         ims, masks = zip(*batch)
#         proc_ims, proc_masks = [], []
#         for img, mask in zip(ims, masks):
#             pil_img = TF.to_pil_image((img*255).byte()) if isinstance(img, torch.Tensor) else img
#             timg = transform(pil_img)  # normalized tensor [3,H',W']
#             # Resize mask to match model input spatial size
#             tmask = TF.resize(mask.unsqueeze(0).float(), size=[timg.shape[1], timg.shape[2]], 
#                              interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()
#             proc_ims.append(timg)
#             proc_masks.append(tmask)
#         return torch.stack(proc_ims, 0), torch.stack(proc_masks, 0)
#     return _collate