import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import logging

from config.training import TrainConfig
from data.transforms import build_transforms
# from data.collate import collate_with_transform_classification, collate_with_transform_segmentation
from data.collate import create_collate_fn
from data.datasets import SegmentationDataset
from models.backbone import build_backbone, apply_fine_tune_strategy
from models.classifier import DinoClassifier
from models.segmentation import DinoV2SegHead
from tracking.tracker import TrackerType, ExperimentTracker
from training.trainer import train_one_epoch
from training.evaluator import evaluate
from training.utils import param_groups
from utils.logging import setup_logging




# Update the create_data_loaders function:
def create_data_loaders(cfg, train_dir, val_dir):
    """Create appropriate data loaders based on task"""
    backbone = build_backbone(cfg.backbone, pretrained=cfg.pretrained)
    transform, image_size, _ = build_transforms(backbone)

    # Create collate functions
    collate_fn = create_collate_fn(transform, image_size, cfg.task)

    if cfg.task == 'classification':
        train_ds = datasets.ImageFolder(train_dir)
        val_ds = datasets.ImageFolder(val_dir)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=1 if cfg.workers > 0 else 0,  # Set to 0 for Windows compatibility
            # pin_memory=torch.cuda.is_available(),  # Only pin memory if CUDA is available
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=1 if cfg.workers > 0 else 0,
            # pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
        )
        criterion = nn.CrossEntropyLoss()
    else:
        train_ds = SegmentationDataset(train_dir, transform=None, mask_suffix=cfg.mask_suffix)
        val_ds = SegmentationDataset(val_dir, transform=None, mask_suffix=cfg.mask_suffix)    
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=1 if cfg.workers > 0 else 0,
            # pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=1 if cfg.workers > 0 else 0,
            # pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
        )
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    return train_loader, val_loader, criterion

# def create_data_loaders(cfg, train_dir, val_dir):
#     """Create appropriate data loaders based on task"""
#     backbone = build_backbone(cfg.backbone, pretrained=cfg.pretrained)
#     transform, image_size, _ = build_transforms(backbone)

#     if cfg.task == 'classification':
#         train_ds = datasets.ImageFolder(train_dir)
#         val_ds = datasets.ImageFolder(val_dir)
#         train_loader = DataLoader(
#             train_ds, batch_size=cfg.batch_size, shuffle=True,
#             num_workers=cfg.workers, pin_memory=True,
#             collate_fn=collate_with_transform_classification(transform)
#         )
#         val_loader = DataLoader(
#             val_ds, batch_size=cfg.batch_size, shuffle=False,
#             num_workers=cfg.workers, pin_memory=True,
#             collate_fn=collate_with_transform_classification(transform)
#         )
#         criterion = nn.CrossEntropyLoss()
#     else:
#         train_ds = SegmentationDataset(train_dir, transform=None, mask_suffix=cfg.mask_suffix)
#         val_ds = SegmentationDataset(val_dir, transform=None, mask_suffix=cfg.mask_suffix)    
#         train_loader = DataLoader(
#             train_ds, batch_size=cfg.batch_size, shuffle=True,
#             num_workers=cfg.workers, pin_memory=True,
#             collate_fn=collate_with_transform_segmentation(transform, image_size)
#         )
#         val_loader = DataLoader(
#             val_ds, batch_size=cfg.batch_size, shuffle=False,
#             num_workers=cfg.workers, pin_memory=True,
#             collate_fn=collate_with_transform_segmentation(transform, image_size)
#         )
#         criterion = nn.CrossEntropyLoss(ignore_index=255)

#     return train_loader, val_loader, criterion


def setup_model(cfg, device):
    """Setup model with appropriate fine-tuning strategy"""
    backbone = build_backbone(cfg.backbone, pretrained=cfg.pretrained)

    # Optional: enable grad checkpointing for big models
    if hasattr(backbone, 'set_grad_checkpointing'):
        backbone.set_grad_checkpointing(True)

    if cfg.task == 'classification':
        model = DinoClassifier(backbone, cfg.num_classes)
    elif cfg.task == 'segmentation':
        model = DinoV2SegHead(backbone, cfg.num_classes)
    else:
        raise ValueError("task must be 'classification' or 'segmentation'")

    apply_fine_tune_strategy(backbone, cfg.fine_tune_strategy, cfg.unfreeze_blocks)
    model = model.to(device)
    
    return model


def run_training_epochs(cfg, model, train_loader, val_loader, criterion, 
                       optimizer, scaler, device, tracker):
    """Run the main training loop"""
    best_val_acc = -1.0
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    patience = 0
    
    for epoch in range(cfg.epochs):
        # Training
        tr_loss, tr_acc, tr_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, 
            cfg.grad_clip, cfg.task, tracker, epoch
        )
        
        # Validation
        va_loss, va_acc, va_time = evaluate(
            model, val_loader, criterion, device, cfg.task, tracker, epoch
        )
        
        # Log epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'train/loss': tr_loss, 'train/accuracy': tr_acc, 'train/time': tr_time,
            'val/loss': va_loss, 'val/accuracy': va_acc, 'val/time': va_time,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        logging.info(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in epoch_metrics.items()}))
        
        if tracker:
            tracker.log_metrics({
                'train/loss': tr_loss,
                'train/accuracy': tr_acc,
                'val/loss': va_loss,
                'val/accuracy': va_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)

        # Save checkpoint
        ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_acc': best_val_acc,
            'cfg': cfg.__dict__
        }, ckpt_path)
        
        if tracker:
            tracker.log_artifact(ckpt_path, "checkpoints")

        # Track best model
        improved = va_acc > best_val_acc
        if improved:
            best_val_acc = va_acc
            best_path = os.path.join(cfg.ckpt_dir, 'best.pt')
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, best_path)
            
            if tracker:
                tracker.log_artifact(best_path, "models")
                    
            patience = 0
        else:
            patience += 1

        if patience >= cfg.early_stop:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    logging.info(f"Best val acc: {best_val_acc:.4f}")
    return best_val_acc
