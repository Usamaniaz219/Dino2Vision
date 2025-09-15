import argparse
import torch
import logging
from dataclasses import asdict
import os
from config.training import TrainConfig
from utils.helpers import create_data_loaders, setup_model, run_training_epochs
from utils.logging import setup_logging
from training.utils import set_seed,param_groups
from tracking.tracker import TrackerType, ExperimentTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DINOv3 for classification/segmentation")
    p.add_argument('--task', type=str,default='classification', choices=['classification', 'segmentation'])
    p.add_argument('--num-classes', type=int, default=2)
    p.add_argument('--backbone', type=str, default='vit_base_patch14_dinov2', help='e.g., vit_small/base/large/giant2')
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    p.add_argument('--fine-tune-strategy', type=str, default='partial', choices=['linear','partial','full'])
    p.add_argument('--unfreeze-blocks', type=int, default=4)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no-amp', dest='amp', action='store_false')
    p.add_argument('--ckpt-dir', type=str, default='./checkpoints')
    p.add_argument('--early-stop', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--train-dir', type=str,default=r'C:\Users\osama\OneDrive\Desktop\Usama_dev\Fine_tuning_DINOv3\data\train')
    p.add_argument('--val-dir', type=str,default=r'C:\Users\osama\OneDrive\Desktop\Usama_dev\Fine_tuning_DINOv3\data\val')
    p.add_argument('--mask-suffix', type=str, default='_mask.png')
    p.add_argument('--tracker', type=str, default='wandb', choices=['wandb', 'tensorboard', 'mlflow', 'none'])
    p.add_argument('--experiment-name', type=str, default='dinov3-training')
    p.add_argument('--log-dir', type=str, default='./logs')
    p.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    p.add_argument('--log-file', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    
    # Setup configuration
    cfg = TrainConfig(
        task=args.task,
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        fine_tune_strategy=args.fine_tune_strategy,
        unfreeze_blocks=args.unfreeze_blocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=args.amp,
        ckpt_dir=args.ckpt_dir,
        early_stop=args.early_stop,
        seed=args.seed,
        workers=args.workers,
        resume=args.resume,
        mask_suffix=args.mask_suffix,
        tracker=args.tracker,
        experiment_name=args.experiment_name,
        log_dir=args.log_dir,
        log_level=args.log_level,
        log_file=args.log_file,
    )
    
    print('configuration',cfg)
    # Setup logging
    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logger = setup_logging(log_level=log_level, log_file=cfg.log_file)
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Setup device
    device = torch.device('cpu')
    
    # Create data loaders
    train_loader, val_loader, criterion = create_data_loaders(cfg, args.train_dir, args.val_dir)
    
    # Setup model
    model = setup_model(cfg, device)
    
    # Setup optimizer and scaler
    # scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == 'cuda')
    # scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == 'cpu')
    scaler = None
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = -1.0
    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', best_val_acc)
        logging.info(f"Resumed from {cfg.resume} at epoch {start_epoch}")
    
    # Setup experiment tracker
    tracker_type = TrackerType(cfg.tracker.lower())
    tracker = ExperimentTracker(
        tracker_type=tracker_type,
        experiment_name=cfg.experiment_name,
        config=asdict(cfg),
        log_dir=cfg.log_dir
    )
    
    # Run training
    best_val_acc = run_training_epochs(
        cfg, model, train_loader, val_loader, criterion, 
        optimizer, scaler, device, tracker
    )
    
    # Finish tracking
    if tracker:
        tracker.finish()


if __name__ == '__main__':
    main()
