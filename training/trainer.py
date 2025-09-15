import time
import torch
from .utils import fmt_time
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from tracking.tracker import TrackerType

RICH_AVAILABLE = True


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, 
                   grad_clip=1.0, task="classification", tracker=None, epoch=0):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    t0 = time.time()
    
    # Create progress bar if available
    if RICH_AVAILABLE and (not tracker or tracker.tracker_type == TrackerType.NONE):
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        task_id = progress.add_task("[green]Training...", total=len(loader))
        progress.start()
    else:
        progress = None
        task_id = None
    
    for batch_idx, (images, target) in enumerate(loader):
        images = images.to(device)
        target = target.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        outputs = model(images)
        loss = criterion(outputs, target)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
        total_loss += loss.item() * images.size(0)
        
        if task == "classification":
            pred = outputs.argmax(1)
            correct += (pred == target).sum().item()
            total += images.size(0)
        else:
            # For segmentation, compute pixel accuracy
            pred = outputs.argmax(1)
            correct += (pred == target).sum().item()
            total += target.numel()
        
        # Update progress bar
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
        
        # Log batch metrics
        if tracker and batch_idx % 10 == 0:  # Log every 10 batches
            batch_metrics = {
                "train/batch_loss": loss.item(),
                "train/batch_lr": optimizer.param_groups[0]['lr']
            }
            if task == "classification":
                batch_acc = (pred == target).sum().item() / images.size(0)
                batch_metrics["train/batch_acc"] = batch_acc
            else:
                batch_acc = (pred == target).sum().item() / target.numel()
                batch_metrics["train/batch_acc"] = batch_acc
            
            tracker.log_metrics(batch_metrics, step=epoch * len(loader) + batch_idx)
    
    if progress:
        progress.stop()
    
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = correct / max(1, total)
    epoch_time = fmt_time(time.time() - t0)
    
    return avg_loss, acc, epoch_time
