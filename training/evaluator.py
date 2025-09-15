import time
import torch
from .utils import fmt_time
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from tracking.tracker import TrackerType

RICH_AVAILABLE = True


def evaluate(model, loader, criterion, device, task="classification", tracker=None, epoch=0):
    model.eval()
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
        task_id = progress.add_task("[green]Evaluating...", total=len(loader))
        progress.start()
    else:
        progress = None
        task_id = None
    
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(loader):
            images = images.to(device)
            target = target.to(device)
            outputs = model(images)
            loss = criterion(outputs, target)
            total_loss += loss.item() * images.size(0)
            
            if task == "classification":
                pred = outputs.argmax(1)
                correct += (pred == target).sum().item()
                total += images.size(0)
            else:
                pred = outputs.argmax(1)
                correct += (pred == target).sum().item()
                total += target.numel()
            
            # Update progress bar
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
    
    if progress:
        progress.stop()
    
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = correct / max(1, total)
    epoch_time = fmt_time(time.time() - t0)
    
    return avg_loss, acc, epoch_time
