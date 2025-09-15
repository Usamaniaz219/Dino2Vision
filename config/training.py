
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    task: str
    num_classes: int
    backbone: str = "vit_base_patch14_dinov2"
    pretrained: bool = True
    fine_tune_strategy: str = "partial"
    unfreeze_blocks: int = 4
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    ckpt_dir: str = "./checkpoints"
    early_stop: int = 10
    seed: int = 42
    workers: int = 4
    resume: Optional[str] = None
    mask_suffix: str = "_mask.png"
    tracker: str = "none"
    experiment_name: str = "dinov3-training"
    log_dir: str = "./logs"
    local_rank: int = 0
    log_level: str = "INFO"
    log_file: Optional[str] = None