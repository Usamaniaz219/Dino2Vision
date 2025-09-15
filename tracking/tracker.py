import mlflow
import wandb
from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path
import logging
from enum import Enum
import torch
from datetime import datetime


class TrackerType(Enum):
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    MLFLOW = "mlflow"
    NONE = "none"


class ExperimentTracker:
    """Unified interface for different tracking backends"""
    
    def __init__(self, tracker_type: TrackerType, experiment_name: str, config: dict, 
                 run_id: str = None, log_dir: str = "./logs"):
        self.tracker_type = tracker_type
        self.experiment_name = experiment_name
        self.config = config
        self.log_dir = Path(log_dir)
        self.run_id = run_id or f"{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize the specific tracker
        if tracker_type == TrackerType.WANDB and wandb is not None:
            wandb.init(
                project=experiment_name,
                config=config,
                id=run_id,
                resume="allow" if run_id else False
            )
            self.tracker = wandb
        elif tracker_type == TrackerType.TENSORBOARD and SummaryWriter is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if run_id is None:
                # Default run_id as timestamp
                run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tracker = SummaryWriter(log_dir=str(self.log_dir / run_id))
        elif tracker_type == TrackerType.MLFLOW and mlflow is not None:
            mlflow.set_experiment(experiment_name)
            if run_id:
                mlflow.start_run(run_id=run_id)
            else:
                mlflow.start_run(run_name=run_id)
            mlflow.log_params(config)
        else:
            self.tracker = None
            logging.warning(f"Tracker {tracker_type} not available or not selected")
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics to the tracker"""
        if self.tracker_type == TrackerType.WANDB and self.tracker:
            self.tracker.log(metrics, step=step)
        elif self.tracker_type == TrackerType.TENSORBOARD and self.tracker:
            for key, value in metrics.items():
                self.tracker.add_scalar(key, value, step)
        elif self.tracker_type == TrackerType.MLFLOW and mlflow is not None:
            mlflow.log_metrics(metrics, step=step)
    
    def log_hyperparams(self, params: dict):
        """Log hyperparameters to the tracker"""
        if self.tracker_type == TrackerType.WANDB and self.tracker:
            self.tracker.config.update(params)
        elif self.tracker_type == TrackerType.TENSORBOARD and self.tracker:
            # TensorBoard doesn't have a dedicated hyperparam API in the same way
            pass
        elif self.tracker_type == TrackerType.MLFLOW and mlflow is not None:
            mlflow.log_params(params)
    
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """Log an artifact (file) to the tracker"""
        file_path = Path(file_path)
        if not file_path.exists():
            logging.warning(f"Artifact {file_path} does not exist")
            return
        
        if self.tracker_type == TrackerType.WANDB and self.tracker:
            self.tracker.save(str(file_path))
        elif self.tracker_type == TrackerType.TENSORBOARD and self.tracker:
            # TensorBoard doesn't have artifact logging
            pass
        elif self.tracker_type == TrackerType.MLFLOW and mlflow is not None:
            mlflow.log_artifact(str(file_path), artifact_path=artifact_type)
    
    def log_image(self, image: torch.Tensor, name: str, step: int = None):
        """Log an image to the tracker"""
        if self.tracker_type == TrackerType.WANDB and self.tracker:
            self.tracker.log({name: wandb.Image(image)}, step=step)
        elif self.tracker_type == TrackerType.TENSORBOARD and self.tracker:
            self.tracker.add_image(name, image, step)
        elif self.tracker_type == TrackerType.MLFLOW and mlflow is not None:
            # MLflow doesn't have direct image logging in the same way
            pass
    
    def finish(self):
        """Finish the tracking session"""
        if self.tracker_type == TrackerType.WANDB and self.tracker:
            wandb.finish()
        elif self.tracker_type == TrackerType.TENSORBOARD and self.tracker:
            self.tracker.close()
        elif self.tracker_type == TrackerType.MLFLOW and mlflow is not None:
            mlflow.end_run()
