# Dino2Vision: DINOv2 Fine-Tuning Framework

A comprehensive, production-ready framework for fine-tuning DINOv2 models for both classification and segmentation tasks. Built with PyTorch and designed for scalability and ease of use.

## 📋 Table of Contents

- [Features](#-features)
- [Supported DINOv2 Models](#-supported-dinov2-models)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Custom DINOv2 Models](#-custom-dinov2-models)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Training Monitoring](#-training-monitoring)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Multi-Task Support**: Classification and semantic segmentation
- **Flexible Backbones**: Support for various DINOv2 model sizes
- **Fine-Tuning Strategies**: Linear, partial, and full fine-tuning options
- **Production Ready**: Logging, checkpointing, and experiment tracking
- **Scalable Architecture**: Modular design for easy extension
- **Multi-Framework Support**: Weights & Biases, TensorBoard, and MLflow integration

## 🎯 Supported DINOv2 Models

The framework supports all DINOv2 models available in the `timm` library:

### Standard DINOv2 Models:
- `vit_small_patch14_dinov2` (21M params)
- `vit_base_patch14_dinov2` (86M params) - **Recommended for most use cases**
- `vit_large_patch14_dinov2` (300M params)
- `vit_giant_patch14_dinov2` (1.1B params)

### Custom DINOv2 Models:
You can use any Vision Transformer model with DINOv2 pre-trained weights:
- Custom patch sizes: `patch14`, `patch16`, `patch8`
- Different model architectures
- Domain-specific pre-trained models

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-enabled GPU (recommended)

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd dinovision-project

# Create virtual environment
python -m venv dino_env
source dino_env/bin/activate  # On Windows: dino_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Requirements
```bash
pip install torch torchvision timm albumentations opencv-python
```

### Optional (for tracking)
```bash
pip install wandb tensorboard mlflow rich
```

## ⚡ Quick Start

### Classification Example
```bash
python main.py \
    --task classification \
    --num-classes 10 \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --backbone vit_base_patch14_dinov2 \
    --epochs 50 \
    --batch-size 32
```

### Segmentation Example
```bash
python main.py \
    --task segmentation \
    --num-classes 21 \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --mask-suffix "_mask.png" \
    --epochs 100 \
    --batch-size 16
```

## 🎯 Usage Examples

### Using Different DINOv2 Models

**Small Model (Fast Training)**
```bash
python main.py \
    --backbone vit_small_patch14_dinov2 \
    --batch-size 64 \
    --epochs 30
```

**Large Model (High Accuracy)**
```bash
python main.py \
    --backbone vit_large_patch14_dinov2 \
    --batch-size 8 \
    --epochs 100 \
    --lr 5e-5
```

**Partial Fine-Tuning**
```bash
python main.py \
    --fine-tune-strategy partial \
    --unfreeze-blocks 2 \
    --lr 1e-4
```

### With Experiment Tracking
```bash
# Weights & Biases
python main.py --tracker wandb --experiment-name "my-experiment"

# TensorBoard
python main.py --tracker tensorboard

# MLflow
python main.py --tracker mlflow
```

## 🔧 Custom DINOv2 Models

### Using Custom Pre-trained DINOv2 Models

If you have custom DINOv2 pre-trained weights:

```python
# In your custom script
import timm
import torch

# Load custom DINOv2 model
model = timm.create_model(
    'vit_base_patch16_224',  # Your model architecture
    pretrained=False,
    num_classes=0
)

# Load custom weights
state_dict = torch.load('path/to/your/custom_dino_weights.pth')
model.load_state_dict(state_dict)

# Use with Dino2Vision
from models.classifier import DinoClassifier
classifier = DinoClassifier(model, num_classes=10)
```

### Supported Custom Architectures

The framework can work with any Vision Transformer architecture that:
1. Has a `forward_features` method or similar
2. Returns token embeddings
3. Supports patch embedding

## 📁 Project Structure

```
dinovision-project/
├── config/
│   └── training.py          # Training configuration dataclass
├── data/
│   ├── datasets.py          # Dataset implementations
│   ├── transforms.py        # Data transformations
│   └── collate.py           # Data collation functions
├── models/
│   ├── backbone.py          # Backbone model utilities
│   ├── classifier.py        # Classification head
│   └── segmentation.py      # Segmentation head
├── training/
│   ├── trainer.py           # Training loop
│   ├── evaluator.py         # Evaluation functions
│   └── utils.py            # Training utilities
├── tracking/
│   └── tracker.py          # Experiment tracking
├── utils/
│   ├── logging.py          # Logging configuration
│   └── helpers.py          # Helper functions
└── main.py                 # Main entry point
```

## ⚙️ Configuration

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--backbone` | DINOv2 model architecture | `vit_base_patch14_dinov2` |
| `--task` | Task type: classification/segmentation | Required |
| `--num-classes` | Number of output classes | Required |
| `--fine-tune-strategy` | linear/partial/full | `partial` |
| `--unfreeze-blocks` | Number of blocks to unfreeze | `4` |
| `--lr` | Learning rate | `1e-4` |
| `--batch-size` | Batch size | `32` |
| `--epochs` | Training epochs | `20` |

### Full Configuration Example

```bash
python main.py \
    --task segmentation \
    --num-classes 5 \
    --backbone vit_large_patch14_dinov2 \
    --pretrained \
    --fine-tune-strategy partial \
    --unfreeze-blocks 3 \
    --epochs 100 \
    --batch-size 8 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --tracker wandb \
    --experiment-name "custom-dino-experiment" \
    --workers 4 \
    --early-stop 15 \
    --seed 42
```

## 📊 Training Monitoring

### Supported Trackers

1. **Weights & Biases**: Comprehensive experiment tracking
   ```bash
   wandb login
   python main.py --tracker wandb
   ```

2. **TensorBoard**: Traditional ML experiment tracking
   ```bash
   python main.py --tracker tensorboard
   tensorboard --logdir=./logs
   ```

3. **MLflow**: Production experiment tracking
   ```bash
   python main.py --tracker mlflow
   ```

### Metrics Tracked
- Training/validation loss and accuracy
- Learning rates
- Gradient norms
- Model checkpoints
- Hardware utilization

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Adding New Features

1. **New Backbones**: Add to `models/backbone.py`
2. **New Tasks**: Create new head in `models/` directory
3. **New Datasets**: Add to `data/datasets.py`
4. **New Transforms**: Add to `data/transforms.py`

### Reporting Issues

Please report bugs and feature requests via GitHub issues.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Facebook Research for the DINOv2 model
- Ross Wightman for the `timm` library
- Albumentations team for image transformations

---

**Note**: This framework is designed for research and production use. For commercial applications, please ensure compliance with the licenses of the pre-trained models used.
