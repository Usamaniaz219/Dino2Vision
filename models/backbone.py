
import timm
import torch.nn as nn
from typing import List, Optional

def build_backbone(backbone_name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Build a backbone model from timm.
    """
    try:
        # num_classes=0 returns features (pooled) for timm models
        model = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0,
            **kwargs
        )
        return model
    except Exception as e:
        raise ValueError(f"Failed to create model {backbone_name}: {e}")


def get_trainable_parameters(backbone: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in backbone.parameters() if p.requires_grad)


def apply_fine_tune_strategy(
    backbone: nn.Module, 
    strategy: str = "partial", 
    unfreeze_blocks: int = 4,
    unfreeze_layers: Optional[List[str]] = None
) -> int:
    """
    Apply fine-tuning strategy to backbone.
    """
    strategy = strategy.lower()
    valid_strategies = {"linear", "partial", "full"}
    if strategy not in valid_strategies:
        raise ValueError(f"fine_tune_strategy must be one of {valid_strategies}")

    # Handle full strategy first
    if strategy == "full":
        for p in backbone.parameters():
            p.requires_grad = True
        return get_trainable_parameters(backbone)

    # Freeze all for linear and partial strategies
    for p in backbone.parameters():
        p.requires_grad = False

    if strategy == "linear":
        # Only classifier/head will be trainable (handled externally)
        return get_trainable_parameters(backbone)

    # Partial fine-tuning strategy
    if unfreeze_layers:
        # Unfreeze specific layers by name
        for name, param in backbone.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):
                param.requires_grad = True
    else:
        # Unfreeze last N blocks (original logic)
        if hasattr(backbone, 'blocks'):
            blocks = list(backbone.blocks)
            for blk in blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        
        # Alternative pattern for models with layers instead of blocks
        elif hasattr(backbone, 'layers'):
            layers = list(backbone.layers)
            for layer in layers[-unfreeze_blocks:]:
                for p in layer.parameters():
                    p.requires_grad = True
        
        # Handle ResNet-style models
        elif hasattr(backbone, 'layer4'):
            # Unfreeze last layer and potentially previous ones
            layers_to_unfreeze = ['layer4']
            if unfreeze_blocks > 1:
                layers_to_unfreeze.append('layer3')
            if unfreeze_blocks > 2:
                layers_to_unfreeze.append('layer2')
            
            for name, param in backbone.named_parameters():
                if any(layer_name in name for layer_name in layers_to_unfreeze):
                    param.requires_grad = True

    # Always unfreeze normalization layers and classifier
    for name, param in backbone.named_parameters():
        if 'norm' in name or 'bn' in name or 'classifier' in name or 'head' in name:
            param.requires_grad = True

    return get_trainable_parameters(backbone)


def print_trainable_parameters(backbone: nn.Module, title: str = "Trainable Parameters"):
    """Print information about trainable parameters."""
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = get_trainable_parameters(backbone)
    
    print(f"{title}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
