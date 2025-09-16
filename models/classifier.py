import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any, Union
import math
import warnings
import logging
from functools import partial
import timm


class DinoV2BackboneWrapper(nn.Module):
    """
    Wrapper for DINOv3 backbones to provide consistent interface.
    """
    
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self._setup_hooks()
        
    def _setup_hooks(self):
        """Setup hooks for feature extraction if needed."""
        self.features = {}
        
        # Register forward hook to capture intermediate features
        if hasattr(self.backbone, 'blocks'):
            # For standard transformer blocks
            self.backbone.blocks[-1].register_forward_hook(
                self._get_features_hook('last_block')
            )
    
    def _get_features_hook(self, name: str):
        """Create a hook to capture features."""
        def hook(module, input, output):
            self.features[name] = output
        return hook
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning pooled features."""
        return self.backbone(x)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning all tokens (including cls token)."""
        if hasattr(self.backbone, 'forward_features'):
            return self.backbone.forward_features(x)
        else:
            # Fallback for DINOv3 models
            return self.backbone.get_intermediate_layers(x, n=1, return_class_token=True)[0]
    
    def get_grid_size(self, x: torch.Tensor) -> Tuple[int, int]:
        """Get the grid size for the given input."""
        if hasattr(self.backbone, 'patch_embed'):
            patch_embed = self.backbone.patch_embed
            if hasattr(patch_embed, 'grid_size'):
                grid_size = patch_embed.grid_size
                if isinstance(grid_size, (tuple, list)):
                    return tuple(grid_size)
                return (grid_size, grid_size)
        
        # Infer from input shape and patch size
        B, C, H, W = x.shape
        patch_size = self.get_patch_size()
        gh, gw = H // patch_size[0], W // patch_size[1]
        return (gh, gw)
    
    def get_patch_size(self) -> Tuple[int, int]:
        """Get the patch size."""
        if hasattr(self.backbone, 'patch_embed'):
            patch_embed = self.backbone.patch_embed
            if hasattr(patch_embed, 'patch_size'):
                ps = patch_embed.patch_size
                if isinstance(ps, int):
                    return (ps, ps)
                return tuple(ps)
        return (14, 14)  # Default for DINOv3
    
    @property
    def num_features(self) -> int:
        """Get the number of features."""
        for attr in ['num_features', 'embed_dim', 'hidden_size']:
            if hasattr(self.backbone, attr):
                return getattr(self.backbone, attr)
        raise ValueError("Cannot determine feature dimension")


class DinoClassifier(nn.Module):
    """
    Production-ready DINOv3 classifier for various tasks.
    """
    
    def __init__(
        self, 
        backbone: nn.Module, 
        num_classes: int,
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        head_type: str = "linear"
    ):
        super().__init__()
        self.backbone = DinoV2BackboneWrapper(backbone)
        feature_dim = self.backbone.num_features
        
        # Build classification head
        if head_type == "linear":
            self.head = nn.Linear(feature_dim, num_classes)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.BatchNorm1d(feature_dim // 2) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, num_classes)
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        feats = self.backbone(x)  # [B, C]
        
        # Apply classifier
        return self.head(feats)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# Factory function for easy model creation
def create_dino_classifier(
    model_name: str = "vit_base_patch14_dinov2",
    num_classes: int = 1000,
    pretrained: bool = True,
    **kwargs
) -> DinoClassifier:
    """Create a DINOv3 classifier."""
    try:
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        return DinoClassifier(backbone, num_classes, **kwargs)
    except ImportError:
        raise ImportError("timm library is required for DINOv3 models")
    except Exception as e:
        raise ValueError(f"Failed to create model {model_name}: {e}")
