import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging
from .classifier import DinoV2BackboneWrapper


class DinoV2SegHead(nn.Module):
    """
    Production-ready DINOv3 segmentation head.
    """
    
    def __init__(
        self, 
        backbone: nn.Module, 
        num_classes: int, 
        upsample_mode: str = "bilinear",
        decoder_channels: Tuple[int, ...] = (384, 192, 96),
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        align_corners: bool = False,
        use_attention: bool = False
    ):
        super().__init__()
        self.backbone = DinoV3BackboneWrapper(backbone)
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.num_classes = num_classes
        
        # Get feature dimension
        c = self.backbone.num_features
        
        # Build decoder
        self.proj = nn.Conv2d(c, c, 1)
        self.decoder = self._build_decoder(
            in_channels=c,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            use_batch_norm=use_batch_norm,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Identity()
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(c, c // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 4, 1, 1),
                nn.Sigmoid()
            )
        
        self._init_weights()

    def _build_decoder(
        self,
        in_channels: int,
        decoder_channels: Tuple[int, ...],
        num_classes: int,
        use_batch_norm: bool,
        dropout: float
    ) -> nn.Sequential:
        """Build the decoder network."""
        layers = []
        current_channels = in_channels
        
        for out_channels in decoder_channels:
            layers.extend([
                nn.Conv2d(current_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            ])
            current_channels = out_channels
        
        # Final classification layer
        layers.append(nn.Conv2d(current_channels, num_classes, 1))
        
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _tokens_to_feature_map(
        self, 
        tokens: torch.Tensor, 
        grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Convert transformer tokens to 2D feature map."""
        B, N, C = tokens.shape
        gh, gw = grid_size
        expected_tokens = gh * gw
        
        # Handle class token (DINOv3 typically includes it)
        if N == expected_tokens + 1:
            tokens = tokens[:, 1:, :]  # Remove class token
        elif N != expected_tokens:
            logging.warning(f"Token count {N} doesn't match expected {expected_tokens}")
            # Try to handle gracefully by taking first N tokens
            if N > expected_tokens:
                tokens = tokens[:, :expected_tokens, :]
            else:
                # Pad if necessary (shouldn't happen with proper config)
                padding = torch.zeros(B, expected_tokens - N, C, device=tokens.device)
                tokens = torch.cat([tokens, padding], dim=1)
        
        # Reshape to spatial format
        return tokens.permute(0, 2, 1).contiguous().view(B, C, gh, gw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, _, H, W = x.shape
        
        try:
            # Extract tokens using DINOv3's feature extraction
            tokens = self.backbone.forward_features(x)
            
            # Get grid size
            grid_size = self.backbone.get_grid_size(x)
            
            # Convert tokens to feature map
            fmap = self._tokens_to_feature_map(tokens, grid_size)
            
            # Apply projection and attention
            fmap = self.proj(fmap)
            fmap = fmap * self.attention(fmap)
            
            # Upsample to input resolution
            fmap = F.interpolate(
                fmap, 
                size=(H, W), 
                mode=self.upsample_mode, 
                align_corners=self.align_corners if self.upsample_mode == 'bilinear' else None
            )
            
            # Decode to segmentation map
            return self.decoder(fmap)
            
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            raise

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# Factory function for easy model creation
def create_dino_segmentation(
    model_name: str = "vit_base_patch14_dinov2",
    num_classes: int = 21,
    pretrained: bool = True,
    **kwargs
) -> DinoV3SegHead:
    """Create a DINOv3 segmentation model."""
    try:
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        return DinoV3SegHead(backbone, num_classes, **kwargs)
    except ImportError:
        raise ImportError("timm library is required for DINOv3 models")
    except Exception as e:
        raise ValueError(f"Failed to create model {model_name}: {e}")
