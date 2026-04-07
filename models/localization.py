"""Localization modules
"""

import torch
import torch.nn as nn

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.2):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        from .vgg11 import VGG11Encoder
        self.enc = VGG11Encoder(in_channels)
        self.avg = nn.AdaptiveAvgPool2d((7, 7))

        self.loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(256, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format,
            normalized to [0, 1].
        """
        feat = self.enc(x, False)
        feat = self.avg(feat)
        feat = feat.flatten(1)
        raw = self.loc(feat)

        # Constrain to valid box range.
        # cx, cy in [0,1], w, h in (0,1]
        cxcy = torch.sigmoid(raw[:, 0:2])
        wh = torch.sigmoid(raw[:, 2:4]).clamp(min=1e-3, max=1.0)
        normalized_coords = torch.cat([cxcy, wh], dim=1)

        # 2. Rescale to pixel space (224x224)
        # This does not affect the weights, only the output of this function
        pixel_coords = normalized_coords * 224.0
        
        return pixel_coords
