"""Localization modules
"""

import torch
import torch.nn as nn

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
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
        
        from .layers import CustomDropout
        drp = CustomDropout(dropout_p)
        self.loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            drp,
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            drp,
            nn.Linear(4096, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        bt = self.enc(x, False)
        tmp = self.avg(bt)
        tmp = tmp.view(tmp.size(0), -1)
        bx = self.loc(tmp)
        return bx
