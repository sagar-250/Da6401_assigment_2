"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        
        c = in_channels
        self.b1 = nn.Sequential(nn.Conv2d(c, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.b2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.b3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.b4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.b5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2))

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        bot = self.b5(f4)
        
        if return_features:
            fts = {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4}
            return bot, fts
        return bot