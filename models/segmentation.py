"""Segmentation model
"""

import torch
import torch.nn as nn

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        from .vgg11 import VGG11Encoder
        self.enc = VGG11Encoder(in_channels)
        
        self.up4 = nn.Sequential(nn.ConvTranspose2d(512, 512, 2, stride=2), nn.ReLU(True))
        self.dc4 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True))
        
        self.up3 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU(True))
        self.dc3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True))
        
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(True))
        self.dc2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True))
        
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(True))
        self.dc1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True))
        
        self.up0 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(True))
        self.hd = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bt, fts = self.enc(x, return_features=True)
        
        d4 = self.up4(bt)
        d4 = torch.cat([d4, fts['f4']], dim=1)
        d4 = self.dc4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, fts['f3']], dim=1)
        d3 = self.dc3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, fts['f2']], dim=1)
        d2 = self.dc2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, fts['f1']], dim=1)
        d1 = self.dc1(d1)
        
        d0 = self.up0(d1)
        st = self.hd(d0)
        
        return st
