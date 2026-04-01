"""Unified multi-task model
"""

import torch
import torch.nn as nn

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        import os
        from .vgg11 import VGG11Encoder
        from .classification import VGG11Classifier
        from .localization import VGG11Localizer
        from .segmentation import VGG11UNet
        
        self.enc = VGG11Encoder(in_channels)
        
        c_mod = VGG11Classifier(num_breeds, in_channels)
        if os.path.exists(classifier_path): c_mod.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        self.cls_head = nn.Sequential(c_mod.avg, nn.Flatten(), c_mod.hd)
        
        l_mod = VGG11Localizer(in_channels)
        if os.path.exists(localizer_path): l_mod.load_state_dict(torch.load(localizer_path, map_location='cpu'))
        self.loc_head = nn.Sequential(l_mod.avg, nn.Flatten(), l_mod.loc)
        
        u_mod = VGG11UNet(seg_classes, in_channels)
        if os.path.exists(unet_path): u_mod.load_state_dict(torch.load(unet_path, map_location='cpu'))
        self.seg_head = nn.ModuleList([
            u_mod.up4, u_mod.dc4,
            u_mod.up3, u_mod.dc3,
            u_mod.up2, u_mod.dc2,
            u_mod.up1, u_mod.dc1,
            u_mod.up0, u_mod.hd
        ])
        
        # Load backbone from classifier initially
        self.enc.load_state_dict(c_mod.enc.state_dict())

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        bt, fts = self.enc(x, return_features=True)
        
        c_out = self.cls_head(bt)
        l_out = self.loc_head(bt)
        
        u4, d4, u3, d3, u2, d2, u1, d1, u0, hd = self.seg_head
        
        s = u4(bt)
        s = torch.cat([s, fts['f4']], dim=1)
        s = d4(s)
        
        s = u3(s)
        s = torch.cat([s, fts['f3']], dim=1)
        s = d3(s)
        
        s = u2(s)
        s = torch.cat([s, fts['f2']], dim=1)
        s = d2(s)
        
        s = u1(s)
        s = torch.cat([s, fts['f1']], dim=1)
        s = d1(s)
        
        s = u0(s)
        s_out = hd(s)
        
        return {
            'classification': c_out,
            'localization': l_out,
            'segmentation': s_out
        }
