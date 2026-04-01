"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError()

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        px, py, pw, ph = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
        tx, ty, tw, th = target_boxes[:,0], target_boxes[:,1], target_boxes[:,2], target_boxes[:,3]
        
        px1 = px - pw/2
        py1 = py - ph/2
        px2 = px + pw/2
        py2 = py + ph/2
        
        tx1 = tx - tw/2
        ty1 = ty - th/2
        tx2 = tx + tw/2
        ty2 = ty + th/2
        
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)
        
        iw = torch.clamp(ix2 - ix1, min=0)
        ih = torch.clamp(iy2 - iy1, min=0)
        
        ints = iw * ih
        ua = pw*ph + tw*th - ints
        
        iou = ints / (ua + self.eps)
        ls = 1.0 - iou
        
        if self.reduction == "mean":
            return ls.mean()
        if self.reduction == "sum":
            return ls.sum()
        return ls