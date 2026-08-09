
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
   

    def __init__(self, num_classes: int, gamma: float = 2.0, smoothing: float = 0.1,
                 weight: torch.Tensor | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()  # fp16 -> fp32 safety under autocast
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, targets, weight=self.weight, reduction="none")
        uniform = -log_probs.mean(dim=-1)
        ce = (1 - self.smoothing) * nll + self.smoothing * uniform
        pt = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1 - pt.detach()) ** self.gamma
        return (focal_w * ce).mean()
