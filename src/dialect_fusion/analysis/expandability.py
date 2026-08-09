"""Utilities demonstrating the model's modularity: growing classes / adding branches."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def expand_model_classes(model, n_new: int, device) -> nn.Module:
    """
    Grow the final Linear layer by ``n_new`` classes, copying existing
    weights and freezing everything except the (expanded) head. This lets
    you add a new emotion class without retraining the backbones.
    """
    old_linear = model.head[-1]
    old_out, in_dim = old_linear.out_features, old_linear.in_features
    new_linear = nn.Linear(in_dim, old_out + n_new).to(device)
    with torch.no_grad():
        new_linear.weight[:old_out] = old_linear.weight
        new_linear.bias[:old_out] = old_linear.bias
        nn.init.normal_(new_linear.weight[old_out:], std=0.02)
        nn.init.zeros_(new_linear.bias[old_out:])
    model.head[-1] = new_linear

    for name, p in model.named_parameters():
        p.requires_grad = "head" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Expanded: {old_out} -> {old_out + n_new} classes")
    print(f"  Trainable params (head only): {trainable:,}")
    return model


def expand_classes_demo(model, device, n_new: int = 1):
    """Return a deep-copied, class-expanded model (does not mutate ``model``)."""
    expanded = copy.deepcopy(model)
    return expand_model_classes(expanded, n_new=n_new, device=device)


FIFTH_BRANCH_NOTE = """
To add a fifth branch (e.g. Arabic):
  1. Load: self.arabert = AutoModel.from_pretrained('aubmindlab/bert-base-arabertv02')
  2. Add:  self.proj_ar = nn.Sequential(LayerNorm(768), Linear(768, 256), GELU())
  3. Expand fused_dim to 1280 (5 x 256)
  4. Update head input: Linear(1280, 512)
  5. Freeze the original 4 branches, train only proj_ar + head
All original branch weights remain exactly unchanged.
"""
