
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import autocast


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device: torch.device, loss_fn, device_str: str, ablate: set | None = None):
    """Run the model over ``loader`` and return (loss, acc, macro_f1, y_true, y_pred, probs)."""
    model.eval()
    all_y, all_pred, all_prob, total_loss = [], [], [], 0.0
    for batch in loader:
        batch = move(batch, device)
        labels = batch["labels"]
        with autocast(device_type=device_str):
            logits = model(
                ctg_ids=batch["ctg_ids"], ctg_len=batch["ctg_len"],
                ctg_bl_ids=batch["ctg_bl_ids"], ctg_bl_len=batch["ctg_bl_len"],
                bb_input_ids=batch["bb_input_ids"], bb_mask=batch["bb_mask"],
                rb_input_ids=batch["rb_input_ids"], rb_mask=batch["rb_mask"],
                ablate=ablate,
            )
            loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        preds = probs.argmax(1)
        all_y.extend(labels.cpu().numpy())
        all_pred.extend(preds)
        all_prob.extend(probs)

    n = len(loader.dataset)
    ay, ap, apr = np.array(all_y), np.array(all_pred), np.array(all_prob)
    return total_loss / n, accuracy_score(ay, ap), f1_score(ay, ap, average="macro"), ay, ap, apr
