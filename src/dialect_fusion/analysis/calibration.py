"""Expected Calibration Error + post-hoc temperature scaling."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from torch.amp import autocast

from dialect_fusion.training.evaluate import move


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error: sum_b |B_b|/n * |acc(B_b) - conf(B_b)|."""
    confs = probs.max(1)
    preds = probs.argmax(1)
    correct = (preds == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confs[mask].mean())
    return ece


class TemperatureScaler(nn.Module):
    """Wraps a model with a single learned scalar T dividing the logits."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        self.T = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, **kwargs):
        logits = self.model(**kwargs)
        return logits / self.T.clamp(min=0.1)


def fit_temperature(base_model, loader, device: torch.device, device_str: str):
    """Fit T on a held-out (validation) loader via LBFGS + NLL loss."""
    ts = TemperatureScaler(base_model).to(device)
    opt = torch.optim.LBFGS([ts.T], lr=0.01, max_iter=200)

    all_logits, all_labels = [], []
    base_model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            with autocast(device_type=device_str):
                logits = base_model(
                    ctg_ids=batch["ctg_ids"], ctg_len=batch["ctg_len"],
                    ctg_bl_ids=batch["ctg_bl_ids"], ctg_bl_len=batch["ctg_bl_len"],
                    bb_input_ids=batch["bb_input_ids"], bb_mask=batch["bb_mask"],
                    rb_input_ids=batch["rb_input_ids"], rb_mask=batch["rb_mask"],
                )
            all_logits.append(logits.float())
            all_labels.append(batch["labels"])
    lc = torch.cat(all_logits)
    yc = torch.cat(all_labels)

    def _closure():
        opt.zero_grad()
        # NOTE: plain .backward() — LBFGS is incompatible with GradScaler.
        loss = F.cross_entropy(lc / ts.T.clamp(min=0.1), yc)
        loss.backward()
        return loss

    opt.step(_closure)
    print(f"Optimal T = {ts.T.item():.4f}")
    return ts, ts.T.item()


@torch.no_grad()
def calibrated_probs(ts_model, loader, device: torch.device, device_str: str) -> np.ndarray:
    ts_model.eval()
    out = []
    for batch in loader:
        batch = move(batch, device)
        with autocast(device_type=device_str):
            logits = ts_model(
                ctg_ids=batch["ctg_ids"], ctg_len=batch["ctg_len"],
                ctg_bl_ids=batch["ctg_bl_ids"], ctg_bl_len=batch["ctg_bl_len"],
                bb_input_ids=batch["bb_input_ids"], bb_mask=batch["bb_mask"],
                rb_input_ids=batch["rb_input_ids"], rb_mask=batch["rb_mask"],
            )
        out.extend(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.array(out)


def plot_calibration(test_probs, cal_probs, test_y, label_map: dict, num_labels: int,
                      ece_before: float, ece_after: float, T_opt: float, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for probs_set, ax, title, c in [
        (test_probs, axes[0], "Before Calibration", "#e74c3c"),
        (cal_probs, axes[1], "After Temperature Scaling", "#2ecc71"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")
        for i in range(0, min(num_labels, 12)):
            bl = (test_y == i).astype(int)
            if bl.sum() < 5:
                continue
            fp, mp = calibration_curve(bl, probs_set[:, i], n_bins=8)
            ax.plot(mp, fp, alpha=0.35, lw=1)
        confs_all = probs_set.max(1)
        correct_all = (probs_set.argmax(1) == test_y).astype(int)
        fp_all, mp_all = calibration_curve(correct_all, confs_all, n_bins=12)
        ax.plot(mp_all, fp_all, c, lw=2, label="Overall", zorder=5)
        ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction Positives",
               title=title, xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        ax.set_aspect("equal")

    brier = {label_map[i]: brier_score_loss((test_y == i).astype(int), cal_probs[:, i]) for i in range(num_labels)}
    bdf = pd.Series(brier).sort_values(ascending=False)
    colors_b = ["#e74c3c" if v > np.percentile(list(brier.values()), 75) else "#3498db" for v in bdf.values]
    axes[2].barh(bdf.index, bdf.values, color=colors_b, edgecolor="white", height=0.72)
    axes[2].axvline(bdf.mean(), color="gray", ls="--", lw=1.2, label=f"mean={bdf.mean():.3f}")
    axes[2].set(xlabel="Brier Score (lower=better)", title="Per-Class Brier (red=worst)")
    axes[2].legend()

    plt.suptitle(f"Calibration Analysis  |  ECE before={ece_before:.4f} -> after={ece_after:.4f}  |  T={T_opt:.3f}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "D_calibration.pdf"), bbox_inches="tight")
    plt.close(fig)


def run_calibration_analysis(model, val_loader, test_loader, test_probs, test_y, label_map, num_labels,
                              device, device_str, fig_dir) -> dict:
    ece_before = compute_ece(test_probs, test_y)
    print(f"ECE before temperature scaling: {ece_before:.4f}")

    ts_model, T_opt = fit_temperature(model, val_loader, device, device_str)
    cal_probs = calibrated_probs(ts_model, test_loader, device, device_str)

    ece_after = compute_ece(cal_probs, test_y)
    print(f"ECE  after temperature scaling : {ece_after:.4f}")
    print(f"ECE  improvement               : {ece_before - ece_after:.4f}")

    plot_calibration(test_probs, cal_probs, test_y, label_map, num_labels, ece_before, ece_after, T_opt, fig_dir)
    return {"ece_before": ece_before, "ece_after": ece_after, "T_opt": T_opt, "cal_probs": cal_probs}
