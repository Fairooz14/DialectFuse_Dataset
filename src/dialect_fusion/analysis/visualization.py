from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from dialect_fusion.utils.plotting import PAL10, PAL20


def plot_dataset_stats(train_df: pd.DataFrame, cfg, fig_dir: str) -> None:
    cnt = train_df[cfg.col_label].value_counts().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(cnt) * 0.38)))

    colors = [PAL20[i % 20] for i in range(len(cnt))]
    hb = axes[0].barh(cnt.index, cnt.values, color=colors, edgecolor="white", height=0.75)
    axes[0].bar_label(hb, padding=3, fontsize=8)
    axes[0].axvline(cnt.mean(), color="crimson", ls="--", lw=1.3, label=f"Mean = {cnt.mean():.0f}")
    axes[0].set(xlabel="Count", title="Emotion Distribution - Training Set")
    axes[0].legend()

    col_lens = {c: train_df[c].astype(str).str.split().str.len() for c in cfg.all_text_cols}
    col_labels = ["Ctg Bangla", "Ctg Banglish", "Std Bangla", "English"]
    bp = axes[1].boxplot([col_lens[c] for c in cfg.all_text_cols], labels=col_labels,
                          patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], PAL10):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].set(ylabel="Word Count", title="Token Length Distribution per Column")
    axes[1].tick_params(axis="x", rotation=20)

    plt.suptitle("Dataset Statistics", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_data_stats.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: dict, best_epoch: int, fig_dir: str) -> float:
    ep = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    axes[0].plot(ep, history["train_loss"], "o-", label="Train", color="#2196F3", lw=1.8, ms=4)
    axes[0].plot(ep, history["val_loss"], "s-", label="Val", color="#FF5722", lw=1.8, ms=4)
    axes[0].axvline(best_epoch, color="green", ls="--", lw=1.3, label=f"Best ep {best_epoch}")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss Curves")
    axes[0].legend()

    axes[1].plot(ep, history["val_f1"], "o-", label="Val Macro-F1", color="#4CAF50", lw=1.8, ms=4)
    axes[1].plot(ep, history["val_acc"], "s-", label="Val Accuracy", color="#9C27B0", lw=1.8, ms=4)
    axes[1].axvline(best_epoch, color="green", ls="--", lw=1.3)
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation Metrics", ylim=(0, 1))
    axes[1].legend()

    gap = [v / max(t, 1e-9) for t, v in zip(history["train_loss"], history["val_loss"])]
    axes[2].plot(ep, gap, "o-", color="#E91E63", lw=1.8, ms=4)
    axes[2].axhline(5, color="orange", ls="--", lw=1, label="5x (mild overfit)")
    axes[2].axhline(10, color="red", ls="--", lw=1, label="10x (severe overfit)")
    axes[2].axvline(best_epoch, color="green", ls="--", lw=1.3)
    axes[2].set(xlabel="Epoch", ylabel="Val Loss / Train Loss", title="Overfit Gap Ratio (target: <5x)")
    axes[2].legend(fontsize=8)

    plt.suptitle("Training History - 4-Column Late Fusion", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_training_history.pdf"), bbox_inches="tight")
    plt.close(fig)

    final_gap = history["val_loss"][-1] / max(history["train_loss"][-1], 1e-9)
    status = "acceptable" if final_gap < 5 else "still overfitting - see recommendations"
    print(f"Final train/val loss gap: {final_gap:.1f}x  ({status})")
    return final_gap


def _draw_confusion_matrix(data, labels, title, fmt, save_path, threshold=None):
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(data, cmap="Blues", square=True, linewidths=0.4, linecolor="white",
                xticklabels=labels, yticklabels=labels, cbar=True, annot=False,
                cbar_kws={"shrink": 0.75, "pad": 0.02})

    max_val = data.max()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if threshold is not None and val < threshold:
                continue
            color = "white" if val > max_val * 0.5 else "black"
            ax.text(j + 0.5, i + 0.5, format(val, fmt), ha="center", va="center", fontsize=8, color=color)

    ax.set_title(title, fontsize=20, fontweight="bold", pad=25)
    ax.set_xlabel("Predicted Label", fontsize=25, labelpad=15)
    ax.set_ylabel("True Label", fontsize=15, labelpad=15)
    plt.xticks(rotation=90, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(test_y, test_pred, target_names, fig_dir: str) -> None:
    cm = confusion_matrix(test_y, test_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    _draw_confusion_matrix(cm, target_names, "Confusion Matrix - Raw Counts", "d",
                            os.path.join(fig_dir, "confusion_matrix_raw.png"))
    _draw_confusion_matrix(cm_n, target_names, "Confusion Matrix - Row Normalized", ".2f",
                            os.path.join(fig_dir, "confusion_matrix_normalized.png"))


def plot_per_class_metrics(test_y, test_pred, label_map: dict, num_labels: int, fig_dir: str) -> pd.DataFrame:
    prec, rec, f1c, sup = precision_recall_fscore_support(test_y, test_pred, average=None,
                                                            labels=list(range(num_labels)))
    metrics_df = pd.DataFrame({
        "Emotion": [label_map[i] for i in range(num_labels)],
        "Precision": prec, "Recall": rec, "F1": f1c, "Support": sup,
    }).sort_values("F1", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, num_labels * 0.42)))
    for ax, metric, color in zip(axes, ["Precision", "Recall", "F1"], ["#3498db", "#e67e22", "#2ecc71"]):
        order = metrics_df.sort_values(metric, ascending=True)
        hbars = ax.barh(order["Emotion"], order[metric], color=color, alpha=0.82, edgecolor="white", height=0.72)
        ax.bar_label(hbars, fmt="%.3f", padding=3, fontsize=7.5)
        ax.axvline(order[metric].mean(), color="crimson", ls="--", lw=1.2, label=f"mean={order[metric].mean():.3f}")
        ax.set(xlim=(0, 1.17), title=metric)
        ax.legend(fontsize=8.5)

    plt.suptitle("Per-Class Metrics - Test Set", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "C_per_class_metrics.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(metrics_df.round(4).to_string(index=False))
    return metrics_df
