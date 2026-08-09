from __future__ import annotations

import os

import pandas as pd
from sklearn.metrics import classification_report

from dialect_fusion.config import Config


def build_results_table(test_y, test_pred, target_names, test_acc, test_f1,
                         ece_after, T_opt, num_labels, model, cfg: Config) -> pd.DataFrame:
    report_dict = classification_report(test_y, test_pred, target_names=target_names, digits=4, output_dict=True)
    res_df = pd.DataFrame(report_dict).T.round(4)
    res_df = res_df.drop(columns=["support"], errors="ignore")

    print("=" * 65)
    print("  Results - Chittagong Bangla Emotion Classification")
    print("  Model: Multi-Input Late Fusion (4-column)")
    print("=" * 65)
    print(res_df.to_string())
    print("=" * 65)
    print(f"  Accuracy          : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Macro-F1          : {test_f1:.4f}")
    print(f"  ECE (calibrated)  : {ece_after:.4f}")
    print(f"  Temp T            : {T_opt:.4f}")
    print(f"  # Classes         : {num_labels}")
    print(f"  # Parameters      : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print("-" * 65)
    print(f"  BanglaBERT        : {cfg.banglabert_name}")
    print(f"  Char hidden       : {cfg.char_hidden}")
    print(f"  Char embed        : {cfg.char_embed_dim}")
    print(f"  Head dropout      : 0.45 / 0.25")
    print(f"  Weight decay      : {cfg.weight_decay}")
    print(f"  Frozen layers     : 0-{cfg.freeze_bottom_layers - 1}")
    print(f"  Epochs (max)      : {cfg.epochs}")
    print(f"  Early stop        : {cfg.early_stop_patience}")
    print("=" * 65)

    res_df.to_csv(os.path.join(cfg.fig_dir, "K_results_table.csv"))
    print(f"\nAll figures -> {cfg.fig_dir}")
    print(f"Model        -> {cfg.save_dir}")
    return res_df
