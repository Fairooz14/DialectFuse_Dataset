from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dialect_fusion.data.dataset import FusionDataset, make_fusion_collate
from dialect_fusion.training.evaluate import move


def default_case(cfg) -> dict:
    return {
        cfg.col_ctg_bangla: "ইচ্ছে আসিল হোনো মিক্কে ঘুইরতু যাইয়্যুম",
        cfg.col_ctg_banglish: "icche asilo hono mikke ghuirtu jaiiyum",
        cfg.col_std_bangla: "ইচ্ছা ছিল কোনো জায়গায় ঘুরতে যাব",
        cfg.col_english: "I wanted to go somewhere",
        cfg.col_label: "desire",
    }


def run_case_study(model, case: dict, cfg, bangla_tok, roberta_tok, char_vocab,
                    label_map: dict, rev_map: dict, device, device_str, fig_dir: str):
    case_df = pd.DataFrame([case])
    true_label = case[cfg.col_label]
    case_ds = FusionDataset(case_df, [rev_map[true_label]], cfg, bangla_tok, roberta_tok,
                             char_vocab, cfg.max_seq_len, augment=False)
    collate_fn = make_fusion_collate(char_vocab.pad_idx)
    case_batch = move(collate_fn([case_ds[0]]), device)

    model.eval()
    with torch.no_grad():
        logits = model(
            ctg_ids=case_batch["ctg_ids"], ctg_len=case_batch["ctg_len"],
            ctg_bl_ids=case_batch["ctg_bl_ids"], ctg_bl_len=case_batch["ctg_bl_len"],
            bb_input_ids=case_batch["bb_input_ids"], bb_mask=case_batch["bb_mask"],
            rb_input_ids=case_batch["rb_input_ids"], rb_mask=case_batch["rb_mask"],
        )
    case_probs = F.softmax(logits.float(), dim=1).squeeze().cpu().numpy()
    sorted_idx = np.argsort(case_probs)[::-1]

    top10_idx = sorted_idx[:10]
    top10_lbl = [label_map[i] for i in top10_idx]
    top10_prob = case_probs[top10_idx]
    bar_colors = ["#2ecc71" if label_map[i] == true_label else "#3498db" for i in top10_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(top10_lbl[::-1], top10_prob[::-1] * 100, color=bar_colors[::-1], edgecolor="white", height=0.7)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set(xlabel="Predicted Probability (%)",
           title=f'Case Study: "{case[cfg.col_ctg_bangla]}"',
           xlim=(0, max(top10_prob) * 100 * 1.25))
    correct_patch = mpatches.Patch(color="#2ecc71", label=f"Ground truth ({true_label})")
    other_patch = mpatches.Patch(color="#3498db", label="Other predictions")
    ax.legend(handles=[correct_patch, other_patch], fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "A_case_study.pdf"), bbox_inches="tight")
    plt.close(fig)

    pred_emotion = label_map[sorted_idx[0]]
    is_correct = pred_emotion == true_label
    print(f"Predicted : {pred_emotion}  ({'CORRECT' if is_correct else 'WRONG'})")
    print(f"Confidence: {case_probs[sorted_idx[0]]*100:.1f}%")
    print(f"True label: {true_label}  (rank #{list(sorted_idx).index(rev_map[true_label]) + 1} in predictions)")
    return case_probs, sorted_idx
