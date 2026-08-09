from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from torch.amp import autocast

from dialect_fusion.data.dataset import FusionDataset, make_fusion_collate
from dialect_fusion.training.evaluate import move
from dialect_fusion.utils.plotting import ensure_bengali_font


def make_predict_fn(model, fixed_row: dict, vary_col: str, cfg, bangla_tok, roberta_tok,
                     char_vocab, device, device_str, chunk_size: int = 8):
    """Return a SHAP/LIME-compatible predict_fn(list[str]) -> probs[N, C]."""
    collate_fn = make_fusion_collate(char_vocab.pad_idx)

    def predict_fn(texts):
        probs_list = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            rows = []
            for t in chunk:
                row = dict(fixed_row)
                row[vary_col] = t
                rows.append(row)
            tmp_df = pd.DataFrame(rows)
            tmp_ds = FusionDataset(tmp_df, [0] * len(rows), cfg, bangla_tok, roberta_tok,
                                    char_vocab, cfg.max_seq_len)
            tmp_batch = move(collate_fn([tmp_ds[j] for j in range(len(rows))]), device)
            with torch.no_grad(), autocast(device_type=device_str):
                logits = model(
                    ctg_ids=tmp_batch["ctg_ids"], ctg_len=tmp_batch["ctg_len"],
                    ctg_bl_ids=tmp_batch["ctg_bl_ids"], ctg_bl_len=tmp_batch["ctg_bl_len"],
                    bb_input_ids=tmp_batch["bb_input_ids"], bb_mask=tmp_batch["bb_mask"],
                    rb_input_ids=tmp_batch["rb_input_ids"], rb_mask=tmp_batch["rb_mask"],
                )
            probs_list.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        return np.vstack(probs_list)

    return predict_fn


def run_shap_analysis(model, case: dict, case_probs: np.ndarray, cfg, bangla_tok, roberta_tok,
                       char_vocab, target_names: list, label_map: dict, device, device_str, fig_dir: str):
    model.eval()
    bn_font, bn_title_font = ensure_bengali_font()

    bangla_masker = shap.maskers.Text(r" ")     # whitespace only - preserves ZWJ/nukta
    latin_masker = shap.maskers.Text(r"\S+")    # word boundary - correct for ASCII

    top_class = int(case_probs.argmax())
    top_name = label_map[top_class]
    print(f"Explaining top predicted class: '{top_name}' (index {top_class})")

    branch_info = [
        (cfg.col_ctg_bangla, "Ctg Bangla (char-BiLSTM)", case[cfg.col_ctg_bangla], bangla_masker, True),
        (cfg.col_ctg_banglish, "Ctg Banglish (char-BiLSTM)", case[cfg.col_ctg_banglish], latin_masker, False),
        (cfg.col_std_bangla, "Std Bangla (BanglaBERT)", case[cfg.col_std_bangla], bangla_masker, True),
        (cfg.col_english, "English (RoBERTa-base)", case[cfg.col_english], latin_masker, False),
    ]

    fig, axes = plt.subplots(len(branch_info), 1, figsize=(13, 4 * len(branch_info)))
    fig.text(0.5, 0.98, f"SHAP Token Attribution - predicted: '{top_name}'",
              ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.965, case[cfg.col_ctg_bangla], ha="center", va="top", fontproperties=bn_title_font)

    shap_results = {}
    for ax, (col, title, text, masker, is_bengali) in zip(axes, branch_info):
        pred_fn = make_predict_fn(model, case, col, cfg, bangla_tok, roberta_tok, char_vocab, device, device_str)
        explainer = shap.Explainer(pred_fn, masker, output_names=target_names, algorithm="permutation", max_evals=256)
        sv = explainer([text])
        shap_results[col] = sv

        words = list(sv.data[0])
        sv_class = sv.values[0, :len(words), top_class]
        colors_sv = ["#e74c3c" if v > 0 else "#3498db" for v in sv_class]

        ax.bar(range(len(words)), sv_class, color=colors_sv, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("SHAP value")
        ax.set_title(title + f"  ->  class: '{top_name}'")

        ax.set_xticks(range(len(words)))
        labels = ax.set_xticklabels(words, rotation=90, ha="right", fontsize=10)
        if is_bengali:
            for lbl in labels:
                lbl.set_fontproperties(bn_font)

    pos_patch = mpatches.Patch(color="#e74c3c", label="Pushes toward prediction")
    neg_patch = mpatches.Patch(color="#3498db", label="Pushes against prediction")
    fig.legend(handles=[pos_patch, neg_patch], loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 0.955])
    plt.savefig(os.path.join(fig_dir, "E_shap_branches.pdf"), bbox_inches="tight")
    plt.close(fig)
    return shap_results


def run_lime_analysis(model, case: dict, case_probs: np.ndarray, cfg, bangla_tok, roberta_tok,
                       char_vocab, target_names: list, device, device_str, fig_dir: str, seed: int = 42):
    lime_explainer = LimeTextExplainer(class_names=target_names, split_expression=r"\S+", bow=False, random_state=seed)
    predict_fn = make_predict_fn(model, case, cfg.col_english, cfg, bangla_tok, roberta_tok, char_vocab, device, device_str)

    lime_exp = lime_explainer.explain_instance(
        case[cfg.col_english], predict_fn, num_features=10, num_samples=500, top_labels=3,
    )

    # Sort by descending predicted probability, not by class index.
    avail = set(lime_exp.available_labels())
    top3_labels = [int(i) for i in np.argsort(case_probs)[::-1] if int(i) in avail][:3]
    fig, axes = plt.subplots(1, len(top3_labels), figsize=(5 * len(top3_labels), 5))
    if len(top3_labels) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, top3_labels):
        feats = lime_exp.as_list(label=lbl)
        words = [f[0] for f in feats]
        values = [f[1] for f in feats]
        colors_l = ["#e74c3c" if v > 0 else "#3498db" for v in values]
        ax.barh(words, values, color=colors_l, edgecolor="white", height=0.72)
        ax.axvline(0, color="black", lw=0.8)
        ax.set(xlabel="LIME Weight", title=f"Class: '{target_names[lbl]}'")
        ax.invert_yaxis()

    plt.suptitle(f"LIME - English branch\n{case[cfg.col_english]}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "F_lime_explanation.pdf"), bbox_inches="tight")
    plt.close(fig)
    return lime_exp
