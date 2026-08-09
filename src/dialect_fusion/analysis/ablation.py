"""Branch-ablation study: zero out each branch (or combination) and re-evaluate."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from dialect_fusion.training.evaluate import evaluate

ABLATION_CONFIGS = {
    "Full model": set(),
    "- Ctg Bangla": {"ctg"},
    "- Ctg Banglish": {"banglish"},
    "- Std Bangla": {"bangla"},
    "- English": {"english"},
    "Ctg Bangla only": {"banglish", "bangla", "english"},
    "Ctg Banglish only": {"ctg", "bangla", "english"},
    "Std Bangla only": {"ctg", "banglish", "english"},
    "English only": {"ctg", "banglish", "bangla"},
    "Char branches only": {"bangla", "english"},
    "Transformer only": {"ctg", "banglish"},
}


def run_ablation_study(model, test_loader, device, loss_fn, device_str, fig_dir: str) -> pd.DataFrame:
    ablation_results = {}
    for name, ablate_set in ABLATION_CONFIGS.items():
        _, acc, f1, *_ = evaluate(model, test_loader, device, loss_fn, device_str, ablate=ablate_set)
        ablation_results[name] = {"acc": acc, "f1": f1}
        print(f"  {name:<30s}  acc={acc:.4f}  macro-F1={f1:.4f}")

    full_f1 = ablation_results["Full model"]["f1"]
    ab_df = pd.DataFrame(ablation_results).T.reset_index()
    ab_df.columns = ["Config", "acc", "f1"]
    ab_df["delta"] = ab_df["f1"] - full_f1
    ab_df = ab_df.sort_values("f1", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(ab_df) * 0.42)))
    bar_c = ["#2ecc71" if r["Config"] == "Full model" else "#e74c3c" if r["delta"] < -0.05 else "#3498db"
             for _, r in ab_df.iterrows()]
    hb = axes[0].barh(ab_df["Config"], ab_df["f1"] * 100, color=bar_c, edgecolor="white", height=0.7)
    axes[0].bar_label(hb, fmt="%.1f%%", padding=3, fontsize=8.5)
    axes[0].axvline(full_f1 * 100, color="green", ls="--", lw=1.3, label=f"Full={full_f1*100:.1f}%")
    axes[0].set(xlabel="Macro-F1 (%)", title="Ablation - Macro-F1 per Config")
    axes[0].legend()

    ab_df2 = ab_df[ab_df["Config"] != "Full model"].copy()
    dc = ["#e74c3c" if d < 0 else "#2ecc71" for d in ab_df2["delta"]]
    hb2 = axes[1].barh(ab_df2["Config"], ab_df2["delta"] * 100, color=dc, edgecolor="white", height=0.7)
    axes[1].bar_label(hb2, fmt="%.1f%%", padding=3, fontsize=8.5)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(xlabel="Delta F1 vs Full Model (%)", title="Ablation - Drop vs Full")

    plt.suptitle("Ablation Study - Branch Contribution", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "G_ablation.pdf"), bbox_inches="tight")
    plt.close(fig)

    return ab_df
