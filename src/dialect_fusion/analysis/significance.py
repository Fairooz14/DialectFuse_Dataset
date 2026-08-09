from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from dialect_fusion.analysis.ablation import ABLATION_CONFIGS
from dialect_fusion.training.evaluate import evaluate


def run_mcnemar(preds_full, preds_other, true_labels, name: str):
    ca = preds_full == true_labels
    cb = preds_other == true_labels
    b = int((ca & ~cb).sum())   # full correct, other wrong
    c = int((~ca & cb).sum())   # full wrong, other correct
    if b + c == 0:
        print(f"  {name:<30s}  b={b} c={c}  p=N/A  (identical)")
        return None
    tbl = [[0, b], [c, 0]]
    result = mcnemar(tbl, exact=(b + c < 25))
    sig = "sig (p<0.05)" if result.pvalue < 0.05 else "n.s."
    print(f"  {name:<30s}  b={b:3d}  c={c:3d}  p={result.pvalue:.4f}  {sig}")
    return result.pvalue


def run_significance_tests(model, test_loader, device, loss_fn, device_str, fig_dir: str) -> pd.DataFrame:
    print("McNemar vs Full Model:\n")
    _, _, _, fy, fp, _ = evaluate(model, test_loader, device, loss_fn, device_str, ablate=set())

    rows = []
    for name, ablate_set in ABLATION_CONFIGS.items():
        if name == "Full model":
            continue
        _, _, _, ay, ap, _ = evaluate(model, test_loader, device, loss_fn, device_str, ablate=ablate_set)
        p = run_mcnemar(fp, ap, fy, name)
        rows.append({"Config": name, "p-value": p, "significant": p is not None and p < 0.05})

    mc_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, max(4, len(mc_df) * 0.45)))
    mc_sorted = mc_df.sort_values("p-value", ascending=True)
    bar_c = ["#e74c3c" if r["significant"] else "#95a5a6" for _, r in mc_sorted.iterrows()]
    ax.barh(mc_sorted["Config"], -np.log10(mc_sorted["p-value"].fillna(1.0).clip(1e-10)),
            color=bar_c, edgecolor="white", height=0.7)
    ax.axvline(-np.log10(0.05), color="crimson", ls="--", lw=1.3, label="p=0.05 threshold")
    ax.set(xlabel="-log10(p-value)", title="McNemar Test - Full vs Ablated")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "I_mcnemar.pdf"), bbox_inches="tight")
    plt.close(fig)

    return mc_df
