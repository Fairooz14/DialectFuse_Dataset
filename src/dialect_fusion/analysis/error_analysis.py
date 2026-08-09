from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

PROXIMITY_GROUPS = [
    {"desire", "optimism", "excitement"},
    {"caring", "love", "admiration"},
    {"curiosity", "realization", "surprise"},
    {"anger", "annoyance", "disapproval"},
    {"sadness", "remorse", "disappointment"},
    {"joy", "amusement", "gratitude"},
]


def error_type(true_lbl: str, pred_lbl: str) -> str:
    for grp in PROXIMITY_GROUPS:
        if true_lbl in grp and pred_lbl in grp:
            return "Type II - Semantic Proximity"
    return "Type I/III - Dialect/Distribution"


def run_error_analysis(test_y, test_pred, label_map: dict, fig_dir: str) -> pd.DataFrame:
    errors = [(test_y[i], test_pred[i]) for i in range(len(test_y)) if test_y[i] != test_pred[i]]
    err_df = pd.DataFrame(errors, columns=["true", "pred"])
    err_df["true_lbl"] = err_df["true"].map(label_map)
    err_df["pred_lbl"] = err_df["pred"].map(label_map)
    err_df["etype"] = err_df.apply(lambda r: error_type(r.true_lbl, r.pred_lbl), axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    type_counts = err_df["etype"].value_counts()
    axes[0].pie(type_counts.values, labels=type_counts.index, colors=["#e74c3c", "#3498db"],
                autopct="%.1f%%", startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("Error Type Distribution")

    pair_counts = err_df.groupby(["true_lbl", "pred_lbl"]).size().reset_index(name="count")
    pair_counts = pair_counts.sort_values("count", ascending=False).head(12)
    pair_counts["pair"] = pair_counts["true_lbl"] + "\n->" + pair_counts["pred_lbl"]
    axes[1].barh(pair_counts["pair"][::-1], pair_counts["count"][::-1], color="#9b59b6", edgecolor="white", height=0.72)
    axes[1].set(xlabel="Error Count", title="Top-12 Confused Pairs")

    err_rate = {}
    num_labels = len(label_map)
    for i in range(num_labels):
        mask = test_y == i
        if mask.sum() == 0:
            continue
        err_rate[label_map[i]] = (test_pred[mask] != test_y[mask]).mean()
    er_df = pd.Series(err_rate).sort_values(ascending=False)
    c_err = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.25 else "#2ecc71" for v in er_df.values]
    axes[2].barh(er_df.index, er_df.values * 100, color=c_err, edgecolor="white", height=0.72)
    axes[2].set(xlabel="Error Rate (%)", title="Per-Class Error Rate")
    axes[2].axvline(er_df.mean() * 100, color="gray", ls="--", lw=1.2, label=f"mean={er_df.mean()*100:.1f}%")
    axes[2].legend()

    plt.suptitle("Error Analysis - 3-Type Taxonomy", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "H_error_analysis.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Total errors    : {len(errors)} / {len(test_y)}")
    print(f"Overall accuracy: {1 - len(errors)/len(test_y):.4f}")
    print("\nError type breakdown:")
    print(err_df["etype"].value_counts().to_string())
    return err_df
