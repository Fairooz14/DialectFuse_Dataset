
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from dialect_fusion.config import Config


@dataclass
class NoiseDetectionResult:
    suspicious: np.ndarray
    vote_count: np.ndarray
    flag_confident_learning: np.ndarray
    flag_high_loss: np.ndarray
    flag_high_entropy: np.ndarray
    flag_knn_mismatch: np.ndarray
    per_sample_loss: np.ndarray
    entropy: np.ndarray
    oof_probs: np.ndarray
    label_encoder: LabelEncoder
    review_df: pd.DataFrame = field(repr=False)


def detect_label_noise(
    train_df: pd.DataFrame,
    cfg: Config,
    vote_thresh: int = 2,
    k_neighbors: int = 7,
    mismatch_thresh: float = 0.60,
    percentile: float = 90.0,
    seed: int = 42,
) -> NoiseDetectionResult:
    df = train_df.copy().reset_index(drop=True)
    text = (
        df[cfg.col_english].astype(str)
        + " "
        + df[cfg.col_std_bangla].astype(str)
        + " "
        + df[cfg.col_ctg_banglish].astype(str)
    )

    le = LabelEncoder()
    y = le.fit_transform(df[cfg.col_label])
    n, k = len(df), le.classes_.shape[0]
    print(f"Label noise detection  |  N={n}  classes={k}")

    # ── Surrogate model: char n-gram TF-IDF + multinomial LR ───────────────
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=60_000, sublinear_tf=True)
    X = vec.fit_transform(text)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", multi_class="multinomial", n_jobs=-1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_p = cross_val_predict(lr, X, y, cv=skf, method="predict_proba")
    print(f"OOF log-loss : {log_loss(y, oof_p):.4f}")

    # S1 - Confident learning
    cl_idx = find_label_issues(
        labels=y, pred_probs=oof_p, filter_by="prune_by_noise_rate", return_indices_ranked_by="self_confidence"
    )
    flag_cl = np.zeros(n, dtype=bool)
    flag_cl[cl_idx] = True
    print(f"S1 - Confident Learning  : {flag_cl.sum():4d} flagged")

    # S2 - High per-sample cross-entropy loss
    per_sample_loss = np.array([-np.log(oof_p[i, y[i]] + 1e-12) for i in range(n)])
    flag_loss = per_sample_loss >= np.percentile(per_sample_loss, percentile)
    print(f"S2 - High-loss (top {100 - percentile:.0f}%) : {flag_loss.sum():4d} flagged")

    # S3 - High prediction entropy
    entropy = -(oof_p * np.log(oof_p + 1e-12)).sum(axis=1)
    flag_ent = entropy >= np.percentile(entropy, percentile)
    print(f"S3 - High entropy        : {flag_ent.sum():4d} flagged")

    # S4 - kNN label mismatch (chunked cosine similarity to bound memory)
    flag_knn = np.zeros(n, dtype=bool)
    chunk = 200
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = cosine_similarity(X[start:end], X)
        for local_i, global_i in enumerate(range(start, end)):
            row_sims = sims[local_i].copy()
            row_sims[global_i] = -1
            nn_idx = np.argpartition(row_sims, -k_neighbors)[-k_neighbors:]
            mismatch = (y[nn_idx] != y[global_i]).mean()
            if mismatch > mismatch_thresh:
                flag_knn[global_i] = True
    print(f"S4 - kNN mismatch        : {flag_knn.sum():4d} flagged")

    vote_count = flag_cl.astype(int) + flag_loss.astype(int) + flag_ent.astype(int) + flag_knn.astype(int)
    suspicious = vote_count >= vote_thresh
    print(f"Suspicious (>= {vote_thresh}/4 strategies) : {suspicious.sum()} ({suspicious.mean()*100:.1f}% of train)")
    print(f"Confident clean          : {(~suspicious).sum()}")

    review_df = _build_review_dataframe(
        df, cfg, suspicious, le, oof_p, per_sample_loss, entropy, vote_count, flag_cl, flag_loss, flag_ent, flag_knn
    )

    return NoiseDetectionResult(
        suspicious=suspicious,
        vote_count=vote_count,
        flag_confident_learning=flag_cl,
        flag_high_loss=flag_loss,
        flag_high_entropy=flag_ent,
        flag_knn_mismatch=flag_knn,
        per_sample_loss=per_sample_loss,
        entropy=entropy,
        oof_probs=oof_p,
        label_encoder=le,
        review_df=review_df,
    )


def _build_review_dataframe(df, cfg, suspicious, le, oof_p, per_sample_loss, entropy, vote_count,
                             flag_cl, flag_loss, flag_ent, flag_knn) -> pd.DataFrame:
    susp_df = df[suspicious].copy()
    susp_df["given_label"] = df.loc[suspicious, cfg.col_label].values
    susp_df["surrogate_top1"] = [le.classes_[i] for i in oof_p[suspicious].argmax(axis=1)]
    susp_df["surrogate_top1_prob"] = oof_p[suspicious].max(axis=1).round(3)
    susp_df["surrogate_top2"] = [le.classes_[np.argsort(p)[-2]] for p in oof_p[suspicious]]
    susp_df["surrogate_top2_prob"] = [round(sorted(p)[-2], 3) for p in oof_p[suspicious]]
    susp_df["loss"] = per_sample_loss[suspicious].round(3)
    susp_df["entropy"] = entropy[suspicious].round(3)
    susp_df["vote_count"] = vote_count[suspicious]
    susp_df["flagged_by_CL"] = flag_cl[suspicious]
    susp_df["flagged_by_loss"] = flag_loss[suspicious]
    susp_df["flagged_by_entropy"] = flag_ent[suspicious]
    susp_df["flagged_by_knn"] = flag_knn[suspicious]
    susp_df["review_action"] = ""  # human fills: keep / relabel / remove

    cols = (
        cfg.all_text_cols
        + [
            "given_label", "surrogate_top1", "surrogate_top1_prob",
            "surrogate_top2", "surrogate_top2_prob", "vote_count",
            "loss", "entropy", "flagged_by_CL", "flagged_by_loss",
            "flagged_by_entropy", "flagged_by_knn", "review_action",
        ]
    )
    return susp_df[cols]


def build_cleaned_training_set(
    train_df: pd.DataFrame,
    result: NoiseDetectionResult,
    strategy: str = "B",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Apply a noise-cleaning strategy to the training split.

    strategy:
        "A" - Conservative: keep all rows, down-weight suspicious rows to 0.25x.
        "B" - Moderate: drop rows flagged by >= 3/4 strategies.
        "C" - Aggressive: drop all suspicious rows (>= 2/4 strategies).
    """
    if strategy == "A":
        sample_weights = np.where(result.suspicious, 0.25, 1.0)
        cleaned = train_df.copy()
        print("[Option A] Keeping all rows; suspicious rows down-weighted to 0.25x")
    elif strategy == "B":
        keep_mask = result.vote_count < 3
        cleaned = train_df[keep_mask].reset_index(drop=True)
        sample_weights = np.ones(len(cleaned))
        print(f"[Option B] Dropped {(~keep_mask).sum()} rows (>=3/4 strategies): "
              f"{len(train_df)} -> {len(cleaned)}")
    elif strategy == "C":
        keep_mask = ~result.suspicious
        cleaned = train_df[keep_mask].reset_index(drop=True)
        sample_weights = np.ones(len(cleaned))
        print(f"[Option C] Dropped {(~keep_mask).sum()} rows (>=2/4 strategies): "
              f"{len(train_df)} -> {len(cleaned)}")
    else:
        raise ValueError(f"Unknown strategy: {strategy!r} (expected 'A', 'B', or 'C')")

    return cleaned, sample_weights
