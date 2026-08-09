
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

PAD_CHAR, UNK_CHAR = "<PAD>", "<UNK>"


@dataclass
class CharVocab:
    char_to_idx: dict
    max_char_len: int

    def __len__(self) -> int:
        return len(self.char_to_idx)

    @property
    def pad_idx(self) -> int:
        return self.char_to_idx[PAD_CHAR]

    @property
    def unk_idx(self) -> int:
        return self.char_to_idx[UNK_CHAR]

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        max_len = max_len or self.max_char_len
        return [self.char_to_idx.get(c, self.unk_idx) for c in str(text)[:max_len]]


def build_char_vocab(series_list: list[pd.Series], min_freq: int = 1) -> dict:
    """
    Build a char -> index vocabulary from one or more text columns.

    min_freq=1 is important for low-resource dialects: with only a few
    hundred samples, many valid Chittagonian characters appear only once,
    and min_freq=2 would silently map them all to <UNK>.
    """
    counts = Counter()
    for s in series_list:
        for text in s.astype(str):
            counts.update(list(text))
    vocab = [PAD_CHAR, UNK_CHAR] + [c for c, n in counts.most_common() if n >= min_freq]
    return {c: i for i, c in enumerate(vocab)}


def build_vocab_from_splits(train_df: pd.DataFrame, cfg, min_freq: int = 1) -> CharVocab:
    """Build the shared vocabulary from all four text columns of the train split."""
    char_to_idx = build_char_vocab(
        [
            train_df[cfg.col_ctg_bangla],
            train_df[cfg.col_ctg_banglish],
            train_df[cfg.col_std_bangla],
            train_df[cfg.col_english],
        ],
        min_freq=min_freq,
    )

    char_lens = (
        train_df[cfg.col_ctg_bangla].astype(str).str.len().tolist()
        + train_df[cfg.col_ctg_banglish].astype(str).str.len().tolist()
    )
    max_char_len = max(32, min(int(np.percentile(char_lens, 95)), 150))

    unk_idx = char_to_idx[UNK_CHAR]
    unk_rate = (
        train_df[cfg.col_ctg_bangla]
        .astype(str)
        .apply(lambda t: sum(1 for c in t if c not in char_to_idx) / max(len(t), 1))
        .mean()
    )
    print(f"Char vocab : {len(char_to_idx)}  MAX_CHAR_LEN (p95): {max_char_len}")
    print(f"UNK rate in Ctg Bangla: {unk_rate*100:.2f}%  (target: <2%)")

    return CharVocab(char_to_idx=char_to_idx, max_char_len=max_char_len)
