
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: str = "data"
    save_dir: str = "outputs/checkpoints"
    fig_dir: str = "outputs/figures"

    train_csv: str = field(init=False)
    val_csv: str = field(init=False)
    test_csv: str = field(init=False)

    # ── Column names ────────────────────────────────────────────────────
    col_ctg_bangla: str = "chittagong_bangla_speech"      # Bengali script  -> char-BiLSTM
    col_ctg_banglish: str = "chittagong_banglish_speech"  # Latin script    -> char-BiLSTM
    col_std_bangla: str = "bangla_speech"                 # Std Bangla      -> BanglaBERT
    col_english: str = "english_speech"                   # English         -> RoBERTa-base
    col_label: str = "emotions"

    # ── Pretrained backbones ───────────────────────────────────────────────
    banglabert_name: str = "csebuetnlp/banglabert"
    roberta_name: str = "roberta-base"

    # ── Architecture ────────────────────────────────────────────────────
    proj_dim: int = 256
    char_embed_dim: int = 128
    char_hidden: int = 256          # BiLSTM output = 2 * char_hidden
    freeze_bottom_layers: int = 9    # bottom N transformer layers frozen

    @property
    def fused_dim(self) -> int:
        return self.proj_dim * 4

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 15
    transformer_lr: float = 2e-5
    lstm_lr: float = 5e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.10
    warmup_ratio: float = 0.12
    batch_size: int = 16
    accum_steps: int = 2
    early_stop_patience: int = 2
    label_smoothing: float = 0.15
    focal_gamma: float = 1.5
    max_seq_len: int = 96
    augment_prob: float = 0.65

    # Overfit guard: hard-stop when val/train loss ratio blows up
    gap_hard_limit: float = 12.0
    gap_patience: int = 2

    # ── Misc ─────────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 2

    def __post_init__(self):
        self.train_csv = str(Path(self.data_dir) / "train_uniform50.csv")
        self.val_csv = str(Path(self.data_dir) / "val_uniform50.csv")
        self.test_csv = str(Path(self.data_dir) / "test_uniform50.csv")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    @property
    def all_text_cols(self) -> list[str]:
        return [
            self.col_ctg_bangla,
            self.col_ctg_banglish,
            self.col_std_bangla,
            self.col_english,
        ]
