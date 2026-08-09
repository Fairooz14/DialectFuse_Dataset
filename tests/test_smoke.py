
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dialect_fusion.config import Config
from dialect_fusion.data.augment import LightAugmenter
from dialect_fusion.data.vocab import build_vocab_from_splits
from dialect_fusion.models.char_encoder import CharBiLSTMEncoder
from dialect_fusion.models.losses import FocalLoss


def test_config_defaults():
    cfg = Config(data_dir="/tmp/dfc_test_data", save_dir="/tmp/dfc_test_ckpt", fig_dir="/tmp/dfc_test_fig")
    assert cfg.fused_dim == cfg.proj_dim * 4
    assert cfg.train_csv.endswith("train_uniform50.csv")


def test_char_vocab_and_augmenter():
    cfg = Config(data_dir="/tmp/dfc_test_data2", save_dir="/tmp/dfc_test_ckpt2", fig_dir="/tmp/dfc_test_fig2")
    df = pd.DataFrame({
        cfg.col_ctg_bangla: ["ইচ্ছে আসিল", "মিক্কে ঘুইরতু"],
        cfg.col_ctg_banglish: ["icche asilo", "mikke ghuirtu"],
        cfg.col_std_bangla: ["ইচ্ছা ছিল", "কোথাও ঘুরতে"],
        cfg.col_english: ["I wanted", "to roam somewhere"],
    })
    vocab = build_vocab_from_splits(df, cfg, min_freq=1)
    assert len(vocab) > 2  # at least PAD + UNK + some chars
    ids = vocab.encode("icche")
    assert all(isinstance(i, int) for i in ids)

    aug = LightAugmenter(prob=1.0)
    out = aug("this is a test sentence")
    assert isinstance(out, str)


def test_char_bilstm_encoder_forward():
    vocab_sz, embed_dim, hidden_dim = 20, 8, 16
    enc = CharBiLSTMEncoder(vocab_sz, embed_dim, hidden_dim, n_layers=1, dropout=0.0)
    ids = torch.randint(1, vocab_sz, (4, 10))
    lengths = torch.tensor([10, 8, 5, 3])
    out = enc(ids, lengths)
    assert out.shape == (4, hidden_dim * 2)


def test_focal_loss_forward():
    loss_fn = FocalLoss(num_classes=5, gamma=1.5, smoothing=0.1)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0
