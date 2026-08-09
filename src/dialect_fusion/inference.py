
from __future__ import annotations

import json
import os

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from dialect_fusion.config import Config
from dialect_fusion.data.dataset import load_tokenizers, make_loader
from dialect_fusion.data.loading import load_splits
from dialect_fusion.models.fusion_classifier import build_model
from dialect_fusion.models.losses import FocalLoss


def load_trained_artifacts(cfg: Config, device):
    """
    Returns a dict with everything needed to run inference / analysis on a
    checkpoint previously produced by ``scripts/train.py``:
    model, loss_fn, char_vocab, label_map, tokenizers, data splits and loaders.
    """
    train_df, val_df, test_df = load_splits(cfg)

    char_vocab = torch.load(os.path.join(cfg.save_dir, "char_vocab.pt"), weights_only=False)
    with open(os.path.join(cfg.save_dir, "label_map.json"), encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    num_labels = len(label_map)
    rev_map = {v: k for k, v in label_map.items()}

    le = LabelEncoder()
    le.classes_ = np.array([label_map[i] for i in range(num_labels)])
    y_train = le.transform(train_df[cfg.col_label])
    y_val = le.transform(val_df[cfg.col_label])
    y_test = le.transform(test_df[cfg.col_label])

    cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)

    bangla_tok, roberta_tok = load_tokenizers(cfg)

    train_loader = make_loader(train_df, y_train, cfg, bangla_tok, roberta_tok, char_vocab,
                                augment=False, batch_size=cfg.batch_size * 2, shuffle=False)
    val_loader = make_loader(val_df, y_val, cfg, bangla_tok, roberta_tok, char_vocab,
                              augment=False, batch_size=cfg.batch_size * 2, shuffle=False)
    test_loader = make_loader(test_df, y_test, cfg, bangla_tok, roberta_tok, char_vocab,
                               augment=False, batch_size=cfg.batch_size * 2, shuffle=False)

    model = build_model(len(char_vocab), num_labels, cfg, device)
    state_dict = torch.load(os.path.join(cfg.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state_dict)

    loss_fn = FocalLoss(num_labels, cfg.focal_gamma, cfg.label_smoothing, class_weights)

    return dict(
        model=model, loss_fn=loss_fn, char_vocab=char_vocab, label_map=label_map, rev_map=rev_map,
        num_labels=num_labels, bangla_tok=bangla_tok, roberta_tok=roberta_tok,
        train_df=train_df, val_df=val_df, test_df=test_df,
        y_train=y_train, y_val=y_val, y_test=y_test,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        target_names=[label_map[i] for i in range(num_labels)],
    )
