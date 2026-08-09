from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dialect_fusion.config import Config
from dialect_fusion.data.dataset import load_tokenizers, make_loader
from dialect_fusion.data.loading import load_splits
from dialect_fusion.data.vocab import build_vocab_from_splits
from dialect_fusion.models.fusion_classifier import build_model
from dialect_fusion.models.losses import FocalLoss
from dialect_fusion.training.optim import build_optimizer_and_scheduler, build_scaler
from dialect_fusion.training.train import train_model
from dialect_fusion.utils.plotting import apply_style
from dialect_fusion.utils.seed import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--save-dir", default="outputs/checkpoints")
    p.add_argument("--fig-dir", default="outputs/figures")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-csv", default=None, help="Override train CSV (e.g. a cleaned split)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(data_dir=args.data_dir, save_dir=args.save_dir, fig_dir=args.fig_dir, seed=args.seed)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.train_csv:
        cfg.train_csv = args.train_csv

    apply_style()
    set_seed(cfg.seed)
    device = get_device()
    device_str = device.type

    train_df, val_df, test_df = load_splits(cfg)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[cfg.col_label])
    y_val = le.transform(val_df[cfg.col_label])
    y_test = le.transform(test_df[cfg.col_label])
    num_labels = len(le.classes_)
    label_map = {i: c for i, c in enumerate(le.classes_)}
    print("Num labels:", num_labels)
    print("Classes:", list(le.classes_))

    cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    print(f"Weight range: {cw.min():.3f} - {cw.max():.3f}")

    char_vocab = build_vocab_from_splits(train_df, cfg)
    bangla_tok, roberta_tok = load_tokenizers(cfg)

    train_loader = make_loader(train_df, y_train, cfg, bangla_tok, roberta_tok, char_vocab,
                                augment=True, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_workers)
    val_loader = make_loader(val_df, y_val, cfg, bangla_tok, roberta_tok, char_vocab,
                              augment=False, batch_size=cfg.batch_size * 2, shuffle=False,
                              num_workers=cfg.num_workers)
    test_loader = make_loader(test_df, y_test, cfg, bangla_tok, roberta_tok, char_vocab,
                               augment=False, batch_size=cfg.batch_size * 2, shuffle=False,
                               num_workers=cfg.num_workers)
    print(f"Loaders - train:{len(train_loader)}  val:{len(val_loader)}  test:{len(test_loader)}")

    model = build_model(len(char_vocab), num_labels, cfg, device)
    loss_fn = FocalLoss(num_labels, cfg.focal_gamma, cfg.label_smoothing, class_weights)

    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, steps_per_epoch=len(train_loader))
    scaler = build_scaler(device_str)

    history = train_model(
        model, train_loader, val_loader, optimizer, scheduler, scaler, loss_fn,
        device, device_str, cfg, char_vocab, label_map,
    )

    import json
    with open(Path(cfg.save_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved -> {Path(cfg.save_dir) / 'history.json'}")


if __name__ == "__main__":
    main()
