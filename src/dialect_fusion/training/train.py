
from __future__ import annotations

import json
import os

import torch
from torch.amp import autocast

from dialect_fusion.config import Config
from dialect_fusion.training.evaluate import evaluate, move


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, scaler, loss_fn,
    device: torch.device, device_str: str, cfg: Config, char_vocab, label_map: dict,
) -> dict:
    """
    Train ``model`` for up to ``cfg.epochs`` epochs.

    Two independent stopping conditions:
      - F1 early stopping: stop after ``cfg.early_stop_patience`` epochs
        without a new best validation macro-F1.
      - Overfit gap guard: stop if val_loss/train_loss exceeds
        ``cfg.gap_hard_limit`` for ``cfg.gap_patience`` consecutive epochs.

    Saves the best checkpoint (by val macro-F1) to ``cfg.save_dir``.
    """
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_epoch, no_improve = 0.0, 0, 0
    gap_overfit_count = 0

    print(f"Training for up to {cfg.epochs} epochs (early stop patience={cfg.early_stop_patience})")
    print(f"Frozen transformer layers: 0-{cfg.freeze_bottom_layers - 1} "
          f"(fine-tuning layers {cfg.freeze_bottom_layers}+)")
    print("-" * 70)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = move(batch, device)
            labels = batch["labels"]
            with autocast(device_type=device_str):
                logits = model(
                    ctg_ids=batch["ctg_ids"], ctg_len=batch["ctg_len"],
                    ctg_bl_ids=batch["ctg_bl_ids"], ctg_bl_len=batch["ctg_bl_len"],
                    bb_input_ids=batch["bb_input_ids"], bb_mask=batch["bb_mask"],
                    rb_input_ids=batch["rb_input_ids"], rb_mask=batch["rb_mask"],
                )
                loss = loss_fn(logits, labels) / cfg.accum_steps

            scaler.scale(loss).backward()
            train_sum += loss.item() * cfg.accum_steps * labels.size(0)

            if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        avg_tr = train_sum / len(train_loader.dataset)
        vl, va, vf, *_ = evaluate(model, val_loader, device, loss_fn, device_str)

        history["train_loss"].append(avg_tr)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["val_f1"].append(vf)

        gap_ratio = vl / max(avg_tr, 1e-9)
        gap_warn = "  [OVERFIT WARN]" if gap_ratio > 10 else ""
        flag = "  <- best" if vf > best_f1 else ""
        print(f"Ep {epoch:3d}/{cfg.epochs}  tr={avg_tr:.4f}  vl={vl:.4f}"
              f"  gap={gap_ratio:.1f}x  acc={va:.4f}  F1={vf:.4f}{flag}{gap_warn}")

        if vf > best_f1:
            best_f1, best_epoch, no_improve = vf, epoch, 0
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_model.pt"))
            torch.save(char_vocab, os.path.join(cfg.save_dir, "char_vocab.pt"))
            with open(os.path.join(cfg.save_dir, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"Early stop (F1 patience) - best epoch {best_epoch}  F1={best_f1:.4f}")
                break

        if gap_ratio > cfg.gap_hard_limit:
            gap_overfit_count += 1
            if gap_overfit_count >= cfg.gap_patience:
                print(f"Early stop (gap={gap_ratio:.1f}x > {cfg.gap_hard_limit}x for "
                      f"{cfg.gap_patience} epochs) - best epoch {best_epoch}  F1={best_f1:.4f}")
                break
        else:
            gap_overfit_count = 0

    print(f"\nDone.  Best Val F1={best_f1:.4f} @ epoch {best_epoch}")
    return history
