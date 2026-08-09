
from __future__ import annotations

from torch.optim import AdamW
from torch.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup

from dialect_fusion.config import Config


def verify_freeze_state(model, cfg: Config) -> None:
    """Sanity-check the freeze/unfreeze state before optimizer construction."""
    print("Verifying parameter freeze state...")
    for name, backbone in [("BanglaBERT", model.banglabert), ("RoBERTa", model.roberta)]:
        trainable = sum(1 for p in backbone.parameters() if p.requires_grad)
        frozen = sum(1 for p in backbone.parameters() if not p.requires_grad)
        assert trainable > 0, f"FATAL: {name} has zero trainable params!"
        print(f"  {name}: trainable={trainable}  frozen={frozen}")

    for name, enc in [("enc_ctg", model.enc_ctg), ("enc_bl", model.enc_bl)]:
        frozen = sum(1 for p in enc.parameters() if not p.requires_grad)
        assert frozen == 0, f"FATAL: {name} has {frozen} frozen params!"
    print("  Char encoders: all trainable")

    assert model.enc_ctg is not model.enc_bl, "FATAL: enc_ctg and enc_bl are the same object!"
    print("  enc_ctg is not enc_bl (separate encoders confirmed)")


def build_optimizer_and_scheduler(model, cfg: Config, steps_per_epoch: int):
    verify_freeze_state(model, cfg)

    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    wd_fn = lambda n: 0.0 if any(nd in n for nd in no_decay) else cfg.weight_decay

    g_transformer, g_lstm, g_head = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ["banglabert", "roberta"]):
            g_transformer.append({"params": [param], "lr": cfg.transformer_lr, "weight_decay": wd_fn(name)})
        elif "head" in name:
            g_head.append({"params": [param], "lr": cfg.head_lr, "weight_decay": wd_fn(name)})
        else:
            g_lstm.append({"params": [param], "lr": cfg.lstm_lr, "weight_decay": wd_fn(name)})

    optimizer = AdamW(g_transformer + g_lstm + g_head)
    total_steps = (steps_per_epoch // cfg.accum_steps) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\nOptimizer groups - transformer:{len(g_transformer)}  lstm:{len(g_lstm)}  head:{len(g_head)}")
    print(f"Steps - total:{total_steps}  warmup:{warmup_steps}")
    print(f"Effective batch size: {cfg.batch_size * cfg.accum_steps}")
    return optimizer, scheduler


def build_scaler(device_str: str) -> GradScaler:
    return GradScaler(device_str)
