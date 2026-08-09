
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from dialect_fusion.models.char_encoder import CharBiLSTMEncoder


class FusionClassifier(nn.Module):


    def __init__(
        self,
        char_vocab_sz: int,
        num_labels: int,
        banglabert_name: str,
        roberta_name: str,
        proj_dim: int = 256,
        fused_dim: int = 1024,
        char_embed: int = 128,
        char_hidden: int = 256,
        freeze_bottom_layers: int = 9,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        self.enc_ctg = CharBiLSTMEncoder(char_vocab_sz, char_embed, char_hidden)
        self.enc_bl = CharBiLSTMEncoder(char_vocab_sz, char_embed, char_hidden)
        char_out = char_hidden * 2  # bidirectional

        self.banglabert = AutoModel.from_pretrained(banglabert_name)
        self.roberta = AutoModel.from_pretrained(roberta_name)

        for backbone in [self.banglabert, self.roberta]:
            for p in backbone.parameters():
                p.requires_grad_(True)
            for i, layer in enumerate(backbone.encoder.layer):
                if i < freeze_bottom_layers:
                    for p in layer.parameters():
                        p.requires_grad_(False)

        self.banglabert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.roberta.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        bb_hid = self.banglabert.config.hidden_size
        rb_hid = self.roberta.config.hidden_size

        def _proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, proj_dim), nn.GELU())

        self.proj_ctg = _proj(char_out)
        self.proj_bl = _proj(char_out)
        self.proj_bb = _proj(bb_hid)
        self.proj_rb = _proj(rb_hid)

        mid = fused_dim // 2
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.45),
            nn.Linear(fused_dim, mid),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(mid, num_labels),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  Trainable params : {n_trainable/1e6:.1f}M")
        print(f"  Frozen params    : {n_frozen/1e6:.1f}M  (bottom {freeze_bottom_layers} layers)")

    def _encode_char(self, enc, proj, ids, lengths):
        return proj(enc(ids, lengths))

    def _encode_transformer(self, mdl, proj, input_ids, mask):
        out = mdl(input_ids=input_ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0, :]
        return proj(cls)

    def encode_branches(
        self, ctg_ids, ctg_len, ctg_bl_ids, ctg_bl_len,
        bb_input_ids, bb_mask, rb_input_ids, rb_mask, ablate: set | None = None,
    ):
        """Encode all four branches, zeroing out any in ``ablate`` (for ablation studies)."""
        ablate = ablate or set()
        B = ctg_ids.size(0)
        zero = lambda: torch.zeros(B, self.proj_dim, device=ctg_ids.device)

        v_ctg = zero() if "ctg" in ablate else self._encode_char(self.enc_ctg, self.proj_ctg, ctg_ids, ctg_len)
        v_bl = zero() if "banglish" in ablate else self._encode_char(self.enc_bl, self.proj_bl, ctg_bl_ids, ctg_bl_len)
        v_bb = zero() if "bangla" in ablate else self._encode_transformer(self.banglabert, self.proj_bb, bb_input_ids, bb_mask)
        v_rb = zero() if "english" in ablate else self._encode_transformer(self.roberta, self.proj_rb, rb_input_ids, rb_mask)
        return v_ctg, v_bl, v_bb, v_rb

    def forward(
        self, ctg_ids, ctg_len, ctg_bl_ids, ctg_bl_len,
        bb_input_ids, bb_mask, rb_input_ids, rb_mask, ablate: set | None = None,
    ):
        v_ctg, v_bl, v_bb, v_rb = self.encode_branches(
            ctg_ids, ctg_len, ctg_bl_ids, ctg_bl_len, bb_input_ids, bb_mask, rb_input_ids, rb_mask, ablate=ablate
        )
        fused = torch.cat([v_ctg, v_bl, v_bb, v_rb], dim=-1)
        return self.head(fused)


def build_model(char_vocab_sz: int, num_labels: int, cfg, device) -> FusionClassifier:
    model = FusionClassifier(
        char_vocab_sz=char_vocab_sz,
        num_labels=num_labels,
        banglabert_name=cfg.banglabert_name,
        roberta_name=cfg.roberta_name,
        proj_dim=cfg.proj_dim,
        fused_dim=cfg.fused_dim,
        char_embed=cfg.char_embed_dim,
        char_hidden=cfg.char_hidden,
        freeze_bottom_layers=cfg.freeze_bottom_layers,
    ).to(device)
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable: {tp/1e6:.1f}M params")
    return model
