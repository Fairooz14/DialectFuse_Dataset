from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from dialect_fusion.config import Config
from dialect_fusion.data.augment import LightAugmenter
from dialect_fusion.data.vocab import CharVocab


def load_tokenizers(cfg: Config) -> tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase]:
    bangla_tok = AutoTokenizer.from_pretrained(cfg.banglabert_name)
    roberta_tok = AutoTokenizer.from_pretrained(cfg.roberta_name)
    print("BanglaBERT:", cfg.banglabert_name)
    print("RoBERTa   :", cfg.roberta_name)
    return bangla_tok, roberta_tok


class FusionDataset(Dataset):
    """Yields per-branch encodings for a single row: 2x char-id sequences + 2x tokenizer outputs."""

    def __init__(
        self,
        df: pd.DataFrame,
        y,
        cfg: Config,
        btok: PreTrainedTokenizerBase,
        rtok: PreTrainedTokenizerBase,
        char_vocab: CharVocab,
        max_seq: int,
        augment: bool = False,
        augmenter: LightAugmenter | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.y = y
        self.cfg = cfg
        self.btok = btok
        self.rtok = rtok
        self.char_vocab = char_vocab
        self.max_seq = max_seq
        self.augment = augment
        self.augmenter = augmenter or LightAugmenter()

    def __len__(self) -> int:
        return len(self.y)

    def _aug(self, text: str) -> str:
        return self.augmenter(text) if self.augment else text

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        ctg_ids = self.char_vocab.encode(self._aug(row[self.cfg.col_ctg_bangla]))
        ctg_bl_ids = self.char_vocab.encode(self._aug(row[self.cfg.col_ctg_banglish]))
        bb = self.btok(self._aug(row[self.cfg.col_std_bangla]), truncation=True,
                        max_length=self.max_seq, return_tensors=None)
        rb = self.rtok(self._aug(row[self.cfg.col_english]), truncation=True,
                        max_length=self.max_seq, return_tensors=None)
        return dict(
            ctg_ids=ctg_ids, ctg_bl_ids=ctg_bl_ids,
            bb_input_ids=bb["input_ids"], bb_mask=bb["attention_mask"],
            rb_input_ids=rb["input_ids"], rb_mask=rb["attention_mask"],
            label=int(self.y[idx]),
        )


def make_fusion_collate(pad_idx: int):
    """Return a collate_fn closed over the char-vocab pad index."""

    def _pad_char(seqs):
        m = max(len(s) for s in seqs)
        return torch.tensor([s + [pad_idx] * (m - len(s)) for s in seqs], dtype=torch.long)

    def _pad_seq(seqs):
        m = max(len(s) for s in seqs)
        return torch.tensor([s + [0] * (m - len(s)) for s in seqs], dtype=torch.long)

    def fusion_collate(batch):
        return dict(
            ctg_ids=_pad_char([b["ctg_ids"] for b in batch]),
            ctg_bl_ids=_pad_char([b["ctg_bl_ids"] for b in batch]),
            ctg_len=torch.tensor([len(b["ctg_ids"]) for b in batch]),
            ctg_bl_len=torch.tensor([len(b["ctg_bl_ids"]) for b in batch]),
            bb_input_ids=_pad_seq([b["bb_input_ids"] for b in batch]),
            bb_mask=_pad_seq([b["bb_mask"] for b in batch]),
            rb_input_ids=_pad_seq([b["rb_input_ids"] for b in batch]),
            rb_mask=_pad_seq([b["rb_mask"] for b in batch]),
            labels=torch.tensor([b["label"] for b in batch], dtype=torch.long),
        )

    return fusion_collate


def make_loader(
    df: pd.DataFrame, y, cfg: Config, btok, rtok, char_vocab: CharVocab,
    augment: bool, batch_size: int, shuffle: bool, num_workers: int = 2,
) -> DataLoader:
    ds = FusionDataset(df, y, cfg, btok, rtok, char_vocab, cfg.max_seq_len, augment=augment)
    collate_fn = make_fusion_collate(char_vocab.pad_idx)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )
