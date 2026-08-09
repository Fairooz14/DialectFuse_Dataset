
from __future__ import annotations

import torch
import torch.nn as nn


class CharBiLSTMEncoder(nn.Module):
    """
    Character-level BiLSTM with attention pooling.

    Attention pooling (rather than mean pooling) lets the encoder up-weight
    the diagnostic suffix positions that carry tense/person meaning in the
    Chittagong dialect (e.g. -ইল, -তু, -উম) instead of weighting every
    character position equally.
    """

    def __init__(self, vocab_sz: int, embed_dim: int, hidden_dim: int,
                 n_layers: int = 2, dropout: float = 0.45, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)   # attention scorer
        self.drop = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.embed(ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Attention pooling: score each timestep, mask padding, softmax, weighted sum
        scores = self.attn(out).squeeze(-1)                     # (B, T)
        scores = scores.masked_fill(ids == self.pad_idx, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)    # (B, T, 1)
        pooled = (out * weights).sum(1)                          # (B, hidden*2)
        return self.drop(pooled)
