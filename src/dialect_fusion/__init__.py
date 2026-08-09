"""
Dialect Fusion CTG
===================
Multi-input late-fusion emotion classification for Chittagonian Bangla dialect.

Fuses four parallel text representations of the same utterance:
  - Chittagong Bangla   (Bengali script)  -> character BiLSTM
  - Chittagong Banglish (Latin script)    -> character BiLSTM
  - Standard Bangla                       -> BanglaBERT
  - English (gloss)                       -> RoBERTa-base
"""

__version__ = "1.0.0"
