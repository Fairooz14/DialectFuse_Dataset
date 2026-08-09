
from __future__ import annotations

import pandas as pd

from dialect_fusion.config import Config


def load_clean(path: str, cfg: Config) -> pd.DataFrame:
    """Load a CSV split and drop rows with missing/blank required columns."""
    df = pd.read_csv(path)
    required = cfg.all_text_cols + [cfg.col_label]
    for col in required:
        df = df.dropna(subset=[col])
        df = df[df[col].astype(str).str.strip().astype(bool)]
    return df.reset_index(drop=True)


def load_splits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean the train/val/test CSVs referenced by ``cfg``."""
    train_df = load_clean(cfg.train_csv, cfg)
    val_df = load_clean(cfg.val_csv, cfg)
    test_df = load_clean(cfg.test_csv, cfg)

    print(f"Splits: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"Classes: {train_df[cfg.col_label].nunique()}")
    return train_df, val_df, test_df
