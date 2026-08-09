from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dialect_fusion.config import Config
from dialect_fusion.data.label_noise import build_cleaned_training_set, detect_label_noise
from dialect_fusion.data.loading import load_clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--strategy", choices=["A", "B", "C"], default="B",
                         help="A=down-weight suspicious rows, B=drop >=3/4 votes, C=drop >=2/4 votes")
    parser.add_argument("--vote-thresh", type=int, default=2)
    args = parser.parse_args()

    cfg = Config(data_dir=args.data_dir)
    train_df = load_clean(cfg.train_csv, cfg)

    result = detect_label_noise(train_df, cfg, vote_thresh=args.vote_thresh)

    review_path = Path(cfg.data_dir) / "suspicious_labels.csv"
    result.review_df.to_csv(review_path, index=True)
    print(f"Exported {len(result.review_df)} suspicious rows -> {review_path}")

    cleaned_df, sample_weights = build_cleaned_training_set(train_df, result, strategy=args.strategy)
    clean_path = Path(cfg.data_dir) / "train_clean.csv"
    cleaned_df.to_csv(clean_path, index=False)
    print(f"Clean training set saved -> {clean_path}")


if __name__ == "__main__":
    main()
