from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dialect_fusion.analysis.visualization import plot_confusion_matrices, plot_per_class_metrics
from dialect_fusion.config import Config
from dialect_fusion.inference import load_trained_artifacts
from dialect_fusion.training.evaluate import evaluate
from dialect_fusion.utils.plotting import apply_style
from dialect_fusion.utils.seed import get_device, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--save-dir", default="outputs/checkpoints")
    p.add_argument("--fig-dir", default="outputs/figures")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = Config(data_dir=args.data_dir, save_dir=args.save_dir, fig_dir=args.fig_dir, seed=args.seed)
    apply_style()
    set_seed(cfg.seed)
    device = get_device()
    device_str = device.type

    art = load_trained_artifacts(cfg, device)

    test_loss, test_acc, test_f1, test_y, test_pred, test_probs = evaluate(
        art["model"], art["test_loader"], device, art["loss_fn"], device_str
    )

    print("=" * 52)
    print(f"  TEST  Accuracy : {test_acc:.4f}  ({test_acc*100:.2f} %)")
    print(f"  TEST  Macro-F1 : {test_f1:.4f}")
    print(f"  TEST  Loss     : {test_loss:.4f}")
    print("=" * 52)
    print(classification_report(test_y, test_pred, target_names=art["target_names"], digits=4))

    plot_confusion_matrices(test_y, test_pred, art["target_names"], cfg.fig_dir)
    plot_per_class_metrics(test_y, test_pred, art["label_map"], art["num_labels"], cfg.fig_dir)


if __name__ == "__main__":
    main()
