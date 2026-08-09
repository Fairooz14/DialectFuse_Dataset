from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dialect_fusion.analysis.ablation import run_ablation_study
from dialect_fusion.analysis.calibration import run_calibration_analysis
from dialect_fusion.analysis.case_study import default_case, run_case_study
from dialect_fusion.analysis.error_analysis import run_error_analysis
from dialect_fusion.analysis.expandability import FIFTH_BRANCH_NOTE, expand_classes_demo
from dialect_fusion.analysis.explainability import run_lime_analysis, run_shap_analysis
from dialect_fusion.analysis.results import build_results_table
from dialect_fusion.analysis.significance import run_significance_tests
from dialect_fusion.analysis.visualization import (
    plot_confusion_matrices, plot_dataset_stats, plot_per_class_metrics, plot_training_history,
)
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
    p.add_argument("--skip-explainability", action="store_true", help="Skip the slow SHAP/LIME step")
    args = p.parse_args()

    cfg = Config(data_dir=args.data_dir, save_dir=args.save_dir, fig_dir=args.fig_dir, seed=args.seed)
    apply_style()
    set_seed(cfg.seed)
    device = get_device()
    device_str = device.type

    art = load_trained_artifacts(cfg, device)
    model, loss_fn = art["model"], art["loss_fn"]

    print("\n== Dataset statistics ==")
    plot_dataset_stats(art["train_df"], cfg, cfg.fig_dir)

    history_path = Path(cfg.save_dir) / "history.json"
    if history_path.exists():
        print("\n== Training history ==")
        with open(history_path) as f:
            history = json.load(f)
        best_epoch = int(max(range(len(history["val_f1"])), key=lambda i: history["val_f1"][i])) + 1
        plot_training_history(history, best_epoch, cfg.fig_dir)
    else:
        print(f"\n(No {history_path} found - skipping training-history plot. "
              f"Run scripts/train.py first to generate it.)")

    print("\n== Test evaluation ==")
    test_loss, test_acc, test_f1, test_y, test_pred, test_probs = evaluate(
        model, art["test_loader"], device, loss_fn, device_str
    )
    print(f"Test accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}  loss={test_loss:.4f}")
    plot_confusion_matrices(test_y, test_pred, art["target_names"], cfg.fig_dir)
    plot_per_class_metrics(test_y, test_pred, art["label_map"], art["num_labels"], cfg.fig_dir)

    print("\n== Linguistic case study ==")
    case = default_case(cfg)
    case_probs, sorted_idx = run_case_study(
        model, case, cfg, art["bangla_tok"], art["roberta_tok"], art["char_vocab"],
        art["label_map"], art["rev_map"], device, device_str, cfg.fig_dir,
    )

    print("\n== Calibration analysis ==")
    calib = run_calibration_analysis(
        model, art["val_loader"], art["test_loader"], test_probs, test_y,
        art["label_map"], art["num_labels"], device, device_str, cfg.fig_dir,
    )

    if not args.skip_explainability:
        print("\n== SHAP explainability ==")
        run_shap_analysis(
            model, case, case_probs, cfg, art["bangla_tok"], art["roberta_tok"], art["char_vocab"],
            art["target_names"], art["label_map"], device, device_str, cfg.fig_dir,
        )
        print("\n== LIME explainability ==")
        run_lime_analysis(
            model, case, case_probs, cfg, art["bangla_tok"], art["roberta_tok"], art["char_vocab"],
            art["target_names"], device, device_str, cfg.fig_dir, seed=cfg.seed,
        )

    print("\n== Branch ablation study ==")
    ab_df = run_ablation_study(model, art["test_loader"], device, loss_fn, device_str, cfg.fig_dir)

    print("\n== Error taxonomy ==")
    run_error_analysis(test_y, test_pred, art["label_map"], cfg.fig_dir)

    print("\n== McNemar significance tests ==")
    run_significance_tests(model, art["test_loader"], device, loss_fn, device_str, cfg.fig_dir)

    print("\n== Class-expandability demo ==")
    expand_classes_demo(model, device, n_new=1)
    print(FIFTH_BRANCH_NOTE)

    print("\n== Final results table ==")
    build_results_table(
        test_y, test_pred, art["target_names"], test_acc, test_f1,
        calib["ece_after"], calib["T_opt"], art["num_labels"], model, cfg,
    )


if __name__ == "__main__":
    main()
