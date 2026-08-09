# Dialect Fusion CTG

Multi-input late-fusion emotion classification for **Chittagonian Bangla**
(a low-resource Bangla dialect spoken in southeastern Bangladesh).


## Architecture

Each training example carries the *same* utterance in four parallel forms.
The model encodes each with a branch suited to its script/resource level,
projects all four to a common dimension, and fuses by concatenation:

```
chittagong_bangla_speech   -> char-BiLSTM (attn pool) -> proj 256 ─┐
chittagong_banglish_speech -> char-BiLSTM (attn pool) -> proj 256 ─┤
bangla_speech               -> BanglaBERT [CLS]         -> proj 256 ─┼─ concat(1024) -> MLP head -> N emotions
english_speech               -> RoBERTa-base [CLS]       -> proj 256 ─┘
```

Key design choices:
- **Separate** (not shared) character encoders for the Bengali-script and
  Latin-script ("Banglish") Chittagonian branches.
- **Attention pooling** over the char-BiLSTM outputs, so the model can
  up-weight diagnostic dialectal suffixes rather than average them away.
- The bottom transformer layers of both BanglaBERT and RoBERTa are
  **frozen** to reduce overfitting on a small, low-resource dataset.
- Loss = **focal loss + label smoothing + class weighting**, tuned for a
  long-tailed emotion label distribution.

## Repository layout

```
dialect-fusion-ctg/
├── src/dialect_fusion/
│   ├── config.py                # all paths + hyperparameters
│   ├── data/
│   │   ├── loading.py           # CSV loading/cleaning
│   │   ├── label_noise.py       # 4-strategy label-noise detection (cleanlab + custom)
│   │   ├── vocab.py             # character vocabulary
│   │   ├── augment.py           # light word-level augmentation
│   │   └── dataset.py           # FusionDataset, collate, DataLoader factory
│   ├── models/
│   │   ├── char_encoder.py      # CharBiLSTMEncoder (attention pooling)
│   │   ├── fusion_classifier.py # 4-branch FusionClassifier
│   │   └── losses.py            # FocalLoss
│   ├── training/
│   │   ├── optim.py             # 3-LR-group optimizer + cosine schedule
│   │   ├── evaluate.py          # shared eval loop
│   │   └── train.py             # training loop w/ early stop + overfit guard
│   ├── analysis/
│   │   ├── visualization.py     # dataset stats, training curves, confusion matrix
│   │   ├── calibration.py       # ECE + temperature scaling
│   │   ├── case_study.py        # single-sentence qualitative prediction
│   │   ├── explainability.py    # SHAP (per-branch) + LIME
│   │   ├── ablation.py          # branch-ablation study
│   │   ├── error_analysis.py    # 3-type error taxonomy
│   │   ├── significance.py      # McNemar significance tests
│   │   ├── expandability.py     # class-growth / new-branch demos
│   │   └── results.py           # final results table
│   ├── inference.py             # reload a trained checkpoint + data artifacts
│   └── utils/                   # seeding, device, plotting style
├── scripts/
│   ├── detect_label_noise.py    # CLI: flag/clean noisy training labels
│   ├── train.py                 # CLI: train the model
│   ├── evaluate.py              # CLI: test-set evaluation + confusion matrix
│   └── run_analysis.py          # CLI: full extended analysis suite
├── notebooks/                   # original research notebook (reference only)
├── data/                        # place train/val/test CSVs here
├── outputs/
│   ├── checkpoints/             # best_model.pt, char_vocab.pt, label_map.json
│   └── figures/                 # all generated PDFs/PNGs
└── tests/                       # offline smoke tests
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

Place `train_uniform50.csv`, `val_uniform50.csv`, `test_uniform50.csv` in
`data/` (see `CTG_Sentiment/README.md` for the required columns). 

Every script also works as an importable module; see `scripts/*.py` for
the small amount of glue code, all real logic lives in `src/dialect_fusion/`.

## Label noise detection

`scripts/detect_label_noise.py` runs four independent, complementary
strategies over a fast TF-IDF + logistic-regression surrogate model and
flags a row as suspicious if at least `--vote-thresh` of them agree:

| Strategy | What it finds |
|---|---|
| Confident Learning (cleanlab) | Predicted-probability inconsistent with the given label |
| High cross-entropy loss | Rows in the top-10% hardest for the surrogate |
| High prediction entropy | Rows where the surrogate is maximally uncertain |
| kNN label mismatch | Rows whose nearest TF-IDF neighbours mostly disagree on label |

## Extended analysis

`scripts/run_analysis.py` reproduces every analysis from the original
research notebook against a trained checkpoint:

- **Calibration**: Expected Calibration Error before/after temperature scaling,
  reliability diagrams, per-class Brier scores.
- **Explainability**: SHAP token attributions per input branch (script-aware
  maskers for Bengali vs. Latin script), plus LIME on the English branch.
- **Ablation**: zero out each branch (or keep only one) and re-evaluate,
  to quantify each branch's contribution to macro-F1.
- **Error taxonomy**: classifies test-set errors into semantic-proximity
  confusions vs. dialect/distribution-shift confusions.
- **Significance**: McNemar's test comparing the full model against every
  ablated variant.
- **Expandability**: demonstrates growing the output head to a new emotion
  class without retraining the backbones, and documents how to add a fifth
  language branch.




## License

MIT — see [LICENSE](LICENSE).
