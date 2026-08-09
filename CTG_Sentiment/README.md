# Data

This folder is expected to contain three CSV:

```
train_uniform50.csv
val_uniform50.csv
test_uniform50.csv
```

Each row must have the following columns:

| Column                       | Description                                   |
|-------------------------------|------------------------------------------------|
| `chittagong_bangla_speech`   | Utterance in Chittagonian dialect, Bengali script |
| `chittagong_banglish_speech` | Same utterance, romanized ("Banglish")        |
| `bangla_speech`              | Standard Bangla translation/gloss             |
| `english_speech`             | English translation/gloss                     |
| `emotions`                   | Emotion label (23-class GoEmotions-style taxonomy in the original study) |

`scripts/detect_label_noise.py` will additionally produce `suspicious_labels.csv` and
`train_clean.csv` in this directory.
