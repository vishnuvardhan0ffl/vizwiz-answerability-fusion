# Data and output formats (Project 2.2)

Everyone reads and writes these formats, so work can happen in parallel. Change them only by pull request.

## Feature parts

Features live in `features_dir` (see `configs/default.yaml`), one folder per part:

```
features/
    trainval/    official train + validation, made from the pilot cache by scripts/convert_pilot_cache.py
    test/        official test split (8,000 pairs), extracted by Sanjai
```

Each part holds `meta.csv` plus five arrays. Row i of every array belongs to row i of `meta.csv`.

`meta.csv` required columns (others are allowed):

| Column | Meaning |
|---|---|
| pair_id | image file name, unique across all parts (e.g. VizWiz_test_00000001.jpg) |
| question | question text |
| official_split | train, val or test (mapped to train, dev, test) |
| y | 1 = unanswerable, 0 = answerable (1 - answerable flag) |
| unans_votes | how many of the 10 crowd answers are "unanswerable" or "unsuitable" (used for soft labels) |

Arrays:

| File | Shape | Type | Content |
|---|---|---|---|
| img_cls.npy | (N, 768) | float16 | ViT-B/16 [CLS] vector |
| img_tok.npy | (N, 49, 768) | float16 | ViT patch tokens, 14x14 average-pooled to 7x7 |
| txt_mean.npy | (N, 768) | float16 | BERT mask-aware mean of token states |
| txt_tok.npy | (N, 40, 768) | float16 | BERT token states, padded or truncated to 40 |
| txt_mask.npy | (N, 40) | uint8 | BERT attention mask |

Use exactly the encoders and preprocessing of `notebooks/01_vizwiz_pilot.ipynb`: `google/vit-base-patch16-224-in21k`, `bert-base-uncased`, EXIF rotation, 224x224 resize without cropping.

## Run outputs

`python -m src.train` writes one folder per model and seed: `runs/<model>/seed<k>/`

| File | Content |
|---|---|
| model.pt | best weights, selected on dev AUPRC |
| history.csv | epoch, train_loss, dev_loss, dev_auprc, seconds |
| predictions.csv | pair_id, split, model, seed, y, p_unanswerable (+ gate_q for gated fusion) for dev and test |
| summary.json | settings, parameter count, epochs run, best epoch, seconds per epoch, peak GPU memory |

Model names: question_only, image_only, early, late, cross_attention, gated.

## Adding a model

1. Add a class to `src/models.py` with a `needs` tuple and the constructor `(d_q, d_v, **hyperparameters)`, and register it in `MODELS`.
2. Add its hyperparameters under `models:` in the config.
3. Add its name to `TRAINABLE` in `src/train.py`.

## Rules

- Never use `split == "test"` for tuning, thresholds or early stopping. Tune on dev only.
- Every reported number must come from a config file and a seed in `runs/`.
- Run `python -m tests.test_smoke` before opening a pull request.
