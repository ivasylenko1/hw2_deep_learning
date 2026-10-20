# Cats & Dogs: multilabel image classification

AI301 Deep Learning, Assignment 2. Kaggle in-class competition `cats-and-dogs`.
Each image gets five independent labels: `Cat`, `Dog` and three named animals `Moris`, `Motya`, `Biatrix`.
The metric is **macro-F1** over the five labels.

**Final submission:** a 5-fold ensemble of ConvNeXt-Small (pretrained, fine-tuned at native image size), threshold 0.5,
horizontal-flip test-time augmentation. Public leaderboard score: **0.98401**.

> Items marked **TODO** are facts I could not verify from the code and should be checked before publishing.

---

## 1. Task and data

| | |
|---|---|
| Train | 12,260 images, stored as one array `Data/Arrays/train.npy`, shape `(12260, 375, 450, 3)`, uint8 |
| Test | `Data/Arrays/test.npy`, same image size |
| Labels | `Data/folds/train_folds.csv`: `image_id, Cat, Dog, Moris, Motya, Biatrix, fold` |
| Folds | 5 fixed folds, shared by every experiment, so out-of-fold (OOF) files are comparable and can be blended |

- Cat and Dog are common (about 40% of images each). The three named animals are rare (about 1% each, roughly 125 images in total).
  Because of that, the rare-class F1 values move by several points when only a few images flip, and this drives most of the noise in every comparison below.
- The labels are hierarchical: Moris and Motya occur with `Cat`, Biatrix occurs with `Dog`. The model does not use this; it is a plain 5-output classifier.

## 2. Method

- **Loss:** `BCEWithLogitsLoss` with `pos_weight = sqrt(neg / pos)` computed on the oversampled training rows.
- **Rare images:** every training image that has any named label is repeated **5×** per epoch (`np.repeat` on the training indices only, never on validation rows). Each copy gets its own random augmentation.
- **Augmentation:** horizontal flip, affine (scale 0.9–1.1, rotation ±15°), brightness/contrast 0.15.
- **Training:** pretrained timm backbone, `CustomAdam` (the course optimizer), lr 2e-4, weight decay 1e-2, dropout 0.2, gradient clipping 1.0, bf16 autocast, batch size 32, early stopping on validation loss, best-epoch checkpoint per fold.
- **Evaluation:** 5-fold cross-validation. Metrics on the OOF predictions: macro-F1 at 0.5, log loss, and per-class F1.
- **Prediction:** average of the 5 fold models, each with flip TTA.
- **Logging:** Weights & Biases, project `cats-and-dogs` (loss curves, learning rate, per-class F1, gradient norms per block).

The final model (`cross_val_small_convnext.py`) differs from the earlier runs in two ways: it uses **no resizing at all** (images enter the network at the stored 375×450) and a cosine learning-rate schedule from 2e-4 to 1e-6 with patience 5.
Its recipe is adapted from a classmate's ConvNeXt-Small notebook (see Credits).

## 3. Results

### Cross-validation (OOF, all 5 folds)

| Model | Resolution | macro-F1 @0.5 | macro-F1, nested thresholds | Log loss |
|---|---|---|---|---|
| ResNet50 (`resnet50.a1_in1k`) | 384×448 | 0.9706 | 0.9714 | 0.00744 |
| ConvNeXt-Tiny (`convnext_tiny.fb_in1k`) | 384×448 | 0.9715 | 0.9569 | 0.00697 |
| ConvNeXt-Small, capped `pos_weight` | native | 0.9688 | 0.9513 | 0.00733 |
| **ConvNeXt-Small, `sqrt` `pos_weight` (final)** | native | **0.9793** | 0.9760 | **0.00670** |

Per-class F1 of the final model: Cat 0.995, Dog 0.994, Moris 0.956, Motya 0.988, Biatrix 0.964.

A first from-scratch CNN reached about 0.62 macro-F1 in an early baseline run. **TODO:** add the result of the residual from-scratch CNN (`cross_val_cnn.py`) if you have it.

### Leaderboard

| Submission | Score |
|---|---|
| ResNet50, one model on all data (`train.py`), thr 0.5 | 0.96454 |
| same, with tuned thresholds | 0.96779 |
| ResNet50 5-fold ensemble | 0.97010 |
| ResNet50 5-fold ensemble, tuned thresholds | 0.96631 |
| ResNet50 5-fold ensemble + the all-data model | 0.97268 |
| single all-data timm model (**TODO:** confirm which model) | 0.9702 |
| blend: ResNet folds + ResNet single + ConvNeXt-Tiny folds (1:1:2) | 0.97629 |
| **ConvNeXt-Small 5-fold ensemble (final)** | **0.98401** |
| OOF grid blend of ResNet50 / ConvNeXt-Tiny / ConvNeXt-Small (0.2 / 0.2 / 0.6) | 0.9799 |

## 4. What I learned

- **The `pos_weight` formula mattered more than the architecture.** Capping `neg/pos` at 20 on top of the 5× repeat made the rare classes over-predicted: Moris had precision 0.871 against recall 0.964. Switching to `sqrt(neg/pos)` raised Moris F1 from 0.915 to 0.956 and gave the best model.
- **Threshold tuning did not help reliably.** Per-class thresholds chosen on the OOF predictions looked good on OOF (0.989 for ConvNeXt-Tiny) but not under nested evaluation (0.957). On the leaderboard they gave +0.003 for a single ResNet50 but lowered the score of the ResNet50 fold ensemble, so the final submission uses 0.5.
- **OOF blends are optimistic.** An OOF grid search (`oof_blend_three_way.py`) found a blend that cut log loss by about 27% against the best single model, but the leaderboard score was lower (0.9799 against 0.98401). With about 125 rare images, differences of 0.004 in macro-F1 are a handful of images, so neither number settles which is better.
- **Adversarial validation:** a classifier on frozen ConvNeXt-Tiny features separating train from test reaches AUC 0.508 (0.504 with shuffled labels), so no detectable shift between train and test images and random stratified K-fold is a reasonable validation scheme.
- **Near duplicates:** at least 10.4% of the rare images have a very similar neighbour (cosine similarity above 0.9) in a different fold, which can make CV slightly optimistic. **TODO:** the proposed robustness check without these images was not run.

## 5. Repository layout

**TODO:** check the names against your folder.

| File | Purpose |
|---|---|
| `ImageDataset.py`, `CustomAdam.py` | dataset reading from the `.npy` arrays, course optimizer |
| `model.py`, `cross_val_cnn.py` | from-scratch residual CNN and its cross-validation |
| `cross_val.py` | cross-validation of ResNet50 and ConvNeXt-Tiny (set `MODEL_NAME`, `H, W`) |
| `cross_val_small_convnext.py` | cross-validation of ConvNeXt-Small, native size (final model) |
| `train.py` | one model on all data, no validation |
| `ensemble.py`, `ensemble_small_native.py` | average the fold models on the test set, write probabilities and a submission |
| `oof_blend_three_way.py` | OOF grid search for the blend weights |
| `validation.ipynb` | adversarial validation |
| `close_image_check.py` | near-duplicate check between folds |

Not included: data, checkpoints (`*.pt`, too large for GitHub), local `wandb/` run folders.

## 6. How to run

```bash
pip install -r requirements.txt            # torch, timm, albumentations, scikit-learn, pandas, wandb
wandb login                                 # or set USE_WANDB = False in the script

python cross_val_small_convnext.py          # 5 folds -> checkpoints/*.pt and *_oof.csv   (~75 s/epoch on a modern GPU)
python ensemble_small_native.py             # fold average on the test set -> submissions/*_submission.csv
```

Data layout expected by the scripts: `Data/Arrays/{train,test}.npy`, `Data/folds/train_folds.csv`, `Data/test.csv`.
Seeds are fixed to 42, but GPU training is not bit-for-bit reproducible, so rerunning gives slightly different checkpoints.

## 7. Limitations

- The leaderboard has only a few rare images, so small differences between submissions are not reliable.
- Best epochs are early and noisy (often epoch 1–5), so the fold checkpoints depend partly on which epoch had the lowest validation loss.