import copy
import os
import random
import time

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from CustomAdam import CustomAdam
from ImageDataset import ImageDataset
from model import CNN

CHANNELS = 256
H, W = 288, 352
REPEAT = 5
BATCH_SIZE = 32
USE_BF16 = True
N_FOLDS = 2
max_epochs, min_delta, patience = 40, 1e-5, 10
LR, WD = 1e-3, 1e-2
NUM_WORKERS = 6
TRAIN_NPY = "Data/Arrays/train.npy"
FOLDS_CSV = "Data/folds/train_folds.csv"
RUN = f"cnn{CHANNELS}_{H}x{W}_rep{REPEAT}"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
    tf = torch.utils.data.get_worker_info().dataset.transform
    if tf is not None and hasattr(tf, "set_random_seed"):
        tf.set_random_seed(seed)


def build_transforms():
    train_tf = A.Compose([
        A.Resize(H, W),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
    ])
    val_tf = A.Compose([A.Resize(H, W)])      # validation: resize only
    return train_tf, val_tf


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16)


def per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None, zero_division=0)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def best_thresholds(y_true, y_prob, grid=np.arange(0.02, 0.98, 0.01)):
    th = []
    for j in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, j], y_prob[:, j] >= t, zero_division=0) for t in grid]
        th.append(grid[int(np.argmax(scores))])
    return np.array(th)


def evaluate_oof(y_true, oof_prob, fold):
    fixed = macro_f1(y_true, oof_prob >= 0.5)
    tuned = macro_f1(y_true, oof_prob >= best_thresholds(y_true, oof_prob))

    pred = np.zeros_like(y_true)
    for f in np.unique(fold):
        va = fold == f
        th = best_thresholds(y_true[~va], oof_prob[~va])
        pred[va] = oof_prob[va] >= th
    honest = macro_f1(y_true, pred)
    return {"fixed_0.5": fixed, "tuned_on_oof": tuned, "tuned_honest": honest}


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = pd.read_csv(FOLDS_CSV)
    target_cols = list(labels.columns[1:-1])
    Y = labels[target_cols].values.astype(np.float32)
    folds = labels["fold"].values
    oof = np.zeros((len(labels), len(target_cols)), dtype=np.float32)

    train_tf, val_tf = build_transforms()

    def prep(x):
        return x.to(device).float() / 255

    loader_kw = dict(num_workers=NUM_WORKERS, pin_memory=True,
                     persistent_workers=NUM_WORKERS > 0,
                     worker_init_fn=seed_worker if NUM_WORKERS > 0 else None)

    os.makedirs("checkpoints", exist_ok=True)

    for fold in range(N_FOLDS):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        rare = tr_idx[Y[tr_idx][:, 2:].any(axis=1)]
        tr_idx_rep = np.concatenate([tr_idx, np.repeat(rare, REPEAT - 1)])
        print(f"fold {fold}: train rows {len(tr_idx)} -> {len(tr_idx_rep)} "
              f"(rare images {len(rare)}), val rows {len(va_idx)}")

        pos = Y[tr_idx_rep].sum(0)
        neg = len(tr_idx_rep) - pos
        pw = torch.tensor(np.sqrt(neg / pos), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        train_loader = DataLoader(
            ImageDataset(TRAIN_NPY, tr_idx_rep, Y[tr_idx_rep], transform=train_tf),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_kw)
        val_loader = DataLoader(
            ImageDataset(TRAIN_NPY, va_idx, Y[va_idx], transform=val_tf),
            batch_size=BATCH_SIZE * 2, shuffle=False, **loader_kw)
        y_va = torch.from_numpy(Y[va_idx]).to(device)

        best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
        model = CNN(in_channels=3, labels=len(target_cols), channels=CHANNELS).to(device)
        optimizer = CustomAdam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = ExponentialLR(optimizer, gamma=0.95)

        for epoch in range(max_epochs):
            t0 = time.time()
            model.train()
            train_loss = 0.
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = prep(x_batch), y_batch.to(device)
                optimizer.zero_grad()
                with amp():
                    outputs = model(x_batch)
                loss = criterion(outputs.float(), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            train_loss /= len(train_loader)

            model.eval()
            with torch.no_grad(), amp():
                val_logs = torch.cat([model(prep(xb)) for xb, _ in val_loader]).float()
            val_loss = criterion(val_logs, y_va).item()

            probs = torch.sigmoid(val_logs).cpu().numpy()
            f1s = per_class_f1(Y[va_idx].astype(int), probs >= 0.5)
            print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  "
                  f"macroF1 {f1s.mean():.4f}  per-class {np.round(f1s, 2)}  "
                  f"{time.time() - t0:.0f}s")

            if val_loss < best_loss - min_delta:
                best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
                best_logits = val_logs
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("Early stopping")
                    break

        model.load_state_dict(best_state)
        torch.save(best_state, f"checkpoints/{RUN}_fold{fold}.pt")
        oof[va_idx] = torch.sigmoid(best_logits).cpu().numpy()
        print(f"fold {fold}: best epoch {best_epoch}, val loss {best_loss:.5f}")

    done = folds < N_FOLDS
    y_done = Y[done].astype(int)
    print(f"[{RUN}]", evaluate_oof(y_done, oof[done], folds[done]))
    print("per-class F1 @0.5:", np.round(per_class_f1(y_done, oof[done] >= 0.5), 3))

    eps = 1e-7
    oc = np.clip(oof[done], eps, 1 - eps)
    print("CV log loss:", -np.mean(Y[done] * np.log(oc) + (1 - Y[done]) * np.log(1 - oc)))

    (pd.DataFrame(oof, columns=target_cols)
       .assign(image_id=labels["image_id"].values)[["image_id"] + target_cols]
       .to_csv(f"{RUN}_oof.csv", index=False))


if __name__ == "__main__":
    main()