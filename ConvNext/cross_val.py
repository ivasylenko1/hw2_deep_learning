import copy
import os
import random
import time

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from CustomAdam import CustomAdam
from ImageDataset import ImageDataset

MODEL_NAME = "convnext_tiny.fb_in1k"
# MODEL_NAME = "resnet50.a1_in1k"
H, W = 384, 448
REPEAT = 5
BATCH_SIZE = 32
USE_BF16 = True
N_FOLDS = 5
max_epochs, min_delta, patience = 25, 1e-5, 8
LR, WD = 2e-4, 1e-2
GAMMA = 0.95
DROP_RATE = 0.2
CLIP_NORM = 1.0
NUM_WORKERS = 6
TRAIN_NPY = "Data/Arrays/train.npy"
FOLDS_CSV = "Data/folds/train_folds.csv"
RUN = f"{MODEL_NAME}_{H}x{W}_rep{REPEAT}"

USE_WANDB = True
WANDB_PROJECT = "cats-and-dogs"
GRAD_PER_PARAM = False


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


def grad_norms(model, per_param=False):
    if per_param:
        return {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
    sq = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        parts = name.split(".")
        key = ".".join(parts[:2]) if len(parts) > 1 and parts[1].isdigit() else parts[0]   # e.g. stages.0, layer1.0, head
        sq[key] = sq.get(key, 0.) + p.grad.norm().item() ** 2
    return {k: v ** 0.5 for k, v in sq.items()}


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
    tuned = macro_f1(y_true, oof_prob >= best_thresholds(y_true, oof_prob))   # optimistic

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

    wandb.init(
        project=WANDB_PROJECT,
        name=RUN,
        mode=None if USE_WANDB else "disabled",
        tags=[MODEL_NAME, f"{H}x{W}", f"rep{REPEAT}"],
        config={
            "model": MODEL_NAME,
            "resolution": f"{H}x{W}",
            "rare_repeat": REPEAT,
            "lr": LR,
            "weight_decay": WD,
            "optimizer": "CustomAdam",
            "scheduler": "ExponentialLR",
            "gamma": GAMMA,
            "drop_rate": DROP_RATE,
            "grad_clip": CLIP_NORM,
            "batch_size": BATCH_SIZE,
            "bf16": USE_BF16,
            "n_folds": N_FOLDS,
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "pos_weight": "sqrt(neg/pos) on the oversampled training rows",
            "augmentation": "flip, affine(scale 0.9-1.1, rot 15), brightness/contrast 0.15",
            "seed": 42,
        },
    )
    wandb.define_metric("epoch")

    train_tf, val_tf = build_transforms()

    cfg = timm.data.resolve_data_config(
        {}, model=timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(target_cols)))
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    def prep(x):
        return ((x.to(device).float() / 255) - mean) / std

    loader_kw = dict(num_workers=NUM_WORKERS, pin_memory=True,
                     persistent_workers=NUM_WORKERS > 0,
                     worker_init_fn=seed_worker if NUM_WORKERS > 0 else None)

    os.makedirs("checkpoints", exist_ok=True)

    for fold in range(N_FOLDS):
        wandb.define_metric(f"fold_{fold}/*", step_metric="epoch")

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
        model = timm.create_model(MODEL_NAME, pretrained=True,
                                  num_classes=len(target_cols), drop_rate=DROP_RATE).to(device)
        optimizer = CustomAdam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = ExponentialLR(optimizer, gamma=GAMMA)

        for epoch in range(max_epochs):
            t0 = time.time()
            model.train()
            train_loss, clip_sum = 0., 0.
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch, y_batch = prep(x_batch), y_batch.to(device)
                optimizer.zero_grad()
                with amp():
                    outputs = model(x_batch)
                loss = criterion(outputs.float(), y_batch)
                loss.backward()
                if i == len(train_loader) - 1:
                    gnorms = grad_norms(model, GRAD_PER_PARAM)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
                optimizer.step()
                train_loss += loss.item()
                clip_sum += total_norm.item()
            lr_now = optimizer.param_groups[0]["lr"]
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

            wandb.log({
                "epoch": epoch,
                f"fold_{fold}/train_loss": train_loss,
                f"fold_{fold}/val_loss": val_loss,
                f"fold_{fold}/lr": lr_now,
                f"fold_{fold}/macro_f1": f1s.mean(),
                f"fold_{fold}/grad_norm_total": clip_sum / len(train_loader),
                f"fold_{fold}/epoch_time_s": time.time() - t0,
                **{f"fold_{fold}/f1_{c}": f for c, f in zip(target_cols, f1s)},
                **{f"fold_{fold}/grad_norm/{k}": v for k, v in gnorms.items()},
            })

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
        wandb.summary[f"fold_{fold}/best_epoch"] = best_epoch
        wandb.summary[f"fold_{fold}/best_val_loss"] = best_loss

    done = folds < N_FOLDS
    y_done = Y[done].astype(int)
    res = evaluate_oof(y_done, oof[done], folds[done])
    print(f"[{RUN}]", res)
    f1_oof = per_class_f1(y_done, oof[done] >= 0.5)
    print("per-class F1 @0.5:", np.round(f1_oof, 3))

    eps = 1e-7
    oc = np.clip(oof[done], eps, 1 - eps)
    cv_loss = -np.mean(Y[done] * np.log(oc) + (1 - Y[done]) * np.log(1 - oc))
    print("CV log loss:", cv_loss)

    wandb.log({"cv_log_loss": cv_loss})
    wandb.summary.update({
        "cv/macro_f1_fixed_0.5": res["fixed_0.5"],
        "cv/macro_f1_tuned_honest": res["tuned_honest"],
        "cv/macro_f1_tuned_on_oof": res["tuned_on_oof"],
        **{f"cv/f1_{c}": f for c, f in zip(target_cols, f1_oof)},
    })

    (pd.DataFrame(oof, columns=target_cols)
       .assign(image_id=labels["image_id"].values)[["image_id"] + target_cols]
       .to_csv(f"{RUN}_oof.csv", index=False))        # rows of folds that were not run stay 0

    wandb.finish()


if __name__ == "__main__":
    main()