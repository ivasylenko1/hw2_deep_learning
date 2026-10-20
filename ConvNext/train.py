"""Train ONE model on all training data (no CV), save the .pt, predict the test set, write the submission."""
import os
import random
import time

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from CustomAdam import CustomAdam
from ImageDataset import ImageDataset

# MODEL_NAME = "resnet50.a1_in1k"
MODEL_NAME = "convnext_small.fb_in1k" #"convnext_tiny.fb_in1k"
H, W = 384, 448
REPEAT = 5
BATCH_SIZE = 32
EPOCHS = 15
LR, WD = 2e-4, 1e-2
USE_BF16 = True
NUM_WORKERS = 6
THRESHOLD = 0.5
SEED = 42

TRAIN_NPY, TEST_NPY = "Data/Arrays/train.npy", "Data/Arrays/test.npy"
LABELS_CSV, TEST_CSV = "Data/folds/train_folds.csv", "Data/test.csv"
RUN = f"final_{MODEL_NAME}_{H}x{W}_rep{REPEAT}"

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet params

train_tf = A.Compose([
    A.Resize(H, W),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
])
test_tf = A.Compose([A.Resize(H, W)])


def seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
    tf = torch.utils.data.get_worker_info().dataset.transform
    if tf is not None and hasattr(tf, "set_random_seed"):
        tf.set_random_seed(seed)


if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def prep(x):
        return ((x.to(device).float() / 255) - mean) / std

    labels = pd.read_csv(LABELS_CSV)
    target_cols = list(labels.columns[1:-1])
    Y = labels[target_cols].values.astype(np.float32)
    assert np.load(TRAIN_NPY, mmap_mode="r").shape[0] == len(Y), "train.npy and the csv are misaligned"

    idx = np.arange(len(Y))
    rare = idx[Y[:, 2:].any(axis=1)]
    idx = np.concatenate([idx, np.repeat(rare, REPEAT - 1)])
    pos = Y[idx].sum(0)
    pw = torch.tensor(np.sqrt((len(idx) - pos) / pos), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    train_loader = DataLoader(ImageDataset(TRAIN_NPY, idx, Y[idx], transform=train_tf),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0, worker_init_fn=seed_worker)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(target_cols), drop_rate=0.2).to(device)
    optimizer = CustomAdam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total = 0.
        for x, y in train_loader:
            x, y = prep(x), y.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                out = model(x)
            loss = criterion(out.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"epoch {epoch:2d}  train loss {total / len(train_loader):.5f}  {time.time() - t0:.0f}s")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": MODEL_NAME, "H": H, "W": W,
                "classes": target_cols, "epochs": EPOCHS}, f"checkpoints/{RUN}.pt")

    test_ids = pd.read_csv(TEST_CSV)["image_id"].values
    n = len(test_ids)
    test_loader = DataLoader(ImageDataset(TEST_NPY, np.arange(n), np.zeros((n, len(target_cols)), np.float32),
                                          transform=test_tf),
                             batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    probs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
        for x, _ in test_loader:
            x = prep(x)         # average of the image and its horizontal flip
            probs.append(((torch.sigmoid(model(x).float()) + torch.sigmoid(model(x.flip(3)).float())) / 2).cpu())
    probs = torch.cat(probs).numpy()

    os.makedirs("submissions", exist_ok=True)
    out = pd.DataFrame(probs, columns=target_cols); out.insert(0, "image_id", test_ids)
    out.to_csv(f"submissions/{RUN}_test_probs.csv", index=False)
    sub = pd.DataFrame((probs >= THRESHOLD).astype(int), columns=target_cols); sub.insert(0, "image_id", test_ids)
    sub.to_csv(f"submissions/{RUN}_submission.csv", index=False)

    print("test positive rate :", np.round(sub[target_cols].mean().values, 4))
    print("train positive rate:", np.round(labels[target_cols].mean().values, 4))