import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from ImageDataset import ImageDataset

MODEL_NAME = "convnext_small.fb_in1k"
N_FOLDS = 5
RUN = "convnext_small.fb_in1k_native_rep5_sqrtpw"
CKPTS = [f"checkpoints/{RUN}_fold{k}.pt" for k in range(N_FOLDS)]
CLASSES = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
THRESHOLDS = [0.5, 0.5, 0.5, 0.5, 0.5]
TEST_NPY, TEST_CSV = "Data/Arrays/test.npy", "Data/test.csv"
OUT = f"submissions/{RUN}_ens"
BATCH_SIZE, NUM_WORKERS = 32, 6

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = timm.data.resolve_data_config({}, model=timm.create_model(MODEL_NAME, pretrained=False, num_classes=5))
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    ids = pd.read_csv(TEST_CSV)["image_id"].values
    n = len(ids)
    loader = DataLoader(ImageDataset(TEST_NPY, np.arange(n), np.zeros((n, 5), np.float32),
                                     transform=A.Compose([])),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    total = 0
    for path in CKPTS:
        ck = torch.load(path, map_location=device)
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=5).to(device)
        model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        model.eval()
        probs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for x, _ in loader:
                x = ((x.to(device).float() / 255) - mean) / std
                probs.append(((torch.sigmoid(model(x).float()) + torch.sigmoid(model(x.flip(3)).float())) / 2).cpu())
        total = total + torch.cat(probs).numpy()
        print("done", path)
    probs = total / len(CKPTS)

    p = pd.DataFrame(probs, columns=CLASSES); p.insert(0, "image_id", ids)
    p.to_csv(f"{OUT}_test_probs.csv", index=False)
    sub = pd.DataFrame((probs >= np.array(THRESHOLDS)).astype(int), columns=CLASSES); sub.insert(0, "image_id", ids)
    sub.to_csv(f"{OUT}_submission.csv", index=False)
    print("test positive rate:", sub[CLASSES].mean().round(4).values)