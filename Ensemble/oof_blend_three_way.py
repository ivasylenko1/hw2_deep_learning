import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLASSES = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
FOLDS_CSV = "Data/folds/train_folds.csv"

MODELS = {
    "resnet50":       dict(oof="resnet50.a1_in1k_384x448_rep5_oof.csv",
                           test="submissions/ensemble5_test_probs.csv", lb=0.972),
    "convnext_tiny":  dict(oof="convnext_tiny.fb_in1k_384x448_rep5_oof.csv",
                           test="submissions/convnext_tiny.fb_in1k_384x448_rep5_ens_test_probs.csv", lb=0.976),
    "convnext_small": dict(oof="convnext_small.fb_in1k_native_rep5_sqrtpw_oof.csv",
                           test="submissions/convnext_small.fb_in1k_native_rep5_sqrtpw_ens_test_probs.csv", lb=0.984),
}
STEP = 10
TOL = 0.02
N_BOOT = 300
WEIGHTS = None
OUT = "submissions/oof_grid_blend"


def log_loss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def macro_f1(y, p):
    return f1_score(y, p >= 0.5, average="macro", zero_division=0)


def main():
    lab = pd.read_csv(FOLDS_CSV)
    Y = lab[CLASSES].values.astype(int)
    names = list(MODELS)

    oofs = []
    for n in names:
        d = pd.read_csv(MODELS[n]["oof"])
        assert (d["image_id"].values == lab["image_id"].values).all(), f"{n}: OOF rows are not in train_folds.csv order"
        assert (d[CLASSES].values.sum(1) > 0).all(), f"{n}: OOF has empty rows (not all folds were run?)"
        oofs.append(d[CLASSES].values)
        print(f"{n:15s} OOF  logloss {log_loss(Y, oofs[-1]):.5f}  macroF1 {macro_f1(Y, oofs[-1]):.4f}   leaderboard {MODELS[n]['lb']}")

    rows = []
    for combo in itertools.product(range(STEP + 1), repeat=len(names)):
        if sum(combo) != STEP:
            continue
        w = np.array(combo) / STEP
        p = sum(wi * o for wi, o in zip(w, oofs))
        rows.append(dict(**{n: wi for n, wi in zip(names, w)},
                         logloss=log_loss(Y, p), f1=macro_f1(Y, p),
                         prior=float(sum(wi * MODELS[n]["lb"] for wi, n in zip(w, names)))))
    g = pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
    print(f"\n{len(g)} weight combinations. Top 10 by OOF log loss:")
    print(g.head(10).round(4).to_string())

    best_ll = g["logloss"].min()
    tied = g[g["logloss"] <= best_ll * (1 + TOL)]
    winner = tied.sort_values(["prior", "logloss"], ascending=[False, True]).iloc[0]
    print(f"\n{len(tied)} combinations are within {TOL:.0%} of the best OOF log loss ({best_ll:.5f}).")
    print("rule pick (highest leaderboard prior among the tied):")
    print(winner.round(4).to_string())

    w = np.array([winner[n] for n in names]) if WEIGHTS is None else np.array([WEIGHTS[n] for n in names])
    w = w / w.sum()
    print("\nweights used:", dict(zip(names, np.round(w, 3).tolist())))

    best_single = int(np.argmax([MODELS[n]["lb"] for n in names]))
    p_blend = sum(wi * o for wi, o in zip(w, oofs))
    p_single = oofs[best_single]
    rng = np.random.default_rng(0)
    d_f1, d_ll = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(Y), len(Y))
        d_f1.append(macro_f1(Y[i], p_blend[i]) - macro_f1(Y[i], p_single[i]))
        d_ll.append(log_loss(Y[i], p_single[i]) - log_loss(Y[i], p_blend[i]))
    d_f1, d_ll = np.array(d_f1), np.array(d_ll)
    print(f"\nbootstrap, blend vs {names[best_single]} alone ({N_BOOT} resamples of the OOF rows):")
    print(f"  macro-F1 difference  mean {d_f1.mean():+.4f}   blend better in {np.mean(d_f1 > 0):.0%} of resamples")
    print(f"  log-loss gain        mean {d_ll.mean():+.5f}   blend better in {np.mean(d_ll > 0):.0%} of resamples")

    tests = [pd.read_csv(MODELS[n]["test"]) for n in names]
    for n, d in zip(names, tests):
        assert (d["image_id"].values == tests[0]["image_id"].values).all(), f"{n}: test image order differs"
    p_test = sum(wi * d[CLASSES].values for wi, d in zip(w, tests))

    out = tests[0].copy(); out[CLASSES] = p_test
    out.to_csv(f"{OUT}_test_probs.csv", index=False)
    sub = tests[0].copy(); sub[CLASSES] = (p_test >= 0.5).astype(int)
    sub.to_csv(f"{OUT}_submission.csv", index=False)
    print("\nwrote", f"{OUT}_submission.csv")
    print("test positive rate:", sub[CLASSES].mean().round(4).values)


if __name__ == "__main__":
    main()