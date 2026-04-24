#!/usr/bin/env python3
"""Generate per-sample predictions (seed=42) and confusion matrix heatmap."""
import json
import random
from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
CLASS_LABELS = ["Eosinophil", "Neutrophil", "Lymphocyte", "Macrophage"]
N_SHOT = 10
SEED = 42
CIDS = sorted(CLASS_NAMES.keys())


def load_cache(model, split):
    d = np.load(CACHE_DIR / f"{model}_{split}.npz")
    return d["feats"], d["morphs"], d["labels"]


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels):
        pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def classify_best(q_bclip, q_dino, q_morph, q_labels,
                  s_bclip, s_dino, s_morph, cids, morph_weights, thr=0.008):
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    snm = {c: (s_morph[c] - gm) / gs for c in cids}
    snm_w = {c: (s_morph[c] - gm) / gs * morph_weights for c in cids}

    gt, pred, confidences = [], [], []
    details = []

    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm) / gs
        qm_w = qm * morph_weights

        scores = {}
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            comb = 0.45 * vs_b + 0.20 * vs_d + 0.35 * ms
            scores[c] = float(np.sort(comb)[::-1][:7].mean())

        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = float(np.sort(s_arr)[::-1][0] - np.sort(s_arr)[::-1][1])

        refined = False
        if top1 in [3, 4] and margin < thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0 / (1.0 + np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(s_bclip[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(s_dino[gc] @ q_dino[i])[::-1][:3].mean())
                scores[gc] = 0.30 * vs_b_s + 0.15 * vs_d_s + 0.55 * mscore
            top1 = 3 if scores[3] > scores[4] else 4
            refined = True

        gt.append(int(q_labels[i]))
        pred.append(top1)
        confidences.append(margin)
        details.append({
            "gt": int(q_labels[i]), "pred": top1,
            "margin": margin, "refined": refined,
            "scores": {str(c): scores[c] for c in cids}
        })

    return gt, pred, confidences, details


def main():
    bclip_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bclip_val, morph_val, labels_val = load_cache("biomedclip", "val")
    dino_train, _, _ = load_cache("dinov2_s", "train")
    dino_val, _, _ = load_cache("dinov2_s", "val")

    n_dims = morph_train.shape[1]
    eos_morph = morph_train[labels_train == 3]
    neu_morph = morph_train[labels_train == 4]
    morph_weights = np.ones(n_dims, dtype=np.float32)
    for d in range(n_dims):
        mu_diff = (np.mean(eos_morph[:, d]) - np.mean(neu_morph[:, d])) ** 2
        var_sum = np.var(eos_morph[:, d]) + np.var(neu_morph[:, d]) + 1e-10
        fisher = mu_diff / var_sum
        morph_weights[d] = 1.0 + fisher * 2.0

    support_idx = select_support(labels_train, SEED, CIDS)
    s_bclip = {c: bclip_train[support_idx[c]] for c in CIDS}
    s_dino = {c: dino_train[support_idx[c]] for c in CIDS}
    s_morph = {c: morph_train[support_idx[c]] for c in CIDS}

    gt, pred, confs, details = classify_best(
        bclip_val, dino_val, morph_val, labels_val,
        s_bclip, s_dino, s_morph, CIDS, morph_weights)

    # Save CSV
    out_csv = Path("/home/xut/csclip/paper_materials/confusion/per_sample_predictions_seed42.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "true_label", "pred_label", "true_name", "pred_name",
                         "confidence_margin", "refined",
                         "score_eos", "score_neu", "score_lym", "score_mac"])
        for i, (g, p, c, d) in enumerate(zip(gt, pred, confs, details)):
            writer.writerow([
                i, g, p, CLASS_NAMES[g], CLASS_NAMES[p],
                f"{c:.6f}", d["refined"],
                f"{d['scores']['3']:.6f}",
                f"{d['scores']['4']:.6f}",
                f"{d['scores']['5']:.6f}",
                f"{d['scores']['6']:.6f}",
            ])
    print(f"Saved CSV: {out_csv} ({len(gt)} samples)")

    # Compute confusion matrix (numpy ordering matches CLASS_LABELS)
    label_map = {3: 0, 4: 1, 5: 2, 6: 3}
    gt_idx = [label_map[g] for g in gt]
    pred_idx = [label_map[p] for p in pred]
    cm = confusion_matrix(gt_idx, pred_idx, labels=list(range(4)))

    # Also compute per-class metrics
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    acc = accuracy_score(gt_idx, pred_idx)
    prec, rec, f1, _ = precision_recall_fscore_support(gt_idx, pred_idx, labels=list(range(4)), zero_division=0)

    metrics = {
        "seed": SEED,
        "accuracy": float(acc),
        "macro_f1": float(f1.mean()),
        "per_class": {name: {"precision": float(p), "recall": float(r), "f1": float(f)}
                      for name, p, r, f in zip(CLASS_LABELS, prec, rec, f1)}
    }
    with open(Path("/home/xut/csclip/paper_materials/confusion/metrics_seed42.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Accuracy: {acc:.4f}, Macro-F1: {f1.mean():.4f}")

    # Plot confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS, fontsize=12)
    ax.set_yticklabels(CLASS_LABELS, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title(f"Confusion Matrix (Seed={SEED}, n={len(gt)}, Acc={acc:.3f}, Macro-F1={f1.mean():.3f})", fontsize=14)

    # Add text annotations
    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            text = ax.text(j, i, str(cm[i, j]),
                           ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white",
                           fontsize=14, fontweight="bold")

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom", fontsize=12)

    plt.tight_layout()
    out_png = Path("/home/xut/csclip/paper_materials/confusion/confusion_matrix_seed42.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix: {out_png}")

    # Also save normalized version (percentage by row)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS, fontsize=12)
    ax.set_yticklabels(CLASS_LABELS, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title(f"Normalized Confusion Matrix (%, Seed={SEED})", fontsize=14)

    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            text = ax.text(j, i, f"{cm_norm[i, j]:.1f}",
                           ha="center", va="center", color="black" if cm_norm[i, j] < 50 else "white",
                           fontsize=14, fontweight="bold")

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Percentage (%)", rotation=-90, va="bottom", fontsize=12)

    plt.tight_layout()
    out_png2 = Path("/home/xut/csclip/paper_materials/confusion/confusion_matrix_normalized_seed42.png")
    fig.savefig(out_png2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved normalized confusion matrix: {out_png2}")


if __name__ == "__main__":
    main()
