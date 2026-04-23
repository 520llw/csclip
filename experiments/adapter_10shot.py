#!/usr/bin/env python3
"""
Lightweight adapter fine-tuning for 10-shot classification.
Train a small MLP on top of frozen BiomedCLIP+DINOv2 features.

Key design choices for 40-sample regime:
- Very small model (feature_dim → 64 → 4)
- Strong L2 regularization (weight_decay)
- High dropout
- Balanced class weights in loss
- Short training (few epochs)
- Support augmentation via mixup in feature space
"""
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_cache(model, split):
    d = np.load(CACHE_DIR / f"{model}_{split}.npz")
    return d["feats"], d["morphs"], d["labels"]


def metrics(gt, pred, cids):
    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pred))
    pc, f1s = {}, []
    for c in cids:
        tp = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        pp = sum(1 for p in pred if p == c)
        gp = sum(1 for g in gt if g == c)
        pr = tp/pp if pp else 0.0
        rc = tp/gp if gp else 0.0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
        pc[c] = {"p": pr, "r": rc, "f1": f1, "n": gp}
        f1s.append(f1)
    return {"acc": correct/total if total else 0, "mf1": float(np.mean(f1s)), "pc": pc}


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels): pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


class AdapterMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, n_classes=4, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def augment_features(feats, labels, n_aug=10, alpha=0.7, noise_std=0.02):
    """Augment support with mixup + noise."""
    aug_feats, aug_labels = [feats.copy()], [labels.copy()]
    for _ in range(n_aug):
        n = len(feats)
        idx1 = np.arange(n)
        idx2 = np.random.permutation(n)
        same_class = labels[idx1] == labels[idx2]
        lam = np.random.beta(alpha, alpha, size=n)
        mixed = lam[:, None] * feats[idx1] + (1 - lam[:, None]) * feats[idx2]
        mixed[same_class] = feats[idx1[same_class]] + np.random.randn(same_class.sum(), feats.shape[1]).astype(np.float32) * noise_std
        aug_feats.append(mixed.astype(np.float32))
        aug_labels.append(labels.copy())
    return np.concatenate(aug_feats), np.concatenate(aug_labels)


def train_adapter(train_feats, train_labels, cids, hidden_dim=64, dropout=0.5,
                   lr=0.01, wd=0.01, epochs=100, n_aug=10):
    """Train adapter MLP on support features."""
    label_map = {c: i for i, c in enumerate(cids)}
    y = np.array([label_map[int(l)] for l in train_labels])

    if n_aug > 0:
        feats_aug, labels_aug = augment_features(train_feats, y, n_aug)
    else:
        feats_aug, labels_aug = train_feats.copy(), y.copy()

    class_counts = np.bincount(labels_aug, minlength=len(cids)).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(cids)
    cw_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    X = torch.FloatTensor(feats_aug).to(DEVICE)
    Y = torch.LongTensor(labels_aug).to(DEVICE)

    model = AdapterMLP(train_feats.shape[1], hidden_dim, len(cids), dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for epoch in range(epochs):
        logits = model(X)
        loss = F.cross_entropy(logits, Y, weight=cw_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, label_map


def predict_adapter(model, feats, label_map):
    inv_map = {v: k for k, v in label_map.items()}
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(feats).to(DEVICE)
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return [inv_map[int(p)] for p in preds], probs


def cls_adapter_cascade(q_bclip, q_dino, q_morph, q_labels,
                         s_bclip, s_dino, s_morph,
                         cids, morph_weights,
                         hidden_dim=64, dropout=0.5, lr=0.01, wd=0.01,
                         epochs=100, n_aug=10, cascade_thr=0.1):
    """Adapter + cascade for Eos/Neu."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8

    train_feats_list, train_labels_list = [], []
    for c in cids:
        norm_m = (s_morph[c]-gm)/gs
        fused = np.concatenate([s_bclip[c], s_dino[c], norm_m * 0.5], axis=1)
        train_feats_list.append(fused)
        train_labels_list.extend([c]*len(s_bclip[c]))
    train_feats = np.concatenate(train_feats_list)
    train_labels = np.array(train_labels_list)

    model, label_map = train_adapter(train_feats, train_labels, cids,
                                       hidden_dim, dropout, lr, wd, epochs, n_aug)

    q_feats_fused = []
    for i in range(len(q_labels)):
        norm_m = (q_morph[i]-gm)/gs
        q_feats_fused.append(np.concatenate([q_bclip[i], q_dino[i], norm_m * 0.5]))
    q_feats_fused = np.array(q_feats_fused)

    preds, probs = predict_adapter(model, q_feats_fused, label_map)

    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    snm_w = {c: (s_morph[c]-gm)/gs * morph_weights for c in cids}

    final_preds = []
    for i in range(len(q_labels)):
        pred = preds[i]
        prob = probs[i]
        sorted_prob = np.sort(prob)[::-1]
        margin = sorted_prob[0] - sorted_prob[1]

        if pred in [3, 4] and margin < cascade_thr:
            qm_w = ((q_morph[i]-gm)/gs) * morph_weights
            scores = {}
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b = float(np.sort(s_bclip[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d = float(np.sort(s_dino[gc] @ q_dino[i])[::-1][:3].mean())
                scores[gc] = 0.30*vs_b + 0.15*vs_d + 0.55*mscore
            pred = 3 if scores[3] > scores[4] else 4
        final_preds.append(pred)

    gt = [int(l) for l in q_labels]
    return metrics(gt, final_preds, cids)


def cls_adapter_knn_ensemble(q_bclip, q_dino, q_morph, q_labels,
                              s_bclip, s_dino, s_morph,
                              cids, morph_weights,
                              hidden_dim=64, dropout=0.5, lr=0.01, wd=0.01,
                              epochs=100, n_aug=10, adapter_w=0.4, knn_w=0.6):
    """Ensemble: adapter logits + kNN scores, then cascade."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8

    train_feats_list, train_labels_list = [], []
    for c in cids:
        norm_m = (s_morph[c]-gm)/gs
        fused = np.concatenate([s_bclip[c], s_dino[c], norm_m * 0.5], axis=1)
        train_feats_list.append(fused)
        train_labels_list.extend([c]*len(s_bclip[c]))
    train_feats = np.concatenate(train_feats_list)
    train_labels = np.array(train_labels_list)

    model, label_map = train_adapter(train_feats, train_labels, cids,
                                       hidden_dim, dropout, lr, wd, epochs, n_aug)

    q_feats_fused = []
    for i in range(len(q_labels)):
        norm_m = (q_morph[i]-gm)/gs
        q_feats_fused.append(np.concatenate([q_bclip[i], q_dino[i], norm_m * 0.5]))
    q_feats_fused = np.array(q_feats_fused)

    _, probs = predict_adapter(model, q_feats_fused, label_map)

    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    snm_w = {c: (s_morph[c]-gm)/gs * morph_weights for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        qm_w = qm * morph_weights

        knn_scores = []
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = 0.45*vs_b + 0.20*vs_d + 0.35*ms
            knn_scores.append(float(np.sort(comb)[::-1][:7].mean()))
        knn_arr = np.array(knn_scores)
        knn_arr = (knn_arr - knn_arr.min()) / (knn_arr.max() - knn_arr.min() + 1e-10)

        adapter_scores = probs[i]

        final_scores = adapter_w * adapter_scores + knn_w * knn_arr

        top1_idx = int(np.argmax(final_scores))
        top1 = cids[top1_idx]
        margin = np.sort(final_scores)[::-1][0] - np.sort(final_scores)[::-1][1]

        if top1 in [3, 4] and margin < 0.05:
            scores = {}
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b = float(np.sort(s_bclip[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d = float(np.sort(s_dino[gc] @ q_dino[i])[::-1][:3].mean())
                scores[gc] = 0.30*vs_b + 0.15*vs_d + 0.55*mscore
            top1 = 3 if scores[3] > scores[4] else 4

        gt.append(int(q_labels[i]))
        pred.append(top1)
    return metrics(gt, pred, cids)


def main():
    bclip_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bclip_val, morph_val, labels_val = load_cache("biomedclip", "val")
    dino_train, _, _ = load_cache("dinov2_s", "train")
    dino_val, _, _ = load_cache("dinov2_s", "val")

    cids = sorted(CLASS_NAMES.keys())

    eos, neu = morph_train[labels_train==3], morph_train[labels_train==4]
    n_dims = morph_train.shape[1]
    fisher_w = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos[:,d])-np.mean(neu[:,d]))**2 / (np.var(eos[:,d])+np.var(neu[:,d])+1e-10)
        fisher_w[d] = 1.0 + f * 2.0

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"Seed {seed}...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}

        # Adapter-only with cascade (focused sweep)
        configs = [
            (64, 0.5, 0.05, 100, 10, 0.1),
            (64, 0.5, 0.01, 100, 10, 0.1),
            (64, 0.5, 0.1,  100, 10, 0.1),
            (64, 0.3, 0.05, 100, 10, 0.1),
            (32, 0.5, 0.05, 100, 10, 0.1),
            (128, 0.5, 0.05, 100, 10, 0.1),
            (64, 0.5, 0.05, 200, 10, 0.1),
            (64, 0.5, 0.05, 50,  10, 0.1),
            (64, 0.5, 0.05, 100, 20, 0.1),
            (64, 0.5, 0.05, 100, 5,  0.1),
            (64, 0.5, 0.05, 100, 10, 0.05),
            (64, 0.5, 0.05, 100, 10, 0.2),
            (64, 0.5, 0.05, 100, 0,  0.1),
        ]
        for hd, dp, wd, ep, na, cthr in configs:
            name = f"adpt_h{hd}_d{dp}_w{wd}_e{ep}_a{na}_c{cthr}"
            m = cls_adapter_cascade(
                bclip_val, dino_val, morph_val, labels_val,
                s_bclip, s_dino, s_morph, cids, fisher_w,
                hd, dp, 0.01, wd, ep, na, cthr)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # Adapter+kNN ensemble
        for aw in [0.3, 0.4, 0.5]:
            name = f"adpt_knn_aw{aw}"
            m = cls_adapter_knn_ensemble(
                bclip_val, dino_val, morph_val, labels_val,
                s_bclip, s_dino, s_morph, cids, fisher_w,
                adapter_w=aw, knn_w=1.0-aw)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

    print(f"\n{'='*130}")
    print("ADAPTER RESULTS (5 seeds)")
    print(f"{'='*130}")
    header = f"{'Strategy':<55} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 130)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:30]:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<55} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    print(f"\n--- Best by Eos F1 ---")
    sorted_eos = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["pc"][3]))
    for name, v in sorted_eos[:15]:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<55} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}")


if __name__ == "__main__":
    main()
