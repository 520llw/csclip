#!/usr/bin/env python3
"""
Feature transformation experiments for 10-shot classification.
Approaches:
1. Tukey's power transform: f_i -> sign(f_i)|f_i|^λ (reduces heavy tails)
2. Feature centering: subtract global/class mean
3. QR decomposition-based subspace projection
4. Support mixup augmentation in feature space
5. Sinkhorn prototype alignment (optimal transport inspired)
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


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
        pr = tp / pp if pp else 0.0
        rc = tp / gp if gp else 0.0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
        pc[c] = {"p": pr, "r": rc, "f1": f1, "n": gp}
        f1s.append(f1)
    return {"acc": correct/total if total else 0, "mf1": float(np.mean(f1s)), "pc": pc}


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels): pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def tukey_transform(feats, lam=0.5):
    """Element-wise power transform: sign(f)|f|^λ"""
    sign = np.sign(feats)
    return sign * np.abs(feats) ** lam


def l2_norm(feats):
    norms = np.linalg.norm(feats, axis=-1, keepdims=True)
    return feats / (norms + 1e-10)


def center_features(feats, mean_vec):
    return feats - mean_vec


def mixup_support(feats, n_aug=5, alpha=0.7):
    """Create augmented support via intra-class mixup."""
    n = len(feats)
    aug = []
    for _ in range(n_aug * n):
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i == j:
            j = (j + 1) % n
        lam = np.random.beta(alpha, alpha)
        aug.append(lam * feats[i] + (1-lam) * feats[j])
    return np.array(aug, dtype=np.float32)


def noise_support(feats, n_aug=3, sigma=0.02):
    """Create augmented support via Gaussian noise."""
    aug = []
    for _ in range(n_aug):
        noisy = feats + np.random.randn(*feats.shape).astype(np.float32) * sigma
        aug.append(noisy)
    return np.concatenate(aug, axis=0)


def classify_knn(q_feats, s_feats_dict, cids, k=7):
    """Simple kNN on fused features."""
    preds = []
    for i in range(len(q_feats)):
        scores = []
        for c in cids:
            dists = s_feats_dict[c] @ q_feats[i]
            scores.append(float(np.sort(dists)[::-1][:k].mean()))
        preds.append(cids[int(np.argmax(scores))])
    return preds


def classify_dual_tukey(q_bclip, q_dino, q_morph, q_labels,
                         s_bclip, s_dino, s_morph,
                         cids, lam=0.5, bw=0.45, dw=0.20, mw=0.35, k=7):
    """Dual-backbone kNN with Tukey-transformed features."""
    s_b_t = {c: l2_norm(tukey_transform(s_bclip[c], lam)) for c in cids}
    s_d_t = {c: l2_norm(tukey_transform(s_dino[c], lam)) for c in cids}
    q_b_t = l2_norm(tukey_transform(q_bclip, lam))
    q_d_t = l2_norm(tukey_transform(q_dino, lam))

    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        scores = []
        for c in cids:
            vs_b = s_b_t[c] @ q_b_t[i]
            vs_d = s_d_t[c] @ q_d_t[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def classify_centered(q_bclip, q_dino, q_morph, q_labels,
                       s_bclip, s_dino, s_morph,
                       cids, bw=0.45, dw=0.20, mw=0.35, k=7):
    """Dual-backbone kNN with centered features."""
    all_s_b = np.concatenate([s_bclip[c] for c in cids])
    all_s_d = np.concatenate([s_dino[c] for c in cids])
    mean_b = all_s_b.mean(0)
    mean_d = all_s_d.mean(0)

    s_b_c = {c: l2_norm(s_bclip[c] - mean_b) for c in cids}
    s_d_c = {c: l2_norm(s_dino[c] - mean_d) for c in cids}
    q_b_c = l2_norm(q_bclip - mean_b)
    q_d_c = l2_norm(q_dino - mean_d)

    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        scores = []
        for c in cids:
            vs_b = s_b_c[c] @ q_b_c[i]
            vs_d = s_d_c[c] @ q_d_c[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def classify_tukey_centered(q_bclip, q_dino, q_morph, q_labels,
                             s_bclip, s_dino, s_morph,
                             cids, lam=0.5, bw=0.45, dw=0.20, mw=0.35, k=7):
    """Tukey + centering combined."""
    s_b_raw = {c: tukey_transform(s_bclip[c], lam) for c in cids}
    s_d_raw = {c: tukey_transform(s_dino[c], lam) for c in cids}
    q_b_raw = tukey_transform(q_bclip, lam)
    q_d_raw = tukey_transform(q_dino, lam)

    all_sb = np.concatenate([s_b_raw[c] for c in cids])
    all_sd = np.concatenate([s_d_raw[c] for c in cids])
    mean_b, mean_d = all_sb.mean(0), all_sd.mean(0)

    s_b_t = {c: l2_norm(s_b_raw[c] - mean_b) for c in cids}
    s_d_t = {c: l2_norm(s_d_raw[c] - mean_d) for c in cids}
    q_b_t = l2_norm(q_b_raw - mean_b)
    q_d_t = l2_norm(q_d_raw - mean_d)

    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        scores = []
        for c in cids:
            vs_b = s_b_t[c] @ q_b_t[i]
            vs_d = s_d_t[c] @ q_d_t[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def classify_augmented_support(q_bclip, q_dino, q_morph, q_labels,
                                s_bclip, s_dino, s_morph,
                                cids, n_mix=3, sigma=0.01,
                                bw=0.45, dw=0.20, mw=0.35, k=7):
    """Support augmentation in feature space (mixup + noise)."""
    s_b_aug = {}
    s_d_aug = {}
    s_m_aug = {}
    for c in cids:
        mixb = mixup_support(s_bclip[c], n_aug=n_mix)
        mixd = mixup_support(s_dino[c], n_aug=n_mix)
        mixm = mixup_support(s_morph[c], n_aug=n_mix)
        noiseb = noise_support(s_bclip[c], n_aug=1, sigma=sigma)
        noised = noise_support(s_dino[c], n_aug=1, sigma=sigma)
        noisem = noise_support(s_morph[c], n_aug=1, sigma=0.1)
        s_b_aug[c] = l2_norm(np.concatenate([s_bclip[c], mixb, noiseb]))
        s_d_aug[c] = l2_norm(np.concatenate([s_dino[c], mixd, noised]))
        s_m_aug[c] = np.concatenate([s_morph[c], mixm, noisem])

    sm_all = np.concatenate([s_m_aug[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_m_aug[c]-gm)/gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        scores = []
        for c in cids:
            vs_b = s_b_aug[c] @ q_bclip[i]
            vs_d = s_d_aug[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def classify_tukey_trans_cascade(q_bclip, q_dino, q_morph, q_labels,
                                  s_bclip, s_dino, s_morph,
                                  cids, morph_weights,
                                  lam=0.5, n_iter=2, top_k=5,
                                  conf_thr=0.025, cascade_thr=0.01,
                                  bw=0.45, dw=0.20, mw=0.35, k=7):
    """Tukey transform + transductive + cascade (all combined)."""
    s_b_t = {c: l2_norm(tukey_transform(s_bclip[c], lam)) for c in cids}
    s_d_t = {c: l2_norm(tukey_transform(s_dino[c], lam)) for c in cids}
    q_b_t = l2_norm(tukey_transform(q_bclip, lam))
    q_d_t = l2_norm(tukey_transform(q_dino, lam))

    s_b_work = {c: s_b_t[c].copy() for c in cids}
    s_d_work = {c: s_d_t[c].copy() for c in cids}
    s_m_work = {c: s_morph[c].copy() for c in cids}

    for _t in range(n_iter):
        sm_all = np.concatenate([s_m_work[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
        snm = {c: (s_m_work[c]-gm)/gs for c in cids}

        preds, margins = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i]-gm)/gs
            scores = []
            for c in cids:
                vs_b = s_b_work[c] @ q_b_t[i]
                vs_d = s_d_work[c] @ q_d_t[i]
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0/(1.0+md)
                comb = bw*vs_b + dw*vs_d + mw*ms
                scores.append(float(np.sort(comb)[::-1][:k].mean()))
            s_arr = np.array(scores)
            sorted_s = np.sort(s_arr)[::-1]
            preds.append(cids[int(np.argmax(s_arr))])
            margins.append(sorted_s[0] - sorted_s[1])
        preds = np.array(preds)
        margins = np.array(margins)

        for c in cids:
            c_mask = (preds == c) & (margins > conf_thr)
            c_idx = np.where(c_mask)[0]
            if len(c_idx) == 0: continue
            sorted_idx = c_idx[np.argsort(margins[c_idx])[::-1][:top_k]]
            s_b_work[c] = np.concatenate([s_b_t[c], q_b_t[sorted_idx]*0.5])
            s_d_work[c] = np.concatenate([s_d_t[c], q_d_t[sorted_idx]*0.5])
            s_m_work[c] = np.concatenate([s_morph[c], q_morph[sorted_idx]])

    sm_all = np.concatenate([s_m_work[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_m_work[c]-gm)/gs for c in cids}
    snm_w = {c: (s_m_work[c]-gm)/gs * morph_weights for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        qm_w = qm * morph_weights
        scores = {}
        for c in cids:
            vs_b = s_b_work[c] @ q_b_t[i]
            vs_d = s_d_work[c] @ q_d_t[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores[c] = float(np.sort(comb)[::-1][:k].mean())

        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0] - np.sort(s_arr)[::-1][1]

        if top1 in [3, 4] and margin < cascade_thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(s_b_work[gc] @ q_b_t[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(s_d_work[gc] @ q_d_t[i])[::-1][:3].mean())
                scores[gc] = 0.30*vs_b_s + 0.15*vs_d_s + 0.55*mscore
            top1 = 3 if scores[3] > scores[4] else 4

        gt.append(int(q_labels[i]))
        pred.append(top1)
    return metrics(gt, pred, cids)


def classify_augmix_trans_cascade(q_bclip, q_dino, q_morph, q_labels,
                                   s_bclip, s_dino, s_morph,
                                   cids, morph_weights,
                                   n_mix=2, sigma=0.015,
                                   n_iter=2, top_k=5,
                                   conf_thr=0.025, cascade_thr=0.01,
                                   bw=0.45, dw=0.20, mw=0.35, k=7):
    """Support augmentation + transductive + cascade."""
    s_b_init, s_d_init, s_m_init = {}, {}, {}
    for c in cids:
        mixb = mixup_support(s_bclip[c], n_aug=n_mix)
        mixd = mixup_support(s_dino[c], n_aug=n_mix)
        mixm = mixup_support(s_morph[c], n_aug=n_mix)
        noiseb = noise_support(s_bclip[c], n_aug=1, sigma=sigma)
        noised = noise_support(s_dino[c], n_aug=1, sigma=sigma)
        noisem = noise_support(s_morph[c], n_aug=1, sigma=0.05)
        s_b_init[c] = l2_norm(np.concatenate([s_bclip[c], mixb, noiseb]))
        s_d_init[c] = l2_norm(np.concatenate([s_dino[c], mixd, noised]))
        s_m_init[c] = np.concatenate([s_morph[c], mixm, noisem])

    s_b_work = {c: s_b_init[c].copy() for c in cids}
    s_d_work = {c: s_d_init[c].copy() for c in cids}
    s_m_work = {c: s_m_init[c].copy() for c in cids}

    for _t in range(n_iter):
        sm_all = np.concatenate([s_m_work[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
        snm = {c: (s_m_work[c]-gm)/gs for c in cids}

        preds, margins_arr = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i]-gm)/gs
            scores = []
            for c in cids:
                vs_b = s_b_work[c] @ q_bclip[i]
                vs_d = s_d_work[c] @ q_dino[i]
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0/(1.0+md)
                comb = bw*vs_b + dw*vs_d + mw*ms
                scores.append(float(np.sort(comb)[::-1][:k].mean()))
            s_arr = np.array(scores)
            sorted_s = np.sort(s_arr)[::-1]
            preds.append(cids[int(np.argmax(s_arr))])
            margins_arr.append(sorted_s[0] - sorted_s[1])
        preds = np.array(preds)
        margins_arr = np.array(margins_arr)

        for c in cids:
            c_mask = (preds == c) & (margins_arr > conf_thr)
            c_idx = np.where(c_mask)[0]
            if len(c_idx) == 0: continue
            sorted_idx = c_idx[np.argsort(margins_arr[c_idx])[::-1][:top_k]]
            s_b_work[c] = np.concatenate([s_b_init[c], q_bclip[sorted_idx]*0.5])
            s_d_work[c] = np.concatenate([s_d_init[c], q_dino[sorted_idx]*0.5])
            s_m_work[c] = np.concatenate([s_m_init[c], q_morph[sorted_idx]])

    sm_all = np.concatenate([s_m_work[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_m_work[c]-gm)/gs for c in cids}
    snm_w = {c: (s_m_work[c]-gm)/gs * morph_weights for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        qm_w = qm * morph_weights
        scores = {}
        for c in cids:
            vs_b = s_b_work[c] @ q_bclip[i]
            vs_d = s_d_work[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores[c] = float(np.sort(comb)[::-1][:k].mean())

        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0] - np.sort(s_arr)[::-1][1]

        if top1 in [3, 4] and margin < cascade_thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(s_b_work[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(s_d_work[gc] @ q_dino[i])[::-1][:3].mean())
                scores[gc] = 0.30*vs_b_s + 0.15*vs_d_s + 0.55*mscore
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
    mw = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos[:,d])-np.mean(neu[:,d]))**2 / (np.var(eos[:,d])+np.var(neu[:,d])+1e-10)
        mw[d] = 1.0 + f * 2.0

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"Seed {seed}...")
        np.random.seed(seed)
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}

        # 1) Tukey transform sweep
        for lam in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            name = f"tukey_l{lam}"
            m = classify_dual_tukey(bclip_val, dino_val, morph_val, labels_val,
                                     s_bclip, s_dino, s_morph, cids, lam)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 2) Centering
        name = "centered"
        m = classify_centered(bclip_val, dino_val, morph_val, labels_val,
                              s_bclip, s_dino, s_morph, cids)
        all_results[name]["acc"].append(m["acc"])
        all_results[name]["mf1"].append(m["mf1"])
        for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 3) Tukey + centering
        for lam in [0.4, 0.5, 0.6]:
            name = f"tukey_center_l{lam}"
            m = classify_tukey_centered(bclip_val, dino_val, morph_val, labels_val,
                                        s_bclip, s_dino, s_morph, cids, lam)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 4) Feature-space augmentation
        for n_mix in [2, 3, 5]:
            for sigma in [0.01, 0.02, 0.03]:
                name = f"augmix_m{n_mix}_s{sigma}"
                m = classify_augmented_support(bclip_val, dino_val, morph_val, labels_val,
                                                s_bclip, s_dino, s_morph, cids, n_mix, sigma)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 5) Tukey + transductive + cascade (combined best ideas)
        for lam in [0.5, 0.6, 0.7, 0.8, 1.0]:
            name = f"tukey_trans_cas_l{lam}"
            m = classify_tukey_trans_cascade(
                bclip_val, dino_val, morph_val, labels_val,
                s_bclip, s_dino, s_morph, cids, mw, lam)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 6) Aug + transductive + cascade
        for n_mix in [2, 3]:
            for sigma in [0.01, 0.015, 0.02]:
                name = f"aug_trans_cas_m{n_mix}_s{sigma}"
                m = classify_augmix_trans_cascade(
                    bclip_val, dino_val, morph_val, labels_val,
                    s_bclip, s_dino, s_morph, cids, mw,
                    n_mix, sigma)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

    print(f"\n{'='*130}")
    print("FEATURE TRANSFORM RESULTS (5 seeds)")
    print(f"{'='*130}")
    header = f"{'Strategy':<50} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 130)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:25]:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    print(f"\n--- Best by Eos F1 ---")
    sorted_eos = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["pc"][3]))
    for name, v in sorted_eos[:15]:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST overall: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}")


if __name__ == "__main__":
    main()
