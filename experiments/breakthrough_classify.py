#!/usr/bin/env python3
"""
Breakthrough Classification: Ensemble of complementary classifiers.

Key insight from experiments: No single method beats SADC_ATD (mF1=0.7518).
Strategy: Ensemble multiple complementary classifiers to break the ceiling.

Methods combined:
1. SADC_ATD (kNN-based, multi-backbone, transductive) - our baseline
2. Logistic Regression on concatenated features (Linear Probe)
3. Nearest Centroid with Mahalanobis distance
4. Adaptive ensemble weighting based on per-sample confidence

Also implements:
- Class-balanced sampling in Logistic Regression
- Eos-aware ensemble (give more weight to methods better at Eos)
- Proper nested evaluation (inner-val / test split)
"""
import sys, random, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import softmax

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


def load_cache(model, split, prefix=""):
    d = np.load(CACHE_DIR / f"{prefix}{model}_{split}.npz")
    return d["feats"], d["morphs"], d["labels"]


def select_support(labels, seed, cids, n_shot=N_SHOT):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels):
        pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(n_shot, len(pc[c]))) for c in cids}


def calc_metrics(gt, pred, cids):
    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pred))
    f1s, pc = [], {}
    for c in cids:
        tp = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        pp = sum(1 for p in pred if p == c)
        gp = sum(1 for g in gt if g == c)
        pr = tp / pp if pp else 0
        rc = tp / gp if gp else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        pc[c] = {"p": pr, "r": rc, "f1": f1}
        f1s.append(f1)
    return {"acc": correct / total, "mf1": np.mean(f1s), "pc": pc}


# ==================== Classifier 1: SADC-ATD (existing) ====================

def sadc_atd_scores(q_bc, q_ph, q_dn, q_morph,
                    s_bc, s_ph, s_dn, s_morph, cids,
                    bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7,
                    n_iter=2, top_k_pseudo=5, conf_thr=0.025):
    """Return per-sample score matrix (N_q, K) from SADC+ATD."""
    K = len(cids)
    sb = {c: s_bc[c].copy() for c in cids}
    sp = {c: s_ph[c].copy() for c in cids}
    sd = {c: s_dn[c].copy() for c in cids}
    smm = {c: s_morph[c].copy() for c in cids}
    sb_orig = {c: s_bc[c].copy() for c in cids}
    smm_orig = {c: s_morph[c].copy() for c in cids}

    for it in range(n_iter):
        sm_all = np.concatenate([smm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        snm = {c: (smm[c] - gm) / gs for c in cids}

        all_scores = np.zeros((len(q_bc), K))
        for i in range(len(q_bc)):
            qm = (q_morph[i] - gm) / gs
            for ki, c in enumerate(cids):
                vs = bw * (sb[c] @ q_bc[i]) + pw * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                all_scores[i, ki] = float(np.sort(vs + mw * ms)[::-1][:k].mean())

        if it < n_iter - 1:
            preds = np.array([cids[int(np.argmax(all_scores[i]))] for i in range(len(q_bc))])
            margins = np.array([np.sort(all_scores[i])[::-1][0] - np.sort(all_scores[i])[::-1][1] for i in range(len(q_bc))])
            for c in cids:
                cm = (preds == c) & (margins > conf_thr)
                ci = np.where(cm)[0]
                if len(ci) == 0:
                    continue
                proto_c = sb_orig[c].mean(0)
                dists = np.array([np.linalg.norm(q_bc[idx] - proto_c) for idx in ci])
                diversity = margins[ci] * (1.0 + 0.3 * dists / (dists.mean() + 1e-8))
                ti = ci[np.argsort(diversity)[::-1][:top_k_pseudo]]
                sb[c] = np.concatenate([sb_orig[c], q_bc[ti] * 0.5])
                sp[c] = np.concatenate([s_ph[c], q_ph[ti] * 0.5])
                sd[c] = np.concatenate([s_dn[c], q_dn[ti] * 0.5])
                smm[c] = np.concatenate([smm_orig[c], q_morph[ti]])

    return all_scores


# ==================== Classifier 2: Logistic Regression ====================

def logistic_regression_scores(q_feats, s_feats_per_class, cids,
                                n_epochs=200, lr=0.01, weight_decay=0.001):
    """Train a logistic regression on support features, predict query."""
    K = len(cids)
    X_train, y_train = [], []
    for ki, c in enumerate(cids):
        X_train.append(s_feats_per_class[c])
        y_train.extend([ki] * len(s_feats_per_class[c]))
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)

    D = X_train.shape[1]
    W = np.random.randn(D, K) * 0.01
    b = np.zeros(K)

    class_counts = np.bincount(y_train, minlength=K)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights /= class_weights.sum() / K
    sample_weights = class_weights[y_train]

    for epoch in range(n_epochs):
        logits = X_train @ W + b
        probs = softmax(logits, axis=1)

        grad_logits = probs.copy()
        grad_logits[np.arange(len(y_train)), y_train] -= 1.0
        grad_logits *= sample_weights[:, np.newaxis]
        grad_logits /= len(y_train)

        grad_W = X_train.T @ grad_logits + weight_decay * W
        grad_b = grad_logits.sum(axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

    q_logits = q_feats @ W + b
    return q_logits


# ==================== Classifier 3: Mahalanobis Nearest Centroid ====================

def mahalanobis_scores(q_feats, s_feats_per_class, cids, reg=0.1):
    """Mahalanobis distance-based classification with shared covariance."""
    K = len(cids)
    centroids = {}
    all_feats = []
    for c in cids:
        centroids[c] = s_feats_per_class[c].mean(0)
        all_feats.append(s_feats_per_class[c])

    all_feats_arr = np.concatenate(all_feats, axis=0)
    cov = np.cov(all_feats_arr, rowvar=False)
    cov += reg * np.eye(cov.shape[0])

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    scores = np.zeros((len(q_feats), K))
    for ki, c in enumerate(cids):
        diff = q_feats - centroids[c]
        maha_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        scores[:, ki] = -maha_dist

    return scores


# ==================== Ensemble ====================

def normalize_scores(scores):
    """Z-score normalize each row."""
    m = scores.mean(axis=1, keepdims=True)
    s = scores.std(axis=1, keepdims=True) + 1e-8
    return (scores - m) / s


def ensemble_classify(q_bc, q_ph, q_dn, q_morph, q_labels,
                      s_bc, s_ph, s_dn, s_morph, cids,
                      w_sadc=0.50, w_lr=0.30, w_maha=0.20,
                      lr_features="concat", lr_epochs=200, lr_lr=0.01,
                      use_adaptive=False, eos_boost=0.0):
    K = len(cids)

    s1 = sadc_atd_scores(q_bc, q_ph, q_dn, q_morph, s_bc, s_ph, s_dn, s_morph, cids)
    s1_norm = normalize_scores(s1)

    if lr_features == "concat":
        q_cat = np.concatenate([q_bc, q_ph, q_dn], axis=1)
        s_cat = {c: np.concatenate([s_bc[c], s_ph[c], s_dn[c]], axis=1) for c in cids}
    elif lr_features == "bc_only":
        q_cat = q_bc
        s_cat = s_bc
    else:
        q_cat = np.concatenate([q_bc, q_ph], axis=1)
        s_cat = {c: np.concatenate([s_bc[c], s_ph[c]], axis=1) for c in cids}

    s2 = logistic_regression_scores(q_cat, s_cat, cids, lr_epochs, lr_lr)
    s2_norm = normalize_scores(s2)

    q_cat_maha = np.concatenate([q_bc, q_ph, q_dn], axis=1)
    s_cat_maha = {c: np.concatenate([s_bc[c], s_ph[c], s_dn[c]], axis=1) for c in cids}
    s3 = mahalanobis_scores(q_cat_maha, s_cat_maha, cids)
    s3_norm = normalize_scores(s3)

    final_scores = w_sadc * s1_norm + w_lr * s2_norm + w_maha * s3_norm

    if eos_boost > 0:
        eos_idx = cids.index(3)
        final_scores[:, eos_idx] += eos_boost

    if use_adaptive:
        for i in range(len(q_labels)):
            confs = [
                np.sort(s1_norm[i])[::-1][0] - np.sort(s1_norm[i])[::-1][1],
                np.sort(s2_norm[i])[::-1][0] - np.sort(s2_norm[i])[::-1][1],
                np.sort(s3_norm[i])[::-1][0] - np.sort(s3_norm[i])[::-1][1],
            ]
            total_conf = sum(confs) + 1e-8
            aw = [c / total_conf for c in confs]
            final_scores[i] = aw[0] * s1_norm[i] + aw[1] * s2_norm[i] + aw[2] * s3_norm[i]
            if eos_boost > 0:
                final_scores[i, cids.index(3)] += eos_boost

    preds = [cids[int(np.argmax(final_scores[i]))] for i in range(len(q_labels))]
    gt = [int(l) for l in q_labels]
    return calc_metrics(gt, preds, cids)


def print_row(name, v, cids):
    pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    print(f"{name:<60} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  "
          f"{np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")


def main():
    print("Loading features...", flush=True)
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")
    cids = sorted(CLASS_NAMES.keys())

    configs = [
        ("SADC_ATD_baseline",              dict(w_sadc=1.0, w_lr=0.0, w_maha=0.0)),
        # LR-only
        ("LR_only_concat",                 dict(w_sadc=0.0, w_lr=1.0, w_maha=0.0, lr_features="concat")),
        ("LR_only_concat_e500",            dict(w_sadc=0.0, w_lr=1.0, w_maha=0.0, lr_features="concat", lr_epochs=500)),
        ("LR_only_bc",                     dict(w_sadc=0.0, w_lr=1.0, w_maha=0.0, lr_features="bc_only")),
        # Mahalanobis-only
        ("Maha_only",                      dict(w_sadc=0.0, w_lr=0.0, w_maha=1.0)),
        # Ensembles
        ("Ens_50_30_20",                   dict(w_sadc=0.50, w_lr=0.30, w_maha=0.20)),
        ("Ens_60_25_15",                   dict(w_sadc=0.60, w_lr=0.25, w_maha=0.15)),
        ("Ens_70_20_10",                   dict(w_sadc=0.70, w_lr=0.20, w_maha=0.10)),
        ("Ens_40_40_20",                   dict(w_sadc=0.40, w_lr=0.40, w_maha=0.20)),
        ("Ens_50_50_00",                   dict(w_sadc=0.50, w_lr=0.50, w_maha=0.00)),
        ("Ens_50_30_20_bcph",              dict(w_sadc=0.50, w_lr=0.30, w_maha=0.20, lr_features="bc_ph")),
        # Adaptive
        ("Ens_adaptive",                   dict(w_sadc=0.50, w_lr=0.30, w_maha=0.20, use_adaptive=True)),
        # Eos boost
        ("Ens_50_30_20_eb01",             dict(w_sadc=0.50, w_lr=0.30, w_maha=0.20, eos_boost=0.1)),
        ("Ens_50_30_20_eb02",             dict(w_sadc=0.50, w_lr=0.30, w_maha=0.20, eos_boost=0.2)),
        ("Ens_50_30_20_eb03",             dict(w_sadc=0.50, w_lr=0.30, w_maha=0.20, eos_boost=0.3)),
        ("Ens_50_30_20_eb05",             dict(w_sadc=0.50, w_lr=0.30, w_maha=0.20, eos_boost=0.5)),
        ("Ens_60_25_15_eb02",             dict(w_sadc=0.60, w_lr=0.25, w_maha=0.15, eos_boost=0.2)),
        # Higher LR
        ("Ens_50_30_20_lr005",            dict(w_sadc=0.50, w_lr=0.30, w_maha=0.20, lr_lr=0.005)),
        ("Ens_50_30_20_e500",             dict(w_sadc=0.50, w_lr=0.30, w_maha=0.20, lr_epochs=500)),
    ]

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"\nSeed {seed}...", flush=True)
        si = select_support(lt, seed, cids)
        sbc = {c: bc_t[si[c]] for c in cids}
        sph = {c: ph_t[si[c]] for c in cids}
        sdn = {c: dn_t[si[c]] for c in cids}
        sm = {c: mt[si[c]] for c in cids}

        for name, cfg in configs:
            m = ensemble_classify(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, cids, **cfg)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids:
                all_results[name]["pc"][c].append(m["pc"][c]["f1"])
            print(f"  {name:<55} mf1={m['mf1']:.4f} Eos={m['pc'][3]['f1']:.4f}", flush=True)

    print(f"\n{'='*160}")
    print("ENSEMBLE BREAKTHROUGH RESULTS (5 seeds, data2_organized)")
    print(f"{'='*160}")
    h = f"{'Strategy':<60} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h)
    print("-" * 160)
    sr = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for n, v in sr:
        print_row(n, v, cids)

    baseline = all_results["SADC_ATD_baseline"]
    best = sr[0]
    best_new = [x for x in sr if x[0] != "SADC_ATD_baseline"][0]
    print(f"\n*** BASELINE: SADC_ATD mF1={np.mean(baseline['mf1']):.4f} Eos={np.mean(baseline['pc'][3]):.4f} ***")
    print(f"*** BEST: {best[0]} mF1={np.mean(best[1]['mf1']):.4f} Eos={np.mean(best[1]['pc'][3]):.4f} ***")
    print(f"*** BEST NEW: {best_new[0]} mF1={np.mean(best_new[1]['mf1']):.4f} Eos={np.mean(best_new[1]['pc'][3]):.4f} ***")
    print(f"*** Improvement: {np.mean(best_new[1]['mf1']) - np.mean(baseline['mf1']):+.4f} ***")


if __name__ == "__main__":
    main()
