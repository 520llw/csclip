#!/usr/bin/env python3
"""
Feature Transformation + Advanced Classification for BALF Cells.

New approaches:
1. Support-Supervised PCA: Project features to discriminative subspace
2. Power Transform: Apply centering + L2 re-normalization for better separation
3. Tukey's Ladder of Powers: Feature calibration for tail classes
4. NCM + Class-specific Scaling: Different scales per class
5. Combined: Best feature transform + best classifier
"""
import sys, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import softmax

sys.stdout.reconfigure(line_buffering=True)

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


# ==================== Feature Transforms ====================

def power_transform(feats, alpha=0.5):
    """Power normalization: element-wise power followed by L2 normalization.
    Shown to improve few-shot classification by reducing hubness."""
    sign = np.sign(feats)
    powered = sign * np.abs(feats) ** alpha
    norms = np.linalg.norm(powered, axis=1, keepdims=True) + 1e-8
    return powered / norms


def centering(q_feats, s_feats_all):
    """Mean-centering using support mean, then L2 normalize."""
    center = s_feats_all.mean(axis=0)
    q_c = q_feats - center
    q_c /= np.linalg.norm(q_c, axis=1, keepdims=True) + 1e-8
    return q_c, center


def support_lda_transform(s_feats_per_class, cids, n_components=None):
    """Support-only LDA projection for dimensionality reduction."""
    K = len(cids)
    if n_components is None:
        n_components = K - 1

    X, y = [], []
    for ki, c in enumerate(cids):
        X.append(s_feats_per_class[c])
        y.extend([ki] * len(s_feats_per_class[c]))
    X = np.concatenate(X, axis=0)
    y = np.array(y)
    D = X.shape[1]

    global_mean = X.mean(axis=0)
    Sw = np.zeros((D, D))
    Sb = np.zeros((D, D))
    for ki in range(K):
        Xi = X[y == ki]
        mean_i = Xi.mean(axis=0)
        diff = Xi - mean_i
        Sw += diff.T @ diff
        nd = (mean_i - global_mean).reshape(-1, 1)
        Sb += len(Xi) * (nd @ nd.T)

    Sw += 1e-4 * np.eye(D)
    try:
        eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(Sw) @ Sb)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(Sw) @ Sb)

    idx = np.argsort(eigvals)[::-1][:n_components]
    W = eigvecs[:, idx]
    return W, global_mean


# ==================== Classification Methods ====================

def ncm_classify(q_feats, s_feats_per_class, cids, metric="cosine"):
    """Nearest Centroid Method."""
    K = len(cids)
    centroids = np.zeros((K, q_feats.shape[1]))
    for ki, c in enumerate(cids):
        centroids[ki] = s_feats_per_class[c].mean(0)
        if metric == "cosine":
            centroids[ki] /= np.linalg.norm(centroids[ki]) + 1e-8

    if metric == "cosine":
        q_norm = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-8)
        scores = q_norm @ centroids.T
    else:
        scores = np.zeros((len(q_feats), K))
        for ki in range(K):
            diff = q_feats - centroids[ki]
            scores[:, ki] = -np.linalg.norm(diff, axis=1)
    return scores


def knn_classify(q_feats, s_feats_per_class, cids, k=7):
    """k-NN classification."""
    K = len(cids)
    scores = np.zeros((len(q_feats), K))
    for ki, c in enumerate(cids):
        sims = q_feats @ s_feats_per_class[c].T
        for i in range(len(q_feats)):
            topk = np.sort(sims[i])[::-1][:min(k, len(s_feats_per_class[c]))]
            scores[i, ki] = topk.mean()
    return scores


def full_pipeline(q_bc, q_ph, q_dn, q_morph, q_labels,
                  s_bc, s_ph, s_dn, s_morph, cids,
                  transform="none", power_alpha=0.5,
                  classifier="knn", k=7,
                  bw=0.42, pw=0.18, dw=0.07, mw=0.33,
                  use_lda=False, lda_comp=3,
                  use_center=False, use_atd=True):
    K = len(cids)

    def apply_transform(feats, s_feats_per_class):
        if transform == "power":
            feats = power_transform(feats, power_alpha)
            for c in cids:
                s_feats_per_class[c] = power_transform(s_feats_per_class[c], power_alpha)
        if use_center:
            s_all = np.concatenate([s_feats_per_class[c] for c in cids])
            feats, center = centering(feats, s_all)
            for c in cids:
                s_feats_per_class[c] = s_feats_per_class[c] - center
                s_feats_per_class[c] /= np.linalg.norm(s_feats_per_class[c], axis=1, keepdims=True) + 1e-8
        if use_lda:
            W, gm = support_lda_transform(s_feats_per_class, cids, lda_comp)
            feats = (feats - gm) @ W
            feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            for c in cids:
                s_feats_per_class[c] = (s_feats_per_class[c] - gm) @ W
                s_feats_per_class[c] /= np.linalg.norm(s_feats_per_class[c], axis=1, keepdims=True) + 1e-8
        return feats, s_feats_per_class

    sb = {c: s_bc[c].copy() for c in cids}
    sp = {c: s_ph[c].copy() for c in cids}
    sd = {c: s_dn[c].copy() for c in cids}
    smm = {c: s_morph[c].copy() for c in cids}

    q_bc_t, sb_t = apply_transform(q_bc.copy(), {c: sb[c].copy() for c in cids})
    q_ph_t, sp_t = apply_transform(q_ph.copy(), {c: sp[c].copy() for c in cids})
    q_dn_t, sd_t = apply_transform(q_dn.copy(), {c: sd[c].copy() for c in cids})

    sb_orig = {c: sb_t[c].copy() for c in cids}
    smm_orig = {c: smm[c].copy() for c in cids}

    n_iter = 2 if use_atd else 1
    for it in range(n_iter):
        sm_all = np.concatenate([smm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        snm = {c: (smm[c] - gm) / gs for c in cids}

        all_scores = np.zeros((len(q_labels), K))
        for i in range(len(q_labels)):
            qm = (q_morph[i] - gm) / gs
            for ki, c in enumerate(cids):
                if classifier == "knn":
                    vs = bw * (sb_t[c] @ q_bc_t[i]) + pw * (sp_t[c] @ q_ph_t[i]) + dw * (sd_t[c] @ q_dn_t[i])
                    md = np.linalg.norm(qm - snm[c], axis=1)
                    ms = 1.0 / (1.0 + md)
                    all_scores[i, ki] = float(np.sort(vs + mw * ms)[::-1][:k].mean())
                elif classifier == "ncm":
                    proto_bc = sb_t[c].mean(0); proto_bc /= np.linalg.norm(proto_bc) + 1e-8
                    proto_ph = sp_t[c].mean(0); proto_ph /= np.linalg.norm(proto_ph) + 1e-8
                    proto_dn = sd_t[c].mean(0); proto_dn /= np.linalg.norm(proto_dn) + 1e-8
                    q_bc_n = q_bc_t[i] / (np.linalg.norm(q_bc_t[i]) + 1e-8)
                    q_ph_n = q_ph_t[i] / (np.linalg.norm(q_ph_t[i]) + 1e-8)
                    q_dn_n = q_dn_t[i] / (np.linalg.norm(q_dn_t[i]) + 1e-8)
                    vs = bw * (q_bc_n @ proto_bc) + pw * (q_ph_n @ proto_ph) + dw * (q_dn_n @ proto_dn)
                    proto_m = snm[c].mean(0)
                    ms = 1.0 / (1.0 + np.linalg.norm(qm - proto_m))
                    all_scores[i, ki] = vs + mw * ms

        if use_atd and it < n_iter - 1:
            preds = np.array([cids[int(np.argmax(all_scores[i]))] for i in range(len(q_labels))])
            margins = np.array([np.sort(all_scores[i])[::-1][0] - np.sort(all_scores[i])[::-1][1] for i in range(len(q_labels))])
            for c in cids:
                cm = (preds == c) & (margins > 0.025)
                ci = np.where(cm)[0]
                if len(ci) == 0:
                    continue
                proto_c = sb_orig[c].mean(0)
                dists = np.array([np.linalg.norm(q_bc_t[idx] - proto_c) for idx in ci])
                diversity = margins[ci] * (1.0 + 0.3 * dists / (dists.mean() + 1e-8))
                ti = ci[np.argsort(diversity)[::-1][:5]]
                sb_t[c] = np.concatenate([sb_orig[c], q_bc_t[ti] * 0.5])
                sp_t[c] = np.concatenate([sp_t[c][:len(sp[c])], q_ph_t[ti] * 0.5])
                sd_t[c] = np.concatenate([sd_t[c][:len(sd[c])], q_dn_t[ti] * 0.5])
                smm[c] = np.concatenate([smm_orig[c], q_morph[ti]])

    preds = [cids[int(np.argmax(all_scores[i]))] for i in range(len(q_labels))]
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
        ("baseline_SADC_ATD",              dict()),
        # Power Transform
        ("power_05",                       dict(transform="power", power_alpha=0.5)),
        ("power_03",                       dict(transform="power", power_alpha=0.3)),
        ("power_07",                       dict(transform="power", power_alpha=0.7)),
        # Centering
        ("center",                         dict(use_center=True)),
        ("center+power05",                 dict(use_center=True, transform="power", power_alpha=0.5)),
        # LDA
        ("lda3",                           dict(use_lda=True, lda_comp=3)),
        ("lda3+center",                    dict(use_lda=True, lda_comp=3, use_center=True)),
        ("lda3+power05",                   dict(use_lda=True, lda_comp=3, transform="power", power_alpha=0.5)),
        # NCM classifier
        ("ncm_baseline",                   dict(classifier="ncm")),
        ("ncm+power05",                    dict(classifier="ncm", transform="power", power_alpha=0.5)),
        ("ncm+center",                     dict(classifier="ncm", use_center=True)),
        # Without ATD
        ("no_atd",                         dict(use_atd=False)),
        ("power05_no_atd",                 dict(transform="power", power_alpha=0.5, use_atd=False)),
        ("center_no_atd",                  dict(use_center=True, use_atd=False)),
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
            m = full_pipeline(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, cids, **cfg)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids:
                all_results[name]["pc"][c].append(m["pc"][c]["f1"])
            print(f"  {name:<55} mf1={m['mf1']:.4f} Eos={m['pc'][3]['f1']:.4f}", flush=True)

    print(f"\n{'='*160}")
    print("FEATURE TRANSFORM RESULTS (5 seeds, data2_organized)")
    print(f"{'='*160}")
    h = f"{'Strategy':<60} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h)
    print("-" * 160)
    sr = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for n, v in sr:
        print_row(n, v, cids)

    baseline = all_results["baseline_SADC_ATD"]
    best = sr[0]
    print(f"\n*** BASELINE: mF1={np.mean(baseline['mf1']):.4f} Eos={np.mean(baseline['pc'][3]):.4f} ***")
    print(f"*** BEST: {best[0]} mF1={np.mean(best[1]['mf1']):.4f} Eos={np.mean(best[1]['pc'][3]):.4f} ***")
    print(f"*** Improvement: {np.mean(best[1]['mf1']) - np.mean(baseline['mf1']):+.4f} ***")


if __name__ == "__main__":
    main()
