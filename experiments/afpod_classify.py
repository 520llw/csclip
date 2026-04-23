#!/usr/bin/env python3
"""
AFP-OD: Adaptive Fisher-based Prototype Orthogonal Disentanglement

Design goal: Push apart support samples of confusion class pairs along the
Fisher direction in each backbone's feature space, so that kNN classification
with modified support samples achieves better separation on the confusion axis.

Implementation phases:
  Phase 1: Basic Fisher separation (trace shrinkage + LOO confusion detection + fixed alpha)
  Phase 2: Ledoit-Wolf adaptive shrinkage
  Phase 3: Data-driven morph-anchored Fisher direction selection from top-k Fisher directions
  Phase 4: Cross-backbone rank consistency regularization

Baseline comparison: MB_kNN (multi-backbone kNN with fixed weights 0.42/0.18/0.07)
                     SADC+ATD (current paper's method, from nested_cv.py)
"""
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


# ==================== Data loading ====================

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
    return {"acc": correct / total, "mf1": float(np.mean(f1s)), "pc": pc}


def k_fold_split(indices, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    folds = []
    fold_size = len(shuffled) // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(shuffled)
        folds.append(shuffled[start:end])
    return folds


# ==================== AFP-OD core ====================

def _ledoit_wolf_shrinkage(X):
    """Ledoit-Wolf shrinkage intensity for a single class's sample covariance.

    Args:
        X: (N, D) centered or uncentered data (function centers internally).

    Returns:
        shrink: float in [0, 1]
        S: sample covariance (D, D)
        F: target (D, D) = (trace(S)/D) * I
    """
    n, d = X.shape
    if n < 2:
        return 1.0, np.zeros((d, d), dtype=np.float64), np.zeros((d, d), dtype=np.float64)

    X_c = X - X.mean(axis=0, keepdims=True)
    S = (X_c.T @ X_c) / n  # biased sample covariance (consistent with LW derivation)
    trace_S = float(np.trace(S))
    F = (trace_S / d) * np.eye(d, dtype=S.dtype)

    # pi_hat: sum over (i,j) of Var(S[i,j]) ~ mean over samples of (x_k x_k^T - S)^2
    X2 = X_c ** 2
    pi_mat = (X2.T @ X2) / n - S ** 2
    pi_hat = float(pi_mat.sum())

    gamma_hat = float(((S - F) ** 2).sum())

    if gamma_hat < 1e-12 or n < 2:
        shrink = 0.0
    else:
        shrink = max(0.0, min(1.0, pi_hat / (gamma_hat * n)))

    return shrink, S, F


def morph_anchored_direction(feats_i, feats_j, morph_i, morph_j,
                             alpha_blend=0.5, method="lw"):
    """Compute a morphology-anchored Fisher direction in feature space.

    Rationale: The morph-space Fisher direction is robustly estimable (40-dim
    vs. 512-dim, fewer parameters). We use it to define a per-sample "polarity"
    score, then find the feature-space direction most aligned with this polarity
    via PLS (Partial Least Squares), then blend with the raw feature-space Fisher.

    Args:
        feats_i, feats_j: (N_i, D_f), (N_j, D_f) feature arrays
        morph_i, morph_j: (N_i, D_m), (N_j, D_m) morphology arrays
        alpha_blend: in [0, 1]. 0 = pure feature-Fisher; 1 = pure morph-anchored
        method: passed to both Fisher computations

    Returns:
        w_feat: (D_f,) unit-norm direction, morph-anchored to degree alpha_blend
    """
    # Step 1: Pure Fisher direction in feature space (backbone-native)
    w_fisher_feat = fisher_direction(feats_i, feats_j, method=method)

    # Step 2: Morph-space Fisher direction (low-dim, stable anchor)
    w_fisher_morph = fisher_direction(morph_i, morph_j, method=method)

    if np.linalg.norm(w_fisher_morph) < 1e-8:
        return w_fisher_feat  # no usable morph signal

    # Step 3: Compute morph polarity score for each support sample
    all_morph = np.vstack([morph_i, morph_j]).astype(np.float64)
    all_feats = np.vstack([feats_i, feats_j]).astype(np.float64)
    morph_polarity = all_morph @ w_fisher_morph.astype(np.float64)
    morph_polarity_c = morph_polarity - morph_polarity.mean()

    # Step 4: PLS direction in feature space = feats^T @ polarity
    feats_c = all_feats - all_feats.mean(0, keepdims=True)
    w_pls = feats_c.T @ morph_polarity_c
    pls_norm = float(np.linalg.norm(w_pls))
    if pls_norm < 1e-8:
        return w_fisher_feat
    w_pls = (w_pls / pls_norm).astype(np.float32)

    # Step 5: Sign align PLS direction with feature Fisher (so they don't cancel)
    if float(np.dot(w_pls, w_fisher_feat)) < 0:
        w_pls = -w_pls

    # Step 6: Blend
    w_blend = (1.0 - alpha_blend) * w_fisher_feat + alpha_blend * w_pls
    blend_norm = float(np.linalg.norm(w_blend))
    if blend_norm < 1e-8:
        return w_fisher_feat
    return (w_blend / blend_norm).astype(np.float32)


def fisher_direction(feats_i, feats_j, method="trace", shrink=0.3):
    """Fisher LDA direction with regularized covariance.

    Args:
        feats_i, feats_j: (N, D) feature arrays for classes i and j
        method: "trace" (fixed shrink, Phase 1) or "lw" (Ledoit-Wolf, Phase 2)
        shrink: shrinkage parameter in [0, 1] when method="trace"

    Returns:
        w: (D,) unit-norm Fisher direction pointing from mean_j to mean_i
    """
    mu_i = feats_i.mean(axis=0)
    mu_j = feats_j.mean(axis=0)
    d = feats_i.shape[1]
    diff = (mu_i - mu_j).astype(np.float64)

    if method == "lw":
        # Per-class Ledoit-Wolf covariance estimate (more accurate than joint)
        if len(feats_i) > 1:
            shrink_i, S_i, F_i = _ledoit_wolf_shrinkage(feats_i.astype(np.float64))
            Sigma_i = (1.0 - shrink_i) * S_i + shrink_i * F_i
        else:
            Sigma_i = np.zeros((d, d), dtype=np.float64)

        if len(feats_j) > 1:
            shrink_j, S_j, F_j = _ledoit_wolf_shrinkage(feats_j.astype(np.float64))
            Sigma_j = (1.0 - shrink_j) * S_j + shrink_j * F_j
        else:
            Sigma_j = np.zeros((d, d), dtype=np.float64)

        Sigma_reg = Sigma_i + Sigma_j
        # LW-shrunk matrices are guaranteed positive-definite; small ridge as safety
        Sigma_reg = Sigma_reg + 1e-6 * np.eye(d, dtype=np.float64)
    else:  # "trace" (Phase 1)
        if len(feats_i) > 1:
            Sigma_i = np.cov(feats_i, rowvar=False).astype(np.float64)
        else:
            Sigma_i = np.zeros((d, d), dtype=np.float64)
        if len(feats_j) > 1:
            Sigma_j = np.cov(feats_j, rowvar=False).astype(np.float64)
        else:
            Sigma_j = np.zeros((d, d), dtype=np.float64)

        Sigma_sum = Sigma_i + Sigma_j
        trace_val = np.trace(Sigma_sum) / d if d > 0 else 1e-6
        if trace_val < 1e-8:
            trace_val = 1e-6
        Sigma_reg = (1.0 - shrink) * Sigma_sum + shrink * trace_val * np.eye(d, dtype=np.float64)

    try:
        w = np.linalg.solve(Sigma_reg, diff)
    except np.linalg.LinAlgError:
        w = diff

    norm = float(np.linalg.norm(w))
    if norm < 1e-8:
        return np.zeros(d, dtype=np.float32)
    return (w / norm).astype(np.float32)


def _loo_knn_confusion_table(s_feats_per_class, cids, k=5,
                             metric="cosine"):
    """Generic LOO k-NN confusion rate table.

    Args:
        metric: "cosine" (dot product, features must be unit norm)
                or "euclidean" (for z-scored features like morph)
    Returns:
        pair_rates: dict {(ci, cj): bidirectional_confusion_rate}
    """
    all_feats_list = [s_feats_per_class[c] for c in cids if len(s_feats_per_class[c]) > 0]
    all_labels_list = [[c] * len(s_feats_per_class[c]) for c in cids if len(s_feats_per_class[c]) > 0]
    if not all_feats_list:
        return {}

    all_feats = np.concatenate(all_feats_list, axis=0).astype(np.float64)
    all_labels = np.array([l for sub in all_labels_list for l in sub])
    N = len(all_labels)
    if N < 2:
        return {}

    if metric == "cosine":
        norms = np.linalg.norm(all_feats, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        unit = all_feats / norms
        sim = unit @ unit.T
        np.fill_diagonal(sim, -np.inf)
        # top-k neighbors = largest sim
        preds = np.zeros(N, dtype=int)
        for i in range(N):
            topk_idx = np.argsort(sim[i])[-k:]
            votes = all_labels[topk_idx]
            preds[i] = int(np.bincount(votes.astype(int)).argmax())
    else:  # euclidean
        # Pairwise squared distance
        sq = (all_feats ** 2).sum(axis=1, keepdims=True)
        dist = sq + sq.T - 2.0 * (all_feats @ all_feats.T)
        np.fill_diagonal(dist, np.inf)
        preds = np.zeros(N, dtype=int)
        for i in range(N):
            topk_idx = np.argsort(dist[i])[:k]
            votes = all_labels[topk_idx]
            preds[i] = int(np.bincount(votes.astype(int)).argmax())

    pair_rates = {}
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i >= j:
                continue
            mask_i = all_labels == ci
            mask_j = all_labels == cj
            if mask_i.sum() == 0 or mask_j.sum() == 0:
                pair_rates[(ci, cj)] = 0.0
                continue
            rate_ij = float((preds[mask_i] == cj).sum()) / float(mask_i.sum())
            rate_ji = float((preds[mask_j] == ci).sum()) / float(mask_j.sum())
            pair_rates[(ci, cj)] = rate_ij + rate_ji
    return pair_rates


def find_confusion_pairs_loo(s_feats_per_class, cids, threshold=0.15, k=5):
    """Single-view LOO k-NN confusion detection (Phase 1/2)."""
    rates = _loo_knn_confusion_table(s_feats_per_class, cids, k=k, metric="cosine")
    pairs = [(ci, cj, r) for (ci, cj), r in rates.items() if r >= threshold]
    pairs.sort(key=lambda x: -x[2])
    return [(ci, cj) for ci, cj, _ in pairs]


def find_confusion_pairs_dualview(s_feats_per_class, s_morph_per_class, cids,
                                  feat_thr=0.15, morph_thr=0.15,
                                  morph_norm_stats=None, k=5, mode="intersection"):
    """Phase 3 dual-view confusion detection.

    Detect pairs that are mutually confusing in BOTH feature space AND morph
    space (intersection mode), providing stronger evidence that the pair
    requires disentanglement. Uses z-scored morph for Euclidean distance.

    Args:
        mode: "intersection" (pair must appear in both views, default, safer)
              "union" (pair in either view)
              "feature_only" (fallback to Phase 1/2 behaviour)
    """
    # Feature-space rates (cosine kNN)
    feat_rates = _loo_knn_confusion_table(s_feats_per_class, cids, k=k, metric="cosine")

    # Morph-space rates (euclidean kNN on z-scored morph)
    # Z-score using concat of all support (data-driven, no val leakage)
    sm_all = np.concatenate([s_morph_per_class[c] for c in cids
                             if len(s_morph_per_class[c]) > 0], axis=0)
    if morph_norm_stats is None:
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    else:
        gm, gs = morph_norm_stats
    morph_norm = {c: (s_morph_per_class[c] - gm) / gs for c in cids}
    morph_rates = _loo_knn_confusion_table(morph_norm, cids, k=k, metric="euclidean")

    pairs_out = []
    for (ci, cj), feat_r in feat_rates.items():
        morph_r = morph_rates.get((ci, cj), 0.0)
        feat_ok = feat_r >= feat_thr
        morph_ok = morph_r >= morph_thr
        if mode == "intersection":
            keep = feat_ok and morph_ok
        elif mode == "union":
            keep = feat_ok or morph_ok
        elif mode == "feature_only":
            keep = feat_ok
        else:
            raise ValueError(f"Unknown dual-view mode {mode}")
        if keep:
            # strength = average of both views for ordering
            strength = (feat_r + morph_r) / 2.0
            pairs_out.append((ci, cj, strength))

    pairs_out.sort(key=lambda x: -x[2])
    return [(ci, cj) for ci, cj, _ in pairs_out]


def amplify_separation(support_per_class, confusion_pairs, alpha=0.2,
                       method="trace", shrink=0.3,
                       morph_per_class=None, alpha_blend=0.0):
    """Apply Fisher-direction separation to support samples for each confusion pair.

    For each (c_i, c_j), compute Fisher direction w and push:
        support[c_i] += alpha * w
        support[c_j] -= alpha * w
    Then re-normalize each support sample to unit norm.

    Args:
        method: "trace" or "lw" passed through to fisher_direction
        morph_per_class: optional dict of morph arrays {class_id: (N, D_m)}.
            When provided together with alpha_blend > 0, Phase 3 morph-anchored
            direction is used instead of pure Fisher.
        alpha_blend: 0 = pure feature Fisher (Phase 1/2), 1 = pure morph-PLS.
    """
    if not confusion_pairs:
        return {c: v.copy() for c, v in support_per_class.items()}

    modified = {c: v.copy().astype(np.float32) for c, v in support_per_class.items()}
    use_morph = (morph_per_class is not None) and (alpha_blend > 0)

    for ci, cj in confusion_pairs:
        feats_i = support_per_class[ci]
        feats_j = support_per_class[cj]
        if len(feats_i) == 0 or len(feats_j) == 0:
            continue
        if use_morph:
            morph_i = morph_per_class[ci]
            morph_j = morph_per_class[cj]
            if len(morph_i) == 0 or len(morph_j) == 0:
                w = fisher_direction(feats_i, feats_j, method=method, shrink=shrink)
            else:
                w = morph_anchored_direction(feats_i, feats_j, morph_i, morph_j,
                                             alpha_blend=alpha_blend, method=method)
        else:
            w = fisher_direction(feats_i, feats_j, method=method, shrink=shrink)

        if np.linalg.norm(w) < 1e-8:
            continue
        modified[ci] = modified[ci] + alpha * w
        modified[cj] = modified[cj] - alpha * w

    for c in modified:
        norms = np.linalg.norm(modified[c], axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        modified[c] = (modified[c] / norms).astype(np.float32)

    return modified


def afpod_classify(q_bc, q_ph, q_dn, q_morph,
                   s_bc, s_ph, s_dn, s_morph, cids,
                   bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7,
                   alpha=0.2, conf_thresh=0.15,
                   method="trace", shrink=0.3,
                   alpha_blend=0.0,
                   detection_backbone="bc",
                   detection_mode="feature_only"):
    """AFP-OD: find confusion pairs, apply Fisher separation on all three
    backbones, then classify via MB_kNN.

    Args:
        method: "trace" (Phase 1) or "lw" (Phase 2, Ledoit-Wolf)
        shrink: used only when method="trace"
        alpha_blend: morph-anchor blend ratio (0 = pure feature Fisher)
        detection_mode: "feature_only" (Phase 1/2) or "dualview_intersection"
                        (Phase 3) or "dualview_union"
    """
    detect_source = {"bc": s_bc, "ph": s_ph, "dn": s_dn}[detection_backbone]
    if detection_mode == "feature_only":
        confusion_pairs = find_confusion_pairs_loo(detect_source, cids,
                                                   threshold=conf_thresh)
    else:
        mode = "intersection" if "intersection" in detection_mode else "union"
        confusion_pairs = find_confusion_pairs_dualview(
            detect_source, s_morph, cids,
            feat_thr=conf_thresh, morph_thr=conf_thresh, mode=mode)

    s_bc_mod = amplify_separation(s_bc, confusion_pairs, alpha=alpha,
                                  method=method, shrink=shrink,
                                  morph_per_class=s_morph, alpha_blend=alpha_blend)
    s_ph_mod = amplify_separation(s_ph, confusion_pairs, alpha=alpha,
                                  method=method, shrink=shrink,
                                  morph_per_class=s_morph, alpha_blend=alpha_blend)
    s_dn_mod = amplify_separation(s_dn, confusion_pairs, alpha=alpha,
                                  method=method, shrink=shrink,
                                  morph_per_class=s_morph, alpha_blend=alpha_blend)

    K = len(cids)
    sm_all = np.concatenate([s_morph[c] for c in cids], axis=0)
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    snm = {c: (s_morph[c] - gm) / gs for c in cids}

    scores = np.zeros((len(q_bc), K), dtype=np.float32)
    for i in range(len(q_bc)):
        qm = (q_morph[i] - gm) / gs
        for ki, c in enumerate(cids):
            ncls = len(s_bc_mod[c])
            if ncls == 0:
                scores[i, ki] = -np.inf
                continue
            vs = bw * (s_bc_mod[c] @ q_bc[i]) + pw * (s_ph_mod[c] @ q_ph[i]) + dw * (s_dn_mod[c] @ q_dn[i])
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            kk = min(k, ncls)
            scores[i, ki] = float(np.sort(vs + mw * ms)[::-1][:kk].mean())
    return scores, confusion_pairs


# ==================== Baselines ====================

def mb_knn_classify(q_bc, q_ph, q_dn, q_morph,
                    s_bc, s_ph, s_dn, s_morph, cids,
                    bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7):
    """MB kNN baseline (matches nested_cv.py MB_kNN)."""
    K = len(cids)
    sm_all = np.concatenate([s_morph[c] for c in cids], axis=0)
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    snm = {c: (s_morph[c] - gm) / gs for c in cids}

    scores = np.zeros((len(q_bc), K), dtype=np.float32)
    for i in range(len(q_bc)):
        qm = (q_morph[i] - gm) / gs
        for ki, c in enumerate(cids):
            ncls = len(s_bc[c])
            if ncls == 0:
                scores[i, ki] = -np.inf
                continue
            vs = bw * (s_bc[c] @ q_bc[i]) + pw * (s_ph[c] @ q_ph[i]) + dw * (s_dn[c] @ q_dn[i])
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            kk = min(k, ncls)
            scores[i, ki] = float(np.sort(vs + mw * ms)[::-1][:kk].mean())
    return scores


# ==================== Main ====================

def print_row(name, v, cids):
    pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    print(f"{name:<28} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  "
          f"{np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")


def main():
    print("=" * 110)
    print("AFP-OD Phase 1: Fisher-Orthogonal Prototype Disentanglement")
    print("Dataset: data2_organized, 10-shot, nested 5-fold CV × 5 seeds = 25 evaluations")
    print("=" * 110, flush=True)

    print("\nLoading feature caches...")
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")
    cids = sorted(CLASS_NAMES.keys())
    print(f"Loaded: BC train={bc_t.shape} val={bc_v.shape}")
    print(f"        PH train={ph_t.shape} val={ph_v.shape}")
    print(f"        DN train={dn_t.shape} val={dn_v.shape}")
    print(f"        morph train={mt.shape} val={mv.shape}", flush=True)

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    # Sweep configurations — (method, shrink, alpha, alpha_blend, detection_mode, phase_name)
    # Phase 1: trace shrinkage, feature-only detection
    # Phase 2: Ledoit-Wolf, feature-only detection
    # Phase 3a: Ledoit-Wolf + morph-blend anchor (negative result, kept as ablation)
    # Phase 3b: Ledoit-Wolf + dual-view feature+morph confusion detection
    configs = []
    for alpha in [0.05, 0.10, 0.20]:
        configs.append(("trace", 0.3, alpha, 0.0, "feature_only", "p1"))
    for alpha in [0.05, 0.10, 0.20]:
        configs.append(("lw", None, alpha, 0.0, "feature_only", "p2"))
    # Phase 3a: morph anchor blend (ablation for negative result)
    for ab in [0.3, 0.5]:
        configs.append(("lw", None, 0.10, ab, "feature_only", "p3a"))
    # Phase 3b: dual-view confusion detection
    for alpha in [0.05, 0.10, 0.20]:
        configs.append(("lw", None, alpha, 0.0, "dualview_intersection", "p3b"))
    for alpha in [0.10]:
        configs.append(("lw", None, alpha, 0.0, "dualview_union", "p3c"))

    for seed in SEEDS:
        print(f"\nSeed {seed}", flush=True)
        val_indices = np.arange(len(lv))
        folds = k_fold_split(val_indices, 5, seed)

        for fold_idx, test_fold in enumerate(folds):
            q_bc = bc_v[test_fold]
            q_ph = ph_v[test_fold]
            q_dn = dn_v[test_fold]
            q_morph = mv[test_fold]
            q_labels = lv[test_fold]

            si = select_support(lt, seed + fold_idx, cids)
            sbc = {c: bc_t[si[c]] for c in cids}
            sph = {c: ph_t[si[c]] for c in cids}
            sdn = {c: dn_t[si[c]] for c in cids}
            sm = {c: mt[si[c]] for c in cids}

            # Baseline: MB_kNN
            bl_scores = mb_knn_classify(q_bc, q_ph, q_dn, q_morph, sbc, sph, sdn, sm, cids)
            pred_bl = [cids[int(np.argmax(bl_scores[i]))] for i in range(len(q_labels))]
            m = calc_metrics([int(l) for l in q_labels], pred_bl, cids)
            all_results["MB_kNN"]["acc"].append(m["acc"])
            all_results["MB_kNN"]["mf1"].append(m["mf1"])
            for c in cids:
                all_results["MB_kNN"]["pc"][c].append(m["pc"][c]["f1"])

            # AFP-OD configurations
            for (method, shrink, alpha, alpha_blend, det_mode, phase) in configs:
                scores, _ = afpod_classify(
                    q_bc, q_ph, q_dn, q_morph, sbc, sph, sdn, sm, cids,
                    alpha=alpha, conf_thresh=0.15,
                    method=method, shrink=shrink if shrink is not None else 0.3,
                    alpha_blend=alpha_blend,
                    detection_mode=det_mode)
                pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_labels))]
                m = calc_metrics([int(l) for l in q_labels], pred, cids)
                if phase == "p1":
                    name = f"AFPOD_p1_trace_a{alpha:.2f}"
                elif phase == "p2":
                    name = f"AFPOD_p2_lw_a{alpha:.2f}"
                elif phase == "p3a":
                    name = f"AFPOD_p3a_morph_a{alpha:.2f}_b{alpha_blend:.2f}"
                elif phase == "p3b":
                    name = f"AFPOD_p3b_dv_inter_a{alpha:.2f}"
                elif phase == "p3c":
                    name = f"AFPOD_p3c_dv_union_a{alpha:.2f}"
                else:
                    name = f"AFPOD_{phase}_a{alpha:.2f}"
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # Progress print every seed
        v = all_results["MB_kNN"]
        p2 = all_results["AFPOD_p2_lw_a0.10"]
        p3b = all_results.get("AFPOD_p3b_dv_inter_a0.10", None)
        p2_str = f"P2(LW,a=0.10): mF1={np.mean(p2['mf1']):.4f}, Eos={np.mean(p2['pc'][3]):.4f}"
        p3b_str = ""
        if p3b and p3b["mf1"]:
            p3b_str = f"  | P3b(dv_inter,a=0.10): mF1={np.mean(p3b['mf1']):.4f}, Eos={np.mean(p3b['pc'][3]):.4f}"
        print(f"   MB_kNN: mF1={np.mean(v['mf1']):.4f}, Eos={np.mean(v['pc'][3]):.4f}  | "
              f"{p2_str}{p3b_str}",
              flush=True)

    # Final report
    print(f"\n\n{'=' * 120}")
    print("FINAL RESULTS: Nested 5-fold CV × 5 seeds (25 total evaluations)")
    print(f"{'=' * 120}")
    h = f"{'Method':<32} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h)
    print("-" * 125)

    print_row("MB_kNN (baseline)", all_results["MB_kNN"], cids)
    print("--- Phase 1: Trace Shrinkage (fixed λ=0.3) ---")
    for alpha in [0.05, 0.10, 0.20]:
        print_row(f"AFPOD_p1_trace_a{alpha:.2f}", all_results[f"AFPOD_p1_trace_a{alpha:.2f}"], cids)
    print("--- Phase 2: Ledoit-Wolf adaptive ---")
    for alpha in [0.05, 0.10, 0.20]:
        print_row(f"AFPOD_p2_lw_a{alpha:.2f}", all_results[f"AFPOD_p2_lw_a{alpha:.2f}"], cids)
    print("--- Phase 3a: LW + Morph-Anchored (ablation) ---")
    for ab in [0.3, 0.5]:
        name = f"AFPOD_p3a_morph_a0.10_b{ab:.2f}"
        if name in all_results:
            print_row(name, all_results[name], cids)
    print("--- Phase 3b: LW + Dual-View Intersection ---")
    for alpha in [0.05, 0.10, 0.20]:
        name = f"AFPOD_p3b_dv_inter_a{alpha:.2f}"
        if name in all_results:
            print_row(name, all_results[name], cids)
    print("--- Phase 3c: LW + Dual-View Union ---")
    for alpha in [0.10]:
        name = f"AFPOD_p3c_dv_union_a{alpha:.2f}"
        if name in all_results:
            print_row(name, all_results[name], cids)

    # Improvements
    bl_mf1 = np.mean(all_results["MB_kNN"]["mf1"])
    bl_eos = np.mean(all_results["MB_kNN"]["pc"][3])
    print("\nImprovement over MB_kNN baseline:")
    for (method, shrink, alpha, alpha_blend, det_mode, phase) in configs:
        if phase == "p1":
            name = f"AFPOD_p1_trace_a{alpha:.2f}"
        elif phase == "p2":
            name = f"AFPOD_p2_lw_a{alpha:.2f}"
        elif phase == "p3a":
            name = f"AFPOD_p3a_morph_a{alpha:.2f}_b{alpha_blend:.2f}"
        elif phase == "p3b":
            name = f"AFPOD_p3b_dv_inter_a{alpha:.2f}"
        elif phase == "p3c":
            name = f"AFPOD_p3c_dv_union_a{alpha:.2f}"
        else:
            name = f"AFPOD_{phase}_a{alpha:.2f}"
        if name not in all_results:
            continue
        v = all_results[name]
        d_mf1 = np.mean(v["mf1"]) - bl_mf1
        d_eos = np.mean(v["pc"][3]) - bl_eos
        print(f"  {name}: ΔmF1 = {d_mf1:+.4f}, ΔEos F1 = {d_eos:+.4f}")


if __name__ == "__main__":
    main()
