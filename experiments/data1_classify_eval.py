#!/usr/bin/env python3
"""
Classification evaluation on data1_organized (7 classes).
Phase 1: Weight sweep for MB-kNN baseline
Phase 2: AFP-OD P3c (dual-view union) with alpha sweep
Protocol: Nested 5-fold CV × 5 seeds = 25 evaluations, 10-shot per class
"""
import sys, random
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {0: "CCEC", 1: "RBC", 2: "SEC", 3: "Eosinophil",
               4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


def load_cache(model, split):
    d = np.load(CACHE_DIR / f"data1_{model}_{split}.npz")
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
        pc[c] = {"p": pr, "r": rc, "f1": f1, "support": gp}
        f1s.append(f1)
    return {"acc": correct / total if total else 0, "mf1": float(np.mean(f1s)), "pc": pc}


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


# ==================== AFP-OD core (from afpod_classify.py) ====================

def _ledoit_wolf_shrinkage(X):
    n, d = X.shape
    if n < 2:
        return 1.0, np.zeros((d, d), dtype=np.float64), np.zeros((d, d), dtype=np.float64)
    X_c = X - X.mean(axis=0, keepdims=True)
    S = (X_c.T @ X_c) / n
    trace_S = float(np.trace(S))
    F = (trace_S / d) * np.eye(d, dtype=S.dtype)
    X2 = X_c ** 2
    pi_mat = (X2.T @ X2) / n - S ** 2
    pi_hat = float(pi_mat.sum())
    gamma_hat = float(((S - F) ** 2).sum())
    if gamma_hat < 1e-12 or n < 2:
        shrink = 0.0
    else:
        shrink = max(0.0, min(1.0, pi_hat / (gamma_hat * n)))
    return shrink, S, F


def fisher_direction(feats_i, feats_j, method="lw"):
    mu_i = feats_i.mean(axis=0)
    mu_j = feats_j.mean(axis=0)
    d = feats_i.shape[1]
    diff = (mu_i - mu_j).astype(np.float64)
    if method == "lw":
        if len(feats_i) > 1:
            si, Si, Fi = _ledoit_wolf_shrinkage(feats_i.astype(np.float64))
            Sigma_i = (1.0 - si) * Si + si * Fi
        else:
            Sigma_i = np.zeros((d, d), dtype=np.float64)
        if len(feats_j) > 1:
            sj, Sj, Fj = _ledoit_wolf_shrinkage(feats_j.astype(np.float64))
            Sigma_j = (1.0 - sj) * Sj + sj * Fj
        else:
            Sigma_j = np.zeros((d, d), dtype=np.float64)
        Sigma_reg = Sigma_i + Sigma_j + 1e-6 * np.eye(d, dtype=np.float64)
    else:
        raise ValueError(method)
    try:
        w = np.linalg.solve(Sigma_reg, diff)
    except np.linalg.LinAlgError:
        w = diff
    norm = float(np.linalg.norm(w))
    if norm < 1e-8:
        return np.zeros(d, dtype=np.float32)
    return (w / norm).astype(np.float32)


def morph_anchored_direction(feats_i, feats_j, morph_i, morph_j, alpha_blend=0.5):
    w_fisher_feat = fisher_direction(feats_i, feats_j)
    w_fisher_morph = fisher_direction(morph_i, morph_j)
    if np.linalg.norm(w_fisher_morph) < 1e-8:
        return w_fisher_feat
    all_morph = np.vstack([morph_i, morph_j]).astype(np.float64)
    all_feats = np.vstack([feats_i, feats_j]).astype(np.float64)
    morph_polarity = all_morph @ w_fisher_morph.astype(np.float64)
    morph_polarity_c = morph_polarity - morph_polarity.mean()
    feats_c = all_feats - all_feats.mean(0, keepdims=True)
    w_pls = feats_c.T @ morph_polarity_c
    pls_norm = float(np.linalg.norm(w_pls))
    if pls_norm < 1e-8:
        return w_fisher_feat
    w_pls = (w_pls / pls_norm).astype(np.float32)
    if float(np.dot(w_pls, w_fisher_feat)) < 0:
        w_pls = -w_pls
    w_blend = (1.0 - alpha_blend) * w_fisher_feat + alpha_blend * w_pls
    blend_norm = float(np.linalg.norm(w_blend))
    if blend_norm < 1e-8:
        return w_fisher_feat
    return (w_blend / blend_norm).astype(np.float32)


def _loo_knn_confusion_table(s_feats_per_class, cids, k=5, metric="cosine"):
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
        preds = np.zeros(N, dtype=int)
        for i in range(N):
            topk_idx = np.argsort(sim[i])[-k:]
            votes = all_labels[topk_idx]
            preds[i] = int(np.bincount(votes.astype(int), minlength=max(cids)+1).argmax())
    else:
        sq = (all_feats ** 2).sum(axis=1, keepdims=True)
        dist = sq + sq.T - 2.0 * (all_feats @ all_feats.T)
        np.fill_diagonal(dist, np.inf)
        preds = np.zeros(N, dtype=int)
        for i in range(N):
            topk_idx = np.argsort(dist[i])[:k]
            votes = all_labels[topk_idx]
            preds[i] = int(np.bincount(votes.astype(int), minlength=max(cids)+1).argmax())
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


def find_confusion_pairs_dualview(s_feats_per_class, s_morph_per_class, cids,
                                  feat_thr=0.15, morph_thr=0.15, k=5, mode="union"):
    feat_rates = _loo_knn_confusion_table(s_feats_per_class, cids, k=k, metric="cosine")
    sm_all = np.concatenate([s_morph_per_class[c] for c in cids
                             if len(s_morph_per_class[c]) > 0], axis=0)
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    morph_norm = {c: (s_morph_per_class[c] - gm) / gs for c in cids}
    morph_rates = _loo_knn_confusion_table(morph_norm, cids, k=k, metric="euclidean")
    pairs_out = []
    for (ci, cj), feat_r in feat_rates.items():
        morph_r = morph_rates.get((ci, cj), 0.0)
        feat_ok = feat_r >= feat_thr
        morph_ok = morph_r >= morph_thr
        if mode == "union":
            keep = feat_ok or morph_ok
        elif mode == "intersection":
            keep = feat_ok and morph_ok
        else:
            keep = feat_ok
        if keep:
            strength = (feat_r + morph_r) / 2.0
            pairs_out.append((ci, cj, strength))
    pairs_out.sort(key=lambda x: -x[2])
    return [(ci, cj) for ci, cj, _ in pairs_out]


def amplify_separation(support_per_class, confusion_pairs, alpha=0.10,
                       morph_per_class=None, alpha_blend=0.0):
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
            mi, mj = morph_per_class[ci], morph_per_class[cj]
            if len(mi) == 0 or len(mj) == 0:
                w = fisher_direction(feats_i, feats_j)
            else:
                w = morph_anchored_direction(feats_i, feats_j, mi, mj, alpha_blend)
        else:
            w = fisher_direction(feats_i, feats_j)
        if np.linalg.norm(w) < 1e-8:
            continue
        modified[ci] = modified[ci] + alpha * w
        modified[cj] = modified[cj] - alpha * w
    for c in modified:
        norms = np.linalg.norm(modified[c], axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        modified[c] = (modified[c] / norms).astype(np.float32)
    return modified


def mb_knn_classify(q_bc, q_ph, q_dn, q_morph,
                    s_bc, s_ph, s_dn, s_morph, cids,
                    bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7):
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


def afpod_classify(q_bc, q_ph, q_dn, q_morph,
                   s_bc, s_ph, s_dn, s_morph, cids,
                   bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7,
                   alpha=0.10, conf_thresh=0.15, alpha_blend=0.0,
                   detection_mode="dualview_union"):
    if "dualview" in detection_mode:
        mode = "union" if "union" in detection_mode else "intersection"
        confusion_pairs = find_confusion_pairs_dualview(
            s_bc, s_morph, cids, feat_thr=conf_thresh, morph_thr=conf_thresh, mode=mode)
    else:
        rates = _loo_knn_confusion_table(s_bc, cids, k=5, metric="cosine")
        confusion_pairs = [(ci, cj) for (ci, cj), r in sorted(rates.items(), key=lambda x: -x[1]) if r >= conf_thresh]

    s_bc_mod = amplify_separation(s_bc, confusion_pairs, alpha=alpha,
                                  morph_per_class=s_morph, alpha_blend=alpha_blend)
    s_ph_mod = amplify_separation(s_ph, confusion_pairs, alpha=alpha,
                                  morph_per_class=s_morph, alpha_blend=alpha_blend)
    s_dn_mod = amplify_separation(s_dn, confusion_pairs, alpha=alpha,
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


def run_nested_cv(name, classify_fn, bc_t, ph_t, dn_t, mt, lt,
                  bc_v, ph_v, dn_v, mv, lv, cids, **kwargs):
    """Run nested 5-fold CV × 5 seeds."""
    all_acc, all_mf1 = [], []
    all_pc = defaultdict(list)

    for seed in SEEDS:
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

            scores = classify_fn(q_bc, q_ph, q_dn, q_morph, sbc, sph, sdn, sm, cids, **kwargs)
            if isinstance(scores, tuple):
                scores = scores[0]
            pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_labels))]
            m = calc_metrics([int(l) for l in q_labels], pred, cids)
            all_acc.append(m["acc"])
            all_mf1.append(m["mf1"])
            for c in cids:
                all_pc[c].append(m["pc"][c]["f1"])

    return {"name": name, "acc": np.mean(all_acc), "acc_std": np.std(all_acc),
            "mf1": np.mean(all_mf1), "mf1_std": np.std(all_mf1),
            "pc": {c: (np.mean(all_pc[c]), np.std(all_pc[c])) for c in cids}}


def print_result(r, cids):
    pc_str = " ".join(f"{r['pc'][c][0]:>6.4f}" for c in cids)
    print(f"  {r['name']:<45} Acc={r['acc']:.4f}±{r['acc_std']:.3f}  "
          f"mF1={r['mf1']:.4f}±{r['mf1_std']:.3f}  [{pc_str}]")


def main():
    print("=" * 110)
    print("Classification Evaluation on data1_organized (7 classes)")
    print("Protocol: Nested 5-fold CV × 5 seeds = 25 evaluations, 10-shot")
    print("=" * 110)

    print("\nLoading feature caches...")
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")
    cids = sorted(CLASS_NAMES.keys())

    print(f"BC train={bc_t.shape} val={bc_v.shape}")
    print(f"PH train={ph_t.shape} val={ph_v.shape}")
    print(f"DN train={dn_t.shape} val={dn_v.shape}")
    print(f"Morph train={mt.shape} val={mv.shape}")
    print(f"Classes: {[CLASS_NAMES[c] for c in cids]}")
    for c in cids:
        n_t = np.sum(lt == c)
        n_v = np.sum(lv == c)
        print(f"  {CLASS_NAMES[c]}: train={n_t} val={n_v}")

    results = []

    # ============================================================
    # Phase 1: Weight sweep for MB-kNN
    # ============================================================
    print(f"\n{'='*90}")
    print("Phase 1: MB-kNN Weight Sweep")
    print(f"{'='*90}")

    weight_configs = [
        (0.42, 0.18, 0.07, 0.33, "data2-optimal"),
        (0.35, 0.20, 0.10, 0.35, "balanced-A"),
        (0.40, 0.15, 0.10, 0.35, "balanced-B"),
        (0.30, 0.25, 0.10, 0.35, "phikon-heavy"),
        (0.45, 0.15, 0.05, 0.35, "bc-heavy"),
        (0.35, 0.15, 0.10, 0.40, "morph-heavy"),
        (0.30, 0.20, 0.10, 0.40, "morph-heavy-B"),
        (0.40, 0.20, 0.10, 0.30, "visual-heavy"),
        (0.35, 0.25, 0.05, 0.35, "bc+ph-focus"),
        (0.30, 0.15, 0.05, 0.50, "morph-dominant"),
    ]

    for bw, pw, dw, mw, tag in weight_configs:
        r = run_nested_cv(f"MB-kNN bw={bw} pw={pw} dw={dw} mw={mw} ({tag})",
                          mb_knn_classify, bc_t, ph_t, dn_t, mt, lt,
                          bc_v, ph_v, dn_v, mv, lv, cids,
                          bw=bw, pw=pw, dw=dw, mw=mw, k=7)
        results.append(r)
        print_result(r, cids)

    # Also sweep k
    print("\nk sweep (using data2-optimal weights):")
    for k in [3, 5, 7, 9, 11]:
        r = run_nested_cv(f"MB-kNN k={k} (data2-weights)",
                          mb_knn_classify, bc_t, ph_t, dn_t, mt, lt,
                          bc_v, ph_v, dn_v, mv, lv, cids,
                          bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=k)
        results.append(r)
        print_result(r, cids)

    # Find best MB-kNN
    best_mbknn = max(results, key=lambda x: x["mf1"])
    print(f"\nBest MB-kNN: {best_mbknn['name']} → mF1={best_mbknn['mf1']:.4f}")

    # ============================================================
    # Phase 2: AFP-OD sweep
    # ============================================================
    print(f"\n{'='*90}")
    print("Phase 2: AFP-OD P3c (LW + Dual-View Union)")
    print(f"{'='*90}")

    # Use top weight configs for AFP-OD
    top_weight_configs = sorted(results[:len(weight_configs)], key=lambda x: -x["mf1"])[:3]

    afpod_results = []
    for base_r in top_weight_configs:
        name_parts = base_r["name"].split("(")[1].rstrip(")")
        # Extract weights from name
        parts = base_r["name"].split()
        bw = float(parts[1].split("=")[1])
        pw = float(parts[2].split("=")[1])
        dw = float(parts[3].split("=")[1])
        mw = float(parts[4].split("=")[1])

        for alpha in [0.05, 0.10, 0.15, 0.20]:
            for conf_thr in [0.10, 0.15, 0.20]:
                r = run_nested_cv(
                    f"AFP-OD a={alpha} ct={conf_thr} ({name_parts})",
                    afpod_classify, bc_t, ph_t, dn_t, mt, lt,
                    bc_v, ph_v, dn_v, mv, lv, cids,
                    bw=bw, pw=pw, dw=dw, mw=mw, k=7,
                    alpha=alpha, conf_thresh=conf_thr,
                    detection_mode="dualview_union")
                afpod_results.append(r)
                print_result(r, cids)

    results.extend(afpod_results)

    # Also test AFP-OD with intersection mode for comparison
    print("\nAFP-OD intersection mode (best weights):")
    bw, pw, dw, mw = 0.42, 0.18, 0.07, 0.33
    for alpha in [0.05, 0.10, 0.20]:
        r = run_nested_cv(
            f"AFP-OD-inter a={alpha} (data2-opt)",
            afpod_classify, bc_t, ph_t, dn_t, mt, lt,
            bc_v, ph_v, dn_v, mv, lv, cids,
            bw=bw, pw=pw, dw=dw, mw=mw, k=7,
            alpha=alpha, conf_thresh=0.15,
            detection_mode="dualview_intersection")
        results.append(r)
        print_result(r, cids)

    # ============================================================
    # Phase 3: Single backbone baselines
    # ============================================================
    print(f"\n{'='*90}")
    print("Phase 3: Single Backbone Baselines")
    print(f"{'='*90}")

    def single_knn(q_bc, q_ph, q_dn, q_morph, s_bc, s_ph, s_dn, s_morph, cids,
                   backbone="bc", k=7):
        K = len(cids)
        scores = np.zeros((len(q_bc), K), dtype=np.float32)
        for i in range(len(q_bc)):
            for ki, c in enumerate(cids):
                if backbone == "bc":
                    vs = s_bc[c] @ q_bc[i]
                elif backbone == "ph":
                    vs = s_ph[c] @ q_ph[i]
                elif backbone == "dn":
                    vs = s_dn[c] @ q_dn[i]
                ncls = len(vs)
                if ncls == 0:
                    scores[i, ki] = -np.inf
                    continue
                kk = min(k, ncls)
                scores[i, ki] = float(np.sort(vs)[::-1][:kk].mean())
        return scores

    for bb_name, bb_key in [("BiomedCLIP", "bc"), ("Phikon-v2", "ph"), ("DINOv2-S", "dn")]:
        r = run_nested_cv(f"kNN-{bb_name} k=7",
                          single_knn, bc_t, ph_t, dn_t, mt, lt,
                          bc_v, ph_v, dn_v, mv, lv, cids,
                          backbone=bb_key, k=7)
        results.append(r)
        print_result(r, cids)

    # ============================================================
    # Final Summary
    # ============================================================
    print(f"\n\n{'='*120}")
    print("FINAL RANKING (sorted by mF1)")
    print(f"{'='*120}")
    header_cls = " ".join(f"{CLASS_NAMES[c][:4]:>6}" for c in cids)
    print(f"{'Method':<48} {'Acc':>7} {'mF1':>7} {header_cls}  {'As':>5} {'Fs':>5}")
    print("-" * 120)

    results.sort(key=lambda x: -x["mf1"])
    for r in results:
        pc_str = " ".join(f"{r['pc'][c][0]:>6.4f}" for c in cids)
        print(f"{r['name']:<48} {r['acc']:>7.4f} {r['mf1']:>7.4f} {pc_str}  "
              f"{r['acc_std']:>5.3f} {r['mf1_std']:>5.3f}")

    best = results[0]
    best_mbknn_final = max([r for r in results if "MB-kNN" in r["name"]], key=lambda x: x["mf1"])
    best_afpod_final = max([r for r in results if "AFP-OD" in r["name"]], key=lambda x: x["mf1"])

    print(f"\nBest MB-kNN:  {best_mbknn_final['name']} → mF1={best_mbknn_final['mf1']:.4f}")
    print(f"Best AFP-OD:  {best_afpod_final['name']} → mF1={best_afpod_final['mf1']:.4f}")
    print(f"Best overall: {best['name']} → mF1={best['mf1']:.4f}")
    if best_afpod_final["mf1"] > best_mbknn_final["mf1"]:
        imp = best_afpod_final["mf1"] - best_mbknn_final["mf1"]
        print(f"AFP-OD improvement over MB-kNN: ΔmF1 = {imp:+.4f} ({imp/best_mbknn_final['mf1']*100:+.1f}%)")


if __name__ == "__main__":
    main()
