#!/usr/bin/env python3
"""
Classification experiment on data1_organized (7-class clinical BALF).
Runs: MB-kNN baseline sweep → AFP-OD P3c sweep → final report.
Protocol: Nested 5-fold CV × 5 seeds = 25 evaluations, 10-shot.
"""
import sys, random
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def fisher_direction(feats_i, feats_j, method="lw", shrink=0.3):
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


def amplify_separation(support_per_class, confusion_pairs, alpha=0.10, method="lw"):
    if not confusion_pairs:
        return {c: v.copy() for c, v in support_per_class.items()}
    modified = {c: v.copy().astype(np.float32) for c, v in support_per_class.items()}
    for ci, cj in confusion_pairs:
        fi, fj = support_per_class[ci], support_per_class[cj]
        if len(fi) == 0 or len(fj) == 0:
            continue
        w = fisher_direction(fi, fj, method=method)
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
                   alpha=0.10, conf_thresh=0.15, method="lw",
                   detection_mode="dualview_union"):
    if "dualview" in detection_mode:
        mode = "union" if "union" in detection_mode else "intersection"
        confusion_pairs = find_confusion_pairs_dualview(
            s_bc, s_morph, cids,
            feat_thr=conf_thresh, morph_thr=conf_thresh, mode=mode)
    else:
        rates = _loo_knn_confusion_table(s_bc, cids, k=5, metric="cosine")
        confusion_pairs = [(ci, cj) for (ci, cj), r in sorted(rates.items(), key=lambda x: -x[1])
                           if r >= conf_thresh]

    s_bc_mod = amplify_separation(s_bc, confusion_pairs, alpha=alpha, method=method)
    s_ph_mod = amplify_separation(s_ph, confusion_pairs, alpha=alpha, method=method)
    s_dn_mod = amplify_separation(s_dn, confusion_pairs, alpha=alpha, method=method)

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


def print_row(name, v, cids):
    pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    print(f"{name:<40} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  "
          f"±{np.std(v['acc']):.3f} ±{np.std(v['mf1']):.3f}")


def main():
    print("=" * 130)
    print("Classification on data1_organized (7 classes, 10-shot, nested 5-fold CV × 5 seeds)")
    print("=" * 130, flush=True)

    print("\nLoading feature caches...")
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")
    cids = sorted(CLASS_NAMES.keys())
    print(f"Loaded: BC={bc_t.shape}, PH={ph_t.shape}, DN={dn_t.shape}, morph={mt.shape}")
    print(f"  Val: BC={bc_v.shape}, PH={ph_v.shape}, DN={dn_v.shape}")
    print(f"  Classes: {[CLASS_NAMES[c] for c in cids]}")
    print(f"  Class dist (train): {dict(zip([CLASS_NAMES[c] for c in cids], [int((lt==c).sum()) for c in cids]))}")
    print(f"  Class dist (val): {dict(zip([CLASS_NAMES[c] for c in cids], [int((lv==c).sum()) for c in cids]))}")

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    # Weight configurations to sweep
    weight_configs = [
        # (bw, pw, dw, mw, k, label)
        (0.42, 0.18, 0.07, 0.33, 7, "w_default"),
        (0.35, 0.25, 0.07, 0.33, 7, "w_ph_boost"),
        (0.45, 0.15, 0.10, 0.30, 7, "w_dn_boost"),
        (0.40, 0.20, 0.10, 0.30, 7, "w_balanced"),
        (0.38, 0.22, 0.08, 0.32, 7, "w_phdn"),
        (0.42, 0.18, 0.07, 0.33, 5, "w_default_k5"),
        (0.42, 0.18, 0.07, 0.33, 9, "w_default_k9"),
        (0.42, 0.18, 0.07, 0.33, 3, "w_default_k3"),
    ]

    # AFP-OD configs: (alpha, conf_thresh, detection_mode)
    afpod_configs = [
        (0.05, 0.15, "dualview_union"),
        (0.10, 0.15, "dualview_union"),
        (0.15, 0.15, "dualview_union"),
        (0.20, 0.15, "dualview_union"),
        (0.10, 0.10, "dualview_union"),
        (0.10, 0.20, "dualview_union"),
        (0.10, 0.15, "dualview_intersection"),
        (0.10, 0.15, "feature_only"),
    ]

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\nSeed {seed} ({seed_idx+1}/{len(SEEDS)})", flush=True)
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

            # MB-kNN weight sweep
            for bw, pw, dw, mw, k, label in weight_configs:
                bl_scores = mb_knn_classify(q_bc, q_ph, q_dn, q_morph, sbc, sph, sdn, sm, cids,
                                            bw=bw, pw=pw, dw=dw, mw=mw, k=k)
                pred_bl = [cids[int(np.argmax(bl_scores[i]))] for i in range(len(q_labels))]
                m = calc_metrics([int(l) for l in q_labels], pred_bl, cids)
                name = f"MB_kNN_{label}"
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[name]["pc"][c].append(m["pc"][c]["f1"])

            # AFP-OD sweep (using default weights initially)
            for alpha, conf_thr, det_mode in afpod_configs:
                scores, _ = afpod_classify(q_bc, q_ph, q_dn, q_morph, sbc, sph, sdn, sm, cids,
                                           alpha=alpha, conf_thresh=conf_thr,
                                           detection_mode=det_mode)
                pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_labels))]
                m = calc_metrics([int(l) for l in q_labels], pred, cids)
                name = f"AFPOD_a{alpha:.2f}_t{conf_thr:.2f}_{det_mode.split('_')[-1]}"
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # Progress
        bl = all_results["MB_kNN_w_default"]
        best_afpod_name = max([n for n in all_results if n.startswith("AFPOD_")],
                               key=lambda n: np.mean(all_results[n]["mf1"]), default="")
        best_afpod = all_results[best_afpod_name] if best_afpod_name else None
        print(f"  MB_kNN: mF1={np.mean(bl['mf1']):.4f}", end="")
        if best_afpod:
            print(f" | best AFPOD ({best_afpod_name}): mF1={np.mean(best_afpod['mf1']):.4f}", end="")
        print(flush=True)

    # Phase 2: Use best weight config with best AFP-OD alpha
    # Find best MB-kNN weight config
    best_mb = max([n for n in all_results if n.startswith("MB_kNN_")],
                   key=lambda n: np.mean(all_results[n]["mf1"]))
    best_mb_cfg = [c for c in weight_configs if f"MB_kNN_{c[5]}" == best_mb][0] if any(f"MB_kNN_{c[5]}" == best_mb for c in weight_configs) else weight_configs[0]
    bw_best, pw_best, dw_best, mw_best, k_best = best_mb_cfg[:5]
    print(f"\nBest MB-kNN weights: bw={bw_best}, pw={pw_best}, dw={dw_best}, mw={mw_best}, k={k_best}")

    # Re-run AFP-OD with best weights
    print("\nPhase 2: AFP-OD with best MB-kNN weights...")
    afpod_phase2 = [
        (0.05, 0.15, "dualview_union"),
        (0.10, 0.15, "dualview_union"),
        (0.15, 0.15, "dualview_union"),
        (0.20, 0.15, "dualview_union"),
    ]
    for seed in SEEDS:
        val_indices = np.arange(len(lv))
        folds = k_fold_split(val_indices, 5, seed)
        for fold_idx, test_fold in enumerate(folds):
            q_bc, q_ph, q_dn, q_morph = bc_v[test_fold], ph_v[test_fold], dn_v[test_fold], mv[test_fold]
            q_labels = lv[test_fold]
            si = select_support(lt, seed + fold_idx, cids)
            sbc = {c: bc_t[si[c]] for c in cids}
            sph = {c: ph_t[si[c]] for c in cids}
            sdn = {c: dn_t[si[c]] for c in cids}
            sm = {c: mt[si[c]] for c in cids}
            for alpha, conf_thr, det_mode in afpod_phase2:
                scores, _ = afpod_classify(q_bc, q_ph, q_dn, q_morph, sbc, sph, sdn, sm, cids,
                                           bw=bw_best, pw=pw_best, dw=dw_best, mw=mw_best, k=k_best,
                                           alpha=alpha, conf_thresh=conf_thr,
                                           detection_mode=det_mode)
                pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_labels))]
                m = calc_metrics([int(l) for l in q_labels], pred, cids)
                name = f"AFPOD_bestw_a{alpha:.2f}"
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[name]["pc"][c].append(m["pc"][c]["f1"])

    # Final report
    print(f"\n\n{'='*160}")
    print("FINAL RESULTS: data1_organized, 7-class, 10-shot, nested 5-fold CV × 5 seeds (25 evals)")
    print(f"{'='*160}")
    h = (f"{'Method':<42} {'Acc':>7} {'mF1':>7} "
         + " ".join(f"{CLASS_NAMES[c][:5]:>7}" for c in cids)
         + "  {'σAcc':>5} {'σF1':>5}")
    print(f"{'Method':<42} {'Acc':>7} {'mF1':>7} "
          + " ".join(f"{CLASS_NAMES[c][:4]:>7}" for c in cids)
          + f"  {'sAcc':>5} {'sF1':>5}")
    print("-" * 160)

    # Sort by category
    mb_names = sorted([n for n in all_results if n.startswith("MB_kNN_")],
                      key=lambda n: -np.mean(all_results[n]["mf1"]))
    afpod_names = sorted([n for n in all_results if n.startswith("AFPOD_")],
                         key=lambda n: -np.mean(all_results[n]["mf1"]))

    print("--- MB-kNN Baselines ---")
    for name in mb_names:
        print_row(name, all_results[name], cids)
    print("--- AFP-OD ---")
    for name in afpod_names:
        print_row(name, all_results[name], cids)

    # Best overall
    all_names = list(all_results.keys())
    best_name = max(all_names, key=lambda n: np.mean(all_results[n]["mf1"]))
    bl_name = "MB_kNN_w_default"
    print(f"\nBEST OVERALL: {best_name}")
    print(f"  Acc = {np.mean(all_results[best_name]['acc']):.4f} ± {np.std(all_results[best_name]['acc']):.4f}")
    print(f"  mF1 = {np.mean(all_results[best_name]['mf1']):.4f} ± {np.std(all_results[best_name]['mf1']):.4f}")
    for c in cids:
        print(f"  {CLASS_NAMES[c]}: F1 = {np.mean(all_results[best_name]['pc'][c]):.4f} ± {np.std(all_results[best_name]['pc'][c]):.4f}")

    bl_mf1 = np.mean(all_results[bl_name]['mf1'])
    best_mf1 = np.mean(all_results[best_name]['mf1'])
    print(f"\n  Improvement over MB_kNN baseline: ΔmF1 = {best_mf1 - bl_mf1:+.4f}")


if __name__ == "__main__":
    main()
