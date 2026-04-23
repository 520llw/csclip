#!/usr/bin/env python3
"""
Transductive Few-Shot Classification via EM-Dirichlet on Multi-Backbone Features.

Based on: Martin et al., "Transductive Zero-Shot and Few-Shot CLIP", CVPR 2024.

Core idea: Instead of classifying each query sample independently,
treat the entire query batch as a joint optimization problem.
Model per-class feature distributions with Dirichlet distributions,
then alternate between updating distribution params and soft assignments
via Block Majorization-Minimization (BMM).

Adaptations for BALF:
  - Use pre-extracted features from 3 backbones (BiomedCLIP, Phikon-v2, DINOv2)
  - Construct probability features z_n via softmax(T * cosine_sim) per backbone
  - Concatenate or fuse probability features before Dirichlet modeling
  - 10-shot support samples initialize class prototypes
"""
import sys, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import digamma, gammaln, softmax

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


def metrics(gt, pred, cids):
    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pred))
    f1s = []
    pc = {}
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


# ==================== Probability Feature Construction ====================

def build_probability_features(query_feats, support_feats_per_class, cids, temperature=30.0):
    """
    Construct probability features z_n = softmax(T * cos(query, prototype_k)).
    For each backbone, compute per-class cosine similarity and apply softmax.
    """
    K = len(cids)
    prototypes = np.zeros((K, query_feats.shape[1]))
    for ki, c in enumerate(cids):
        prototypes[ki] = support_feats_per_class[c].mean(axis=0)
        prototypes[ki] /= np.linalg.norm(prototypes[ki]) + 1e-8

    q_norm = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
    logits = temperature * (q_norm @ prototypes.T)  # (N, K)
    z = softmax(logits, axis=1)  # (N, K)
    return z, prototypes


def build_support_probability_features(support_feats_per_class, prototypes, cids, temperature=30.0):
    """Build probability features for support samples."""
    K = len(cids)
    all_z, all_labels = [], []
    for ki, c in enumerate(cids):
        s_norm = support_feats_per_class[c] / (np.linalg.norm(support_feats_per_class[c], axis=1, keepdims=True) + 1e-8)
        logits = temperature * (s_norm @ prototypes.T)
        z = softmax(logits, axis=1)
        all_z.append(z)
        all_labels.extend([ki] * len(z))
    return np.concatenate(all_z, axis=0), np.array(all_labels)


# ==================== EM-Dirichlet Core ====================

def dirichlet_log_likelihood(z, alpha):
    """Log-likelihood of z under Dirichlet(alpha)."""
    return (gammaln(alpha.sum()) - gammaln(alpha).sum() +
            ((alpha - 1) * np.log(z + 1e-30)).sum(axis=-1))


def em_dirichlet(z_query, z_support, y_support, K, n_iter=20, lambda_reg=1.0):
    """
    EM-Dirichlet transductive inference.

    z_query:   (N_q, K) probability features of query samples
    z_support: (N_s, K) probability features of support samples
    y_support: (N_s,)   class indices (0..K-1) for support
    K:         number of classes
    n_iter:    BMM iterations
    lambda_reg: regularization for class balance (higher = more balanced)

    Returns: soft assignments u (N_q, K) and final predictions
    """
    N_q = len(z_query)
    N_s = len(z_support)

    alpha = np.ones((K, K)) * 1.0  # Dirichlet params, (K_classes, K_dims)
    for k in range(K):
        mask = y_support == k
        if mask.sum() > 0:
            z_k = z_support[mask]
            mean_z = z_k.mean(axis=0) + 1e-6
            mean_z /= mean_z.sum()
            s = 10.0
            alpha[k] = mean_z * s

    pi = np.ones(K) / K  # class prior

    u = np.ones((N_q, K)) / K  # soft assignments

    for it in range(n_iter):
        # E-step: update alpha from both support and weighted query
        for k in range(K):
            s_mask = y_support == k
            n_s_k = s_mask.sum()
            w_q_k = u[:, k].sum()

            if n_s_k + w_q_k < 0.5:
                continue

            weighted_sum = np.zeros(K)
            if n_s_k > 0:
                weighted_sum += np.log(z_support[s_mask] + 1e-30).sum(axis=0)
            weighted_sum += (u[:, k:k+1] * np.log(z_query + 1e-30)).sum(axis=0)

            total_weight = n_s_k + w_q_k
            mean_log_z = weighted_sum / total_weight

            alpha_k = np.exp(mean_log_z) + 0.5
            alpha_k = np.clip(alpha_k, 0.1, 100.0)
            alpha[k] = alpha_k

        # M-step: update pi (class proportions)
        for k in range(K):
            s_k = (y_support == k).sum()
            pi[k] = (s_k + u[:, k].sum()) / (N_s + N_q)
        pi = np.clip(pi, 1e-6, None)
        pi /= pi.sum()

        # Assignment update: u_n = softmax(log p(z_n | alpha_k) + lambda * log pi_k)
        log_like = np.zeros((N_q, K))
        for k in range(K):
            log_like[:, k] = dirichlet_log_likelihood(z_query, alpha[k])

        log_prior = lambda_reg * np.log(pi + 1e-30)
        logits = log_like + log_prior[np.newaxis, :]
        u = softmax(logits, axis=1)

    return u


# ==================== Multi-Backbone Fusion ====================

def fuse_probability_features(z_list, weights=None):
    """Fuse probability features from multiple backbones via weighted geometric mean."""
    if weights is None:
        weights = [1.0 / len(z_list)] * len(z_list)

    log_z = sum(w * np.log(z + 1e-30) for z, w in zip(z_list, weights))
    z_fused = np.exp(log_z)
    z_fused /= z_fused.sum(axis=1, keepdims=True)
    return z_fused


def transductive_classify(q_bc, q_ph, q_dn, q_morph, q_labels,
                          s_bc, s_ph, s_dn, s_morph, cids,
                          temperature=30.0, n_iter=20, lambda_reg=1.0,
                          backbone_weights=None, use_morph=True,
                          morph_weight=0.3):
    """
    Full transductive classification pipeline.
    1. Build probability features per backbone using support prototypes
    2. Fuse probability features
    3. Run EM-Dirichlet transductive inference
    4. Optionally incorporate morphology via post-hoc rescoring
    """
    K = len(cids)

    z_bc_q, proto_bc = build_probability_features(q_bc, s_bc, cids, temperature)
    z_ph_q, proto_ph = build_probability_features(q_ph, s_ph, cids, temperature)
    z_dn_q, proto_dn = build_probability_features(q_dn, s_dn, cids, temperature)

    z_bc_s, y_s = build_support_probability_features(s_bc, proto_bc, cids, temperature)
    z_ph_s, _ = build_support_probability_features(s_ph, proto_ph, cids, temperature)
    z_dn_s, _ = build_support_probability_features(s_dn, proto_dn, cids, temperature)

    if backbone_weights is None:
        backbone_weights = [0.50, 0.30, 0.20]  # BC, PH, DN

    z_q = fuse_probability_features([z_bc_q, z_ph_q, z_dn_q], backbone_weights)
    z_s = fuse_probability_features([z_bc_s, z_ph_s, z_dn_s], backbone_weights)

    u = em_dirichlet(z_q, z_s, y_s, K, n_iter=n_iter, lambda_reg=lambda_reg)

    if use_morph:
        sm_all = np.concatenate([s_morph[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

        morph_scores = np.zeros((len(q_labels), K))
        for ki, c in enumerate(cids):
            snm = (s_morph[c] - gm) / gs
            for i in range(len(q_labels)):
                qm = (q_morph[i] - gm) / gs
                md = np.linalg.norm(qm - snm, axis=1)
                morph_scores[i, ki] = np.mean(1.0 / (1.0 + np.sort(md)[:5]))

        morph_probs = softmax(morph_scores * 5.0, axis=1)
        u_final = (1.0 - morph_weight) * u + morph_weight * morph_probs
    else:
        u_final = u

    pred_indices = np.argmax(u_final, axis=1)
    pred = [cids[pi] for pi in pred_indices]
    gt = [int(l) for l in q_labels]

    return metrics(gt, pred, cids)


# ==================== Transductive + ATD Hybrid ====================

def transductive_atd_hybrid(q_bc, q_ph, q_dn, q_morph, q_labels,
                            s_bc0, s_ph0, s_dn0, s_morph0, cids,
                            temperature=30.0, n_iter_td=15, lambda_reg=1.0,
                            backbone_weights=None, morph_weight=0.3,
                            atd_rounds=2, top_k_pseudo=5, conf_thr=0.85):
    """
    Hybrid: EM-Dirichlet transduction + ATD pseudo-label augmentation.
    1. Run EM-Dirichlet to get soft assignments
    2. Select high-confidence pseudo-labels and add to support
    3. Re-run EM-Dirichlet with augmented support
    """
    K = len(cids)
    sb = {c: s_bc0[c].copy() for c in cids}
    sp = {c: s_ph0[c].copy() for c in cids}
    sd = {c: s_dn0[c].copy() for c in cids}
    sm = {c: s_morph0[c].copy() for c in cids}

    if backbone_weights is None:
        backbone_weights = [0.50, 0.30, 0.20]

    best_u = None
    for rnd in range(atd_rounds):
        z_bc_q, proto_bc = build_probability_features(q_bc, sb, cids, temperature)
        z_ph_q, proto_ph = build_probability_features(q_ph, sp, cids, temperature)
        z_dn_q, proto_dn = build_probability_features(q_dn, sd, cids, temperature)

        z_bc_s, y_s = build_support_probability_features(sb, proto_bc, cids, temperature)
        z_ph_s, _ = build_support_probability_features(sp, proto_ph, cids, temperature)
        z_dn_s, _ = build_support_probability_features(sd, proto_dn, cids, temperature)

        z_q = fuse_probability_features([z_bc_q, z_ph_q, z_dn_q], backbone_weights)
        z_s = fuse_probability_features([z_bc_s, z_ph_s, z_dn_s], backbone_weights)

        u = em_dirichlet(z_q, z_s, y_s, K, n_iter=n_iter_td, lambda_reg=lambda_reg)
        best_u = u

        if rnd < atd_rounds - 1:
            for ki, c in enumerate(cids):
                conf = u[:, ki]
                high_conf = np.where(conf > conf_thr)[0]
                if len(high_conf) == 0:
                    continue

                proto_c = sb[c].mean(0)
                dists = np.array([np.linalg.norm(q_bc[idx] - proto_c) for idx in high_conf])
                diversity = conf[high_conf] * (1.0 + 0.3 * dists / (dists.mean() + 1e-8))
                selected = high_conf[np.argsort(diversity)[::-1][:top_k_pseudo]]

                sb[c] = np.concatenate([s_bc0[c], q_bc[selected] * 0.5])
                sp[c] = np.concatenate([s_ph0[c], q_ph[selected] * 0.5])
                sd[c] = np.concatenate([s_dn0[c], q_dn[selected] * 0.5])
                sm[c] = np.concatenate([s_morph0[c], q_morph[selected]])

    if morph_weight > 0:
        sm_all = np.concatenate([sm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        morph_scores = np.zeros((len(q_labels), K))
        for ki, c in enumerate(cids):
            snm = (sm[c] - gm) / gs
            for i in range(len(q_labels)):
                qm = (q_morph[i] - gm) / gs
                md = np.linalg.norm(qm - snm, axis=1)
                morph_scores[i, ki] = np.mean(1.0 / (1.0 + np.sort(md)[:5]))
        morph_probs = softmax(morph_scores * 5.0, axis=1)
        u_final = (1.0 - morph_weight) * best_u + morph_weight * morph_probs
    else:
        u_final = best_u

    pred_indices = np.argmax(u_final, axis=1)
    pred = [cids[pi] for pi in pred_indices]
    gt = [int(l) for l in q_labels]

    return metrics(gt, pred, cids)


def print_row(name, v, cids):
    pc = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    print(f"{name:<55} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc}  "
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
        ("SADC_ATD_baseline",           "sadc"),
        ("TD_basic_T30",                dict(temperature=30.0, n_iter=20, lambda_reg=1.0, use_morph=False)),
        ("TD_basic_T20",                dict(temperature=20.0, n_iter=20, lambda_reg=1.0, use_morph=False)),
        ("TD_basic_T50",                dict(temperature=50.0, n_iter=20, lambda_reg=1.0, use_morph=False)),
        ("TD_morph_T30_mw03",           dict(temperature=30.0, n_iter=20, lambda_reg=1.0, use_morph=True, morph_weight=0.3)),
        ("TD_morph_T30_mw05",           dict(temperature=30.0, n_iter=20, lambda_reg=1.0, use_morph=True, morph_weight=0.5)),
        ("TD_morph_T30_mw02",           dict(temperature=30.0, n_iter=20, lambda_reg=1.0, use_morph=True, morph_weight=0.2)),
        ("TD_bw_50_30_20_T30",          dict(temperature=30.0, n_iter=20, lambda_reg=1.0, use_morph=True, morph_weight=0.3,
                                              backbone_weights=[0.50, 0.30, 0.20])),
        ("TD_bw_42_18_07_T30",          dict(temperature=30.0, n_iter=20, lambda_reg=1.0, use_morph=True, morph_weight=0.3,
                                              backbone_weights=[0.42, 0.18, 0.07])),
        ("TD_lam05_T30",                dict(temperature=30.0, n_iter=20, lambda_reg=0.5, use_morph=True, morph_weight=0.3)),
        ("TD_lam20_T30",                dict(temperature=30.0, n_iter=20, lambda_reg=2.0, use_morph=True, morph_weight=0.3)),
        ("TD_lam50_T30",                dict(temperature=30.0, n_iter=20, lambda_reg=5.0, use_morph=True, morph_weight=0.3)),
        # Hybrid: Transductive + ATD
        ("TD_ATD_hybrid_T30",           dict(mode="hybrid", temperature=30.0, n_iter_td=15, lambda_reg=1.0,
                                              morph_weight=0.3, atd_rounds=2, conf_thr=0.85)),
        ("TD_ATD_hybrid_T30_ct90",      dict(mode="hybrid", temperature=30.0, n_iter_td=15, lambda_reg=1.0,
                                              morph_weight=0.3, atd_rounds=2, conf_thr=0.90)),
        ("TD_ATD_hybrid_T30_ct80",      dict(mode="hybrid", temperature=30.0, n_iter_td=15, lambda_reg=1.0,
                                              morph_weight=0.3, atd_rounds=2, conf_thr=0.80)),
        ("TD_ATD_hybrid_T30_3rnd",      dict(mode="hybrid", temperature=30.0, n_iter_td=15, lambda_reg=1.0,
                                              morph_weight=0.3, atd_rounds=3, conf_thr=0.85)),
    ]

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"\n{'='*80}\nSeed {seed}\n{'='*80}", flush=True)
        si = select_support(lt, seed, cids)
        sbc = {c: bc_t[si[c]] for c in cids}
        sph = {c: ph_t[si[c]] for c in cids}
        sdn = {c: dn_t[si[c]] for c in cids}
        sm = {c: mt[si[c]] for c in cids}

        for name, cfg in configs:
            if cfg == "sadc":
                from sadc_v3 import sadc_v3
                m = sadc_v3(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, cids,
                            use_sfa=False, use_bdc=False, use_mtks=False, use_atd=True)
            elif isinstance(cfg, dict) and cfg.get("mode") == "hybrid":
                kw = {k: v for k, v in cfg.items() if k != "mode"}
                m = transductive_atd_hybrid(
                    bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, cids, **kw)
            else:
                m = transductive_classify(
                    bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, cids, **cfg)

            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids:
                all_results[name]["pc"][c].append(m["pc"][c]["f1"])
            print(f"  {name:<50} acc={m['acc']:.4f} mf1={m['mf1']:.4f} "
                  f"Eos={m['pc'][3]['f1']:.4f}", flush=True)

    print(f"\n{'='*150}")
    print("TRANSDUCTIVE CLIP RESULTS (5 seeds, data2_organized)")
    print(f"{'='*150}")
    h = f"{'Strategy':<55} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h)
    print("-" * 150)
    sr = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for n, v in sr:
        print_row(n, v, cids)

    best = sr[0]
    print(f"\n*** BEST: {best[0]} ***")
    print(f"    Acc={np.mean(best[1]['acc']):.4f}, mF1={np.mean(best[1]['mf1']):.4f}")
    baseline = all_results.get("SADC_ATD_baseline", {})
    if baseline:
        print(f"    vs SADC ATD baseline: mF1={np.mean(baseline['mf1']):.4f}")
        print(f"    Improvement: {np.mean(best[1]['mf1']) - np.mean(baseline['mf1']):+.4f}")


if __name__ == "__main__":
    main()
