#!/usr/bin/env python3
"""
Tip-Adapter-F + Laplacian Label Propagation for BALF 10-shot Classification.

Two complementary approaches:
1. Tip-Adapter-F: Learnable cache model with exp(-beta*(1-cos)) affinity
2. Label Propagation: Graph-based transductive inference on query+support

Both operate on pre-extracted multi-backbone features.
"""
import sys, random, itertools
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


# ==================== Tip-Adapter ====================

def tip_adapter(q_feats, s_feats_per_class, cids, beta=5.5, alpha=1.0):
    """
    Tip-Adapter: Training-free cache model.
    phi(x) = exp(-beta * (1 - x)) where x is cosine similarity.
    """
    K = len(cids)
    keys = []
    values = []
    for ki, c in enumerate(cids):
        sf = s_feats_per_class[c]
        sf_norm = sf / (np.linalg.norm(sf, axis=1, keepdims=True) + 1e-8)
        keys.append(sf_norm)
        one_hot = np.zeros((len(sf), K))
        one_hot[:, ki] = 1.0
        values.append(one_hot)

    keys = np.concatenate(keys, axis=0)     # (N_support, D)
    values = np.concatenate(values, axis=0)  # (N_support, K)

    q_norm = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-8)
    cos_sim = q_norm @ keys.T               # (N_query, N_support)
    affinity = np.exp(-beta * (1.0 - cos_sim))  # (N_query, N_support)
    cache_logits = affinity @ values         # (N_query, K)

    prototypes = np.zeros((K, q_feats.shape[1]))
    for ki, c in enumerate(cids):
        prototypes[ki] = s_feats_per_class[c].mean(0)
        prototypes[ki] /= np.linalg.norm(prototypes[ki]) + 1e-8
    zero_shot_logits = q_norm @ prototypes.T  # (N_query, K)

    logits = alpha * cache_logits + zero_shot_logits
    return logits


def tip_adapter_f(q_feats, s_feats_per_class, s_labels_flat, cids,
                  beta=5.5, alpha=1.0, lr=0.001, epochs=20, train_feats=None):
    """
    Tip-Adapter-F: Fine-tuned cache keys via SGD on support set.
    Only updates cache keys, not backbone.
    """
    K = len(cids)
    keys_list, values_list = [], []
    for ki, c in enumerate(cids):
        sf = s_feats_per_class[c].copy()
        sf_norm = sf / (np.linalg.norm(sf, axis=1, keepdims=True) + 1e-8)
        keys_list.append(sf_norm)
        one_hot = np.zeros((len(sf), K))
        one_hot[:, ki] = 1.0
        values_list.append(one_hot)

    keys = np.concatenate(keys_list, axis=0)
    values = np.concatenate(values_list, axis=0)
    labels = np.array(s_labels_flat)

    prototypes = np.zeros((K, q_feats.shape[1]))
    for ki, c in enumerate(cids):
        prototypes[ki] = s_feats_per_class[c].mean(0)
        prototypes[ki] /= np.linalg.norm(prototypes[ki]) + 1e-8

    s_all = np.concatenate([s_feats_per_class[c] for c in cids], axis=0)
    s_norm = s_all / (np.linalg.norm(s_all, axis=1, keepdims=True) + 1e-8)

    for epoch in range(epochs):
        cos_sim = s_norm @ keys.T
        affinity = np.exp(-beta * (1.0 - cos_sim))
        cache_logits = affinity @ values
        zs_logits = s_norm @ prototypes.T
        logits = alpha * cache_logits + zs_logits

        probs = softmax(logits, axis=1)
        loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))

        grad_logits = probs.copy()
        grad_logits[np.arange(len(labels)), labels] -= 1.0
        grad_logits /= len(labels)

        grad_cache = alpha * grad_logits
        grad_affinity = grad_cache @ values.T
        grad_cos = grad_affinity * affinity * beta
        grad_keys = grad_cos.T @ s_norm

        keys -= lr * grad_keys
        keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8

    q_norm = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-8)
    cos_sim = q_norm @ keys.T
    affinity = np.exp(-beta * (1.0 - cos_sim))
    cache_logits = affinity @ values
    zs_logits = q_norm @ prototypes.T
    logits = alpha * cache_logits + zs_logits
    return logits


# ==================== Label Propagation ====================

def label_propagation(q_feats, s_feats_per_class, cids, k_nn=20, alpha_lp=0.5, n_iter=20):
    """
    Laplacian-based label propagation.
    Build a kNN graph over support+query, propagate labels from support.
    """
    K = len(cids)
    support_feats, support_labels = [], []
    for ki, c in enumerate(cids):
        support_feats.append(s_feats_per_class[c])
        support_labels.extend([ki] * len(s_feats_per_class[c]))
    support_feats = np.concatenate(support_feats, axis=0)
    support_labels = np.array(support_labels)

    N_s = len(support_feats)
    N_q = len(q_feats)
    all_feats = np.concatenate([support_feats, q_feats], axis=0)
    all_norm = all_feats / (np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8)

    N = N_s + N_q
    W = np.zeros((N, N))
    sim = all_norm @ all_norm.T

    for i in range(N):
        neighbors = np.argsort(sim[i])[::-1][1:k_nn+1]
        for j in neighbors:
            w = max(sim[i, j], 0)
            W[i, j] = w
            W[j, i] = w

    D = np.diag(W.sum(axis=1) + 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    S = D_inv_sqrt @ W @ D_inv_sqrt

    Y = np.zeros((N, K))
    for i in range(N_s):
        Y[i, support_labels[i]] = 1.0

    F = Y.copy()
    for it in range(n_iter):
        F = alpha_lp * (S @ F) + (1.0 - alpha_lp) * Y

    query_scores = F[N_s:]
    return query_scores


# ==================== Multi-backbone Fusion ====================

def fuse_logits(logits_list, weights):
    result = np.zeros_like(logits_list[0])
    for l, w in zip(logits_list, weights):
        l_norm = l - l.mean(axis=1, keepdims=True)
        l_std = l.std(axis=1, keepdims=True) + 1e-8
        result += w * (l_norm / l_std)
    return result


def run_classification(q_bc, q_ph, q_dn, q_morph, q_labels,
                       s_bc, s_ph, s_dn, s_morph, cids,
                       method="tip_adapter", beta=5.5, alpha_ta=1.0,
                       backbone_weights=None, morph_weight=0.33,
                       k_nn=20, alpha_lp=0.5, lp_iters=20,
                       tip_f_epochs=20, tip_f_lr=0.001):
    K = len(cids)

    if backbone_weights is None:
        backbone_weights = [0.50, 0.30, 0.20]

    if method == "tip_adapter":
        l_bc = tip_adapter(q_bc, s_bc, cids, beta, alpha_ta)
        l_ph = tip_adapter(q_ph, s_ph, cids, beta, alpha_ta)
        l_dn = tip_adapter(q_dn, s_dn, cids, beta, alpha_ta)
        logits = fuse_logits([l_bc, l_ph, l_dn], backbone_weights)

    elif method == "tip_adapter_f":
        s_labels_flat = []
        for ki, c in enumerate(cids):
            s_labels_flat.extend([ki] * len(s_bc[c]))
        l_bc = tip_adapter_f(q_bc, s_bc, s_labels_flat, cids, beta, alpha_ta,
                             tip_f_lr, tip_f_epochs)
        l_ph = tip_adapter_f(q_ph, s_ph, s_labels_flat, cids, beta, alpha_ta,
                             tip_f_lr, tip_f_epochs)
        l_dn = tip_adapter_f(q_dn, s_dn, s_labels_flat, cids, beta, alpha_ta,
                             tip_f_lr, tip_f_epochs)
        logits = fuse_logits([l_bc, l_ph, l_dn], backbone_weights)

    elif method == "label_prop":
        l_bc = label_propagation(q_bc, s_bc, cids, k_nn, alpha_lp, lp_iters)
        l_ph = label_propagation(q_ph, s_ph, cids, k_nn, alpha_lp, lp_iters)
        l_dn = label_propagation(q_dn, s_dn, cids, k_nn, alpha_lp, lp_iters)
        logits = fuse_logits([l_bc, l_ph, l_dn], backbone_weights)

    else:
        raise ValueError(f"Unknown method: {method}")

    if morph_weight > 0:
        sm_all = np.concatenate([s_morph[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        morph_scores = np.zeros((len(q_labels), K))
        for ki, c in enumerate(cids):
            snm = (s_morph[c] - gm) / gs
            for i in range(len(q_labels)):
                qm = (q_morph[i] - gm) / gs
                md = np.linalg.norm(qm - snm, axis=1)
                morph_scores[i, ki] = np.mean(1.0 / (1.0 + np.sort(md)[:5]))

        morph_norm = morph_scores - morph_scores.mean(1, keepdims=True)
        morph_norm /= morph_norm.std(1, keepdims=True) + 1e-8
        logits = (1.0 - morph_weight) * logits + morph_weight * morph_norm

    preds = [cids[int(np.argmax(logits[i]))] for i in range(len(q_labels))]
    gt = [int(l) for l in q_labels]
    return calc_metrics(gt, preds, cids)


def print_row(name, v, cids):
    pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    print(f"{name:<55} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  "
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
        ("SADC_ATD_baseline",            "sadc"),
        # Tip-Adapter (training-free)
        ("TipA_b3_a1",                   dict(method="tip_adapter", beta=3.0, alpha_ta=1.0, morph_weight=0.33)),
        ("TipA_b5_a1",                   dict(method="tip_adapter", beta=5.5, alpha_ta=1.0, morph_weight=0.33)),
        ("TipA_b10_a1",                  dict(method="tip_adapter", beta=10.0, alpha_ta=1.0, morph_weight=0.33)),
        ("TipA_b5_a05",                  dict(method="tip_adapter", beta=5.5, alpha_ta=0.5, morph_weight=0.33)),
        ("TipA_b5_a2",                   dict(method="tip_adapter", beta=5.5, alpha_ta=2.0, morph_weight=0.33)),
        ("TipA_b5_mw0",                  dict(method="tip_adapter", beta=5.5, alpha_ta=1.0, morph_weight=0.0)),
        ("TipA_b5_mw05",                 dict(method="tip_adapter", beta=5.5, alpha_ta=1.0, morph_weight=0.5)),
        ("TipA_b5_bw42_18_07",           dict(method="tip_adapter", beta=5.5, alpha_ta=1.0, morph_weight=0.33,
                                                backbone_weights=[0.42, 0.18, 0.07])),
        # Tip-Adapter-F (fine-tuned)
        ("TipAF_b5_e20",                 dict(method="tip_adapter_f", beta=5.5, alpha_ta=1.0, morph_weight=0.33,
                                               tip_f_epochs=20, tip_f_lr=0.001)),
        ("TipAF_b5_e50",                 dict(method="tip_adapter_f", beta=5.5, alpha_ta=1.0, morph_weight=0.33,
                                               tip_f_epochs=50, tip_f_lr=0.001)),
        ("TipAF_b5_e20_lr01",            dict(method="tip_adapter_f", beta=5.5, alpha_ta=1.0, morph_weight=0.33,
                                               tip_f_epochs=20, tip_f_lr=0.01)),
        ("TipAF_b10_e20",                dict(method="tip_adapter_f", beta=10.0, alpha_ta=1.0, morph_weight=0.33,
                                               tip_f_epochs=20, tip_f_lr=0.001)),
        ("TipAF_b5_e20_a2",              dict(method="tip_adapter_f", beta=5.5, alpha_ta=2.0, morph_weight=0.33,
                                               tip_f_epochs=20, tip_f_lr=0.001)),
        # Label Propagation
        ("LP_k10_a05",                   dict(method="label_prop", k_nn=10, alpha_lp=0.5, morph_weight=0.33)),
        ("LP_k20_a05",                   dict(method="label_prop", k_nn=20, alpha_lp=0.5, morph_weight=0.33)),
        ("LP_k20_a03",                   dict(method="label_prop", k_nn=20, alpha_lp=0.3, morph_weight=0.33)),
        ("LP_k20_a07",                   dict(method="label_prop", k_nn=20, alpha_lp=0.7, morph_weight=0.33)),
        ("LP_k20_a05_mw0",              dict(method="label_prop", k_nn=20, alpha_lp=0.5, morph_weight=0.0)),
        ("LP_k50_a05",                   dict(method="label_prop", k_nn=50, alpha_lp=0.5, morph_weight=0.33)),
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
            else:
                m = run_classification(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, cids, **cfg)

            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids:
                all_results[name]["pc"][c].append(m["pc"][c]["f1"])
            print(f"  {name:<50} acc={m['acc']:.4f} mf1={m['mf1']:.4f} "
                  f"Eos={m['pc'][3]['f1']:.4f}", flush=True)

    print(f"\n{'='*150}")
    print("TIP-ADAPTER & LABEL PROPAGATION RESULTS (5 seeds, data2_organized)")
    print(f"{'='*150}")
    h = f"{'Strategy':<55} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h)
    print("-" * 150)
    sr = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for n, v in sr:
        print_row(n, v, cids)

    best = sr[0]
    best_non_baseline = [x for x in sr if x[0] != "SADC_ATD_baseline"][0]
    baseline = all_results.get("SADC_ATD_baseline", {})
    print(f"\n*** BEST overall: {best[0]} mF1={np.mean(best[1]['mf1']):.4f} ***")
    print(f"*** BEST new method: {best_non_baseline[0]} mF1={np.mean(best_non_baseline[1]['mf1']):.4f} ***")
    if baseline:
        print(f"*** SADC ATD baseline: mF1={np.mean(baseline['mf1']):.4f} ***")
        print(f"*** Improvement: {np.mean(best_non_baseline[1]['mf1']) - np.mean(baseline['mf1']):+.4f} ***")


if __name__ == "__main__":
    main()
