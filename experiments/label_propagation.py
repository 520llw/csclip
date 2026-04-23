#!/usr/bin/env python3
"""
Label Propagation for Few-Shot Classification (iLPC-inspired).
Uses the manifold structure of query + support to propagate labels.

Algorithm:
1. Build affinity graph over all samples (support + query)
2. Initialize labels from support set
3. Propagate labels through the graph
4. Output soft predictions for query set

Combined with our dual-backbone features.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

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


def build_combined_features(bclip, dino, morph, morph_mean, morph_std, bw=0.6, dw=0.2, mw=0.2):
    """Build combined feature vector with weighted components."""
    morph_norm = (morph - morph_mean) / morph_std
    combined = np.concatenate([
        bclip * bw,
        dino * dw,
        morph_norm * mw
    ], axis=1)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    return combined / (norms + 1e-8)


def label_propagation(support_feats, support_labels, query_feats, cids,
                       alpha=0.5, sigma=0.5, knn=20):
    """Label propagation on the combined feature space.
    
    F = (I - alpha*W)^{-1} * Y
    where W is the row-normalized affinity matrix and Y is the initial label matrix.
    """
    n_support = support_feats.shape[0]
    n_query = query_feats.shape[0]
    n_total = n_support + n_query
    n_classes = len(cids)
    cid2i = {c: i for i, c in enumerate(cids)}
    
    # Combine all features
    all_feats = np.concatenate([support_feats, query_feats])
    
    # Build kNN affinity matrix
    sims = all_feats @ all_feats.T
    
    # Convert to distance-based weights
    W = np.zeros((n_total, n_total))
    for i in range(n_total):
        # Top-k neighbors
        topk_idx = np.argsort(sims[i])[::-1][1:knn+1]  # exclude self
        for j in topk_idx:
            w = np.exp(-(1 - sims[i, j]) / sigma)
            W[i, j] = w
    
    # Symmetrize
    W = (W + W.T) / 2
    
    # Row normalize
    D = W.sum(axis=1)
    D[D == 0] = 1
    W_norm = W / D[:, None]
    
    # Initial label matrix
    Y = np.zeros((n_total, n_classes))
    for i in range(n_support):
        Y[i, cid2i[support_labels[i]]] = 1.0
    
    # Label propagation: F = (I - alpha*W)^(-1) * Y
    I = np.eye(n_total)
    F = np.linalg.solve(I - alpha * W_norm, Y)
    
    # Extract query predictions
    query_F = F[n_support:]
    preds = [cids[int(np.argmax(query_F[i]))] for i in range(n_query)]
    
    return preds


def lp_with_cascade(support_bclip, support_dino, support_morph, support_labels,
                     query_bclip, query_dino, query_morph, query_labels,
                     cids, morph_weights, alpha=0.5, sigma=0.5, knn=20,
                     cascade_thr=0.008, bw=0.6, dw=0.2, mw=0.2):
    """Label propagation + cascade for Eos/Neu."""
    # Morph normalization from support only
    s_morph = np.concatenate([support_morph[c] for c in cids])
    gm, gs = s_morph.mean(0), s_morph.std(0)+1e-8
    
    # Build combined features
    s_list = []
    s_labels_list = []
    for c in cids:
        for i in range(len(support_bclip[c])):
            s_list.append(np.concatenate([
                support_bclip[c][i] * bw,
                support_dino[c][i] * dw,
                ((support_morph[c][i] - gm)/gs) * mw
            ]))
            s_labels_list.append(c)
    s_feats = np.stack(s_list)
    s_feats = s_feats / (np.linalg.norm(s_feats, axis=1, keepdims=True) + 1e-8)
    
    q_feats_list = []
    for i in range(len(query_labels)):
        q_feats_list.append(np.concatenate([
            query_bclip[i] * bw,
            query_dino[i] * dw,
            ((query_morph[i] - gm)/gs) * mw
        ]))
    q_feats = np.stack(q_feats_list)
    q_feats = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-8)
    
    # Label propagation
    preds = label_propagation(s_feats, s_labels_list, q_feats, cids, alpha, sigma, knn)
    
    gt = [int(l) for l in query_labels]
    return metrics(gt, preds, cids)


def dual_bb_knn(q_bclip, q_dino, q_morph, q_labels,
                 s_bclip, s_dino, s_morph, cids,
                 morph_weights, cascade_thr=0.01,
                 bw=0.45, dw=0.20, mw=0.35, k=7):
    """Our best: dual-backbone + cascade (baseline for comparison)."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    snm_w = {c: (s_morph[c]-gm)/gs * morph_weights for c in cids}
    
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        qm_w = qm * morph_weights
        scores = {}
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
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
                vs_b_s = float(np.sort(s_bclip[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(s_dino[gc] @ q_dino[i])[::-1][:3].mean())
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
    
    # Fisher weights
    eos, neu = morph_train[labels_train==3], morph_train[labels_train==4]
    n_dims = morph_train.shape[1]
    mw_fisher = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos[:,d])-np.mean(neu[:,d]))**2 / (np.var(eos[:,d])+np.var(neu[:,d])+1e-10)
        mw_fisher[d] = 1.0 + f * 2.0
    
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})
    
    for seed in SEEDS:
        print(f"Seed {seed}...")
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}
        
        # Our best baseline: transductive + cascade
        m = dual_bb_knn(bclip_val, dino_val, morph_val, labels_val,
                         s_bclip, s_dino, s_morph, cids, mw_fisher, 0.01)
        all_results["baseline_cas"]["acc"].append(m["acc"])
        all_results["baseline_cas"]["mf1"].append(m["mf1"])
        for c in cids: all_results["baseline_cas"]["pc"][c].append(m["pc"][c]["f1"])
        
        # Label propagation variants
        for alpha in [0.3, 0.5, 0.7]:
            for sigma in [0.3, 0.5, 0.7, 1.0]:
                for knn in [10, 20, 30]:
                    for bw, dw, mwt in [(0.6, 0.2, 0.2), (0.5, 0.2, 0.3), (0.5, 0.15, 0.35)]:
                        name = f"lp:a{alpha}_s{sigma}_k{knn}_b{bw}_d{dw}_m{mwt}"
                        try:
                            m = lp_with_cascade(
                                s_bclip, s_dino, s_morph, None,
                                bclip_val, dino_val, morph_val, labels_val,
                                cids, mw_fisher, alpha, sigma, knn, 0.01, bw, dw, mwt)
                            all_results[name]["acc"].append(m["acc"])
                            all_results[name]["mf1"].append(m["mf1"])
                            for c in cids:
                                all_results[name]["pc"][c].append(m["pc"][c]["f1"])
                        except Exception as e:
                            pass
    
    print(f"\n{'='*125}")
    print("LABEL PROPAGATION RESULTS (5 seeds)")
    print(f"{'='*125}")
    header = f"{'Strategy':<55} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}"
    print(header)
    print("-" * 125)
    
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for i, (name, v) in enumerate(sorted_r[:20]):
        if len(v["acc"]) < 5: continue
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<55} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}")
    
    best = sorted_r[0]
    print(f"\nBEST: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}, Acc={np.mean(best[1]['acc']):.4f}")


if __name__ == "__main__":
    main()
