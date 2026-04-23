#!/usr/bin/env python3
"""
Final ensemble: combine multiple diverse classifiers for robust 10-shot.
Strategies: weighted vote from diverse classifier pool.
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


def score_cell(i, q_bclip, q_dino, q_morph, s_bclip, s_dino, s_morph,
               snm, cids, bw, dw, mw, k, gm, gs):
    qm = (q_morph[i]-gm)/gs
    scores = {}
    for c in cids:
        vs_b = s_bclip[c] @ q_bclip[i]
        vs_d = s_dino[c] @ q_dino[i]
        md = np.linalg.norm(qm - snm[c], axis=1)
        ms = 1.0/(1.0+md)
        comb = bw*vs_b + dw*vs_d + mw*ms
        scores[c] = float(np.sort(comb)[::-1][:k].mean())
    return scores


def ensemble_classify(q_bclip, q_dino, q_morph, q_labels,
                       s_bclip, s_dino, s_morph, cids, morph_weights):
    """Ensemble of diverse classifiers with confidence-weighted voting."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    snm_w = {c: (s_morph[c]-gm)/gs * morph_weights for c in cids}
    
    configs = [
        (0.45, 0.20, 0.35, 7, 1.0),
        (0.40, 0.25, 0.35, 7, 0.9),
        (0.45, 0.15, 0.40, 7, 0.9),
        (0.40, 0.20, 0.40, 5, 0.8),
        (0.35, 0.25, 0.40, 7, 0.8),
        (0.50, 0.20, 0.30, 7, 0.7),
    ]
    
    gt, pred = [], []
    for i in range(len(q_labels)):
        # Collect votes from all classifiers
        vote_scores = defaultdict(float)
        for bw, dw, mw, k, weight in configs:
            scores = score_cell(i, q_bclip, q_dino, q_morph,
                                s_bclip, s_dino, s_morph, snm, cids,
                                bw, dw, mw, k, gm, gs)
            s_arr = np.array([scores[c] for c in cids])
            # Softmax-like normalization
            s_arr = np.exp(s_arr * 20) / np.exp(s_arr * 20).sum()
            for j, c in enumerate(cids):
                vote_scores[c] += weight * s_arr[j]
        
        # Initial prediction from ensemble
        top1 = max(vote_scores, key=vote_scores.get)
        
        # Cascade for Eos/Neu if uncertain
        v_arr = np.array([vote_scores[c] for c in cids])
        v_sorted = np.sort(v_arr)[::-1]
        margin = v_sorted[0] - v_sorted[1]
        
        if top1 in [3, 4] and margin < 0.05:
            qm = (q_morph[i]-gm)/gs
            qm_w = qm * morph_weights
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(s_bclip[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(s_dino[gc] @ q_dino[i])[::-1][:3].mean())
                vote_scores[gc] = 0.30*vs_b_s + 0.15*vs_d_s + 0.55*mscore
            top1 = 3 if vote_scores[3] > vote_scores[4] else 4
        
        gt.append(int(q_labels[i]))
        pred.append(top1)
    return metrics(gt, pred, cids)


def transductive_ensemble(q_bclip, q_dino, q_morph, q_labels,
                           s_bclip_init, s_dino_init, s_morph_init,
                           cids, morph_weights, n_iter=2, top_k=5, conf_thr=0.025):
    """Transductive refinement + ensemble."""
    s_bclip = {c: s_bclip_init[c].copy() for c in cids}
    s_dino = {c: s_dino_init[c].copy() for c in cids}
    s_morph = {c: s_morph_init[c].copy() for c in cids}
    
    for t in range(n_iter):
        sm_all = np.concatenate([s_morph[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
        snm = {c: (s_morph[c]-gm)/gs for c in cids}
        
        preds = []; margins = []
        for i in range(len(q_labels)):
            scores = score_cell(i, q_bclip, q_dino, q_morph,
                                s_bclip, s_dino, s_morph, snm, cids,
                                0.45, 0.20, 0.35, 7, gm, gs)
            s_arr = np.array([scores[c] for c in cids])
            s_sorted = np.sort(s_arr)[::-1]
            preds.append(cids[int(np.argmax(s_arr))])
            margins.append(s_sorted[0] - s_sorted[1])
        
        preds = np.array(preds)
        margins = np.array(margins)
        
        for c in cids:
            c_mask = (preds == c) & (margins > conf_thr)
            c_idx = np.where(c_mask)[0]
            if len(c_idx) == 0: continue
            sorted_idx = c_idx[np.argsort(margins[c_idx])[::-1][:top_k]]
            s_bclip[c] = np.concatenate([s_bclip_init[c], q_bclip[sorted_idx]*0.5])
            s_dino[c] = np.concatenate([s_dino_init[c], q_dino[sorted_idx]*0.5])
            s_morph[c] = np.concatenate([s_morph_init[c], q_morph[sorted_idx]])
    
    return ensemble_classify(q_bclip, q_dino, q_morph, q_labels,
                              s_bclip, s_dino, s_morph, cids, morph_weights)


def main():
    bclip_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bclip_val, morph_val, labels_val = load_cache("biomedclip", "val")
    dino_train, _, _ = load_cache("dinov2_s", "train")
    dino_val, _, _ = load_cache("dinov2_s", "val")
    
    cids = sorted(CLASS_NAMES.keys())
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
        
        # Ensemble only
        m = ensemble_classify(bclip_val, dino_val, morph_val, labels_val,
                               s_bclip, s_dino, s_morph, cids, mw_fisher)
        all_results["ensemble"]["acc"].append(m["acc"]); all_results["ensemble"]["mf1"].append(m["mf1"])
        for c in cids: all_results["ensemble"]["pc"][c].append(m["pc"][c]["f1"])
        
        # Transductive + ensemble
        for n_iter in [2, 3]:
            for top_k in [5, 10]:
                for conf in [0.020, 0.025]:
                    name = f"trans_ens:i{n_iter}_k{top_k}_c{conf}"
                    m = transductive_ensemble(
                        bclip_val, dino_val, morph_val, labels_val,
                        s_bclip, s_dino, s_morph, cids, mw_fisher,
                        n_iter, top_k, conf)
                    all_results[name]["acc"].append(m["acc"])
                    all_results[name]["mf1"].append(m["mf1"])
                    for c in cids:
                        all_results[name]["pc"][c].append(m["pc"][c]["f1"])
    
    print(f"\n{'='*120}")
    print("ENSEMBLE RESULTS (5 seeds)")
    print(f"{'='*120}")
    header = f"{'Strategy':<45} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 120)
    
    for name, v in sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"])):
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<45} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    best = max(all_results.items(), key=lambda x: np.mean(x[1]["mf1"]))
    print(f"\nBEST: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}")


if __name__ == "__main__":
    main()
