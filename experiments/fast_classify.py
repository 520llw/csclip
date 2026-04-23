#!/usr/bin/env python3
"""
Fast classification experiments using pre-cached features.
No GPU needed - pure numpy classification with rapid iteration.

Strategies:
1. Dual-backbone kNN (BiomedCLIP + DINOv2 + Morph)
2. Cascade Eos/Neu refinement  
3. Per-class optimized weights
4. ProKeR-inspired kernel regression
5. Feature augmentation via support mixup
6. Confidence-weighted voting
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


def load_cache(model_name, split):
    d = np.load(CACHE_DIR / f"{model_name}_{split}.npz")
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


def select_support(train_labels, seed, cids):
    """Select 10 indices per class."""
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(train_labels):
        pc[int(l)].append(i)
    support_idx = {}
    for c in cids:
        support_idx[c] = random.sample(pc[c], min(N_SHOT, len(pc[c])))
    return support_idx


# ========== Classifiers ==========

def cls_dual_bb(q_bclip, q_dino, q_morph, q_labels,
                s_bclip, s_dino, s_morph, s_labels_dict,
                cids, bw=0.40, dw=0.25, mw=0.35, k=7):
    sm = []
    for c in cids:
        sm.append(s_morph[c])
    sm_all = np.concatenate(sm)
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        scores = []
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            top = np.sort(comb)[::-1][:k]
            scores.append(float(top.mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_cascade_v2(q_bclip, q_dino, q_morph, q_labels,
                    s_bclip, s_dino, s_morph, s_labels_dict,
                    cids, thr=0.015):
    """Cascade: global classifier → Eos/Neu specialist when uncertain."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        scores = {}
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = 0.40*vs_b + 0.25*vs_d + 0.35*ms
            scores[c] = float(np.sort(comb)[::-1][:7].mean())
        
        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0] - np.sort(s_arr)[::-1][1]
        
        if top1 in [3, 4] and margin < thr:
            # Eos/Neu specialist: heavy morph + color features
            morph_idx = list(range(12, len(qm)))
            q_sub = qm[morph_idx]
            for gc_id in [3, 4]:
                sub = snm[gc_id][:, morph_idx]
                dd = np.linalg.norm(q_sub - sub, axis=1)
                scores[gc_id] = float(np.mean(1.0/(1.0+np.sort(dd)[:5])))
                # Also boost with visual
                scores[gc_id] += 0.3 * float(np.sort(s_bclip[gc_id] @ q_bclip[i])[::-1][:3].mean())
            top1 = 3 if scores[3] > scores[4] else 4
        
        gt.append(int(q_labels[i]))
        pred.append(top1)
    return metrics(gt, pred, cids)


def cls_proker(q_bclip, q_dino, q_morph, q_labels,
               s_bclip, s_dino, s_morph, s_labels_dict,
               cids, sigma=0.5, lam=0.1):
    """ProKeR-inspired: Kernel regression with Nadaraya-Watson estimator + regularization."""
    n_c = len(cids)
    cid2i = {c: i for i, c in enumerate(cids)}
    
    # Build cache (all support features concatenated)
    all_feats = np.concatenate([s_bclip[c] for c in cids])
    all_dino = np.concatenate([s_dino[c] for c in cids])
    all_labels = []
    for c in cids:
        for _ in range(len(s_bclip[c])):
            l = np.zeros(n_c, np.float32); l[cid2i[c]] = 1.0
            all_labels.append(l)
    all_labels = np.stack(all_labels)
    
    # Concat features for richer representation
    all_cat = np.concatenate([all_feats, all_dino * 0.5], axis=1)
    all_cat = all_cat / (np.linalg.norm(all_cat, axis=1, keepdims=True) + 1e-8)
    
    # Morph
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm_all = (sm_all - gm) / gs
    
    gt, pred = [], []
    for i in range(len(q_labels)):
        q_cat = np.concatenate([q_bclip[i], q_dino[i] * 0.5])
        q_cat = q_cat / (np.linalg.norm(q_cat) + 1e-8)
        
        # Visual kernel
        sims = all_cat @ q_cat
        weights_vis = np.exp(sims / sigma)
        
        # Morph kernel
        qm = (q_morph[i] - gm) / gs
        dists = np.linalg.norm(qm - snm_all, axis=1)
        weights_morph = np.exp(-dists / (2 * sigma))
        
        # Combined kernel with regularization
        weights = weights_vis + 0.5 * weights_morph
        weights = weights / (weights.sum() + lam)
        
        logits = weights @ all_labels
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(logits))])
    return metrics(gt, pred, cids)


def cls_augmented_support(q_bclip, q_dino, q_morph, q_labels,
                          s_bclip, s_dino, s_morph, s_labels_dict,
                          cids, bw=0.40, dw=0.25, mw=0.35, k=7, n_aug=5):
    """Augment support set with noise + mixup in feature space."""
    aug_bclip = {}; aug_dino = {}; aug_morph = {}
    for c in cids:
        augb = [s_bclip[c]]
        augd = [s_dino[c]]
        augm = [s_morph[c]]
        for _ in range(n_aug):
            noise_b = np.random.randn(*s_bclip[c].shape) * 0.02
            noise_d = np.random.randn(*s_dino[c].shape) * 0.02
            ab = s_bclip[c] + noise_b
            ab = ab / (np.linalg.norm(ab, axis=1, keepdims=True) + 1e-8)
            ad = s_dino[c] + noise_d
            ad = ad / (np.linalg.norm(ad, axis=1, keepdims=True) + 1e-8)
            augb.append(ab.astype(np.float32))
            augd.append(ad.astype(np.float32))
            augm.append(s_morph[c])
        # Intra-class mixup
        n = len(s_bclip[c])
        for _ in range(n_aug):
            idx1 = np.random.randint(0, n, n)
            idx2 = np.random.randint(0, n, n)
            lam = np.random.beta(0.5, 0.5, n)[:, None]
            mb = lam * s_bclip[c][idx1] + (1-lam) * s_bclip[c][idx2]
            mb = mb / (np.linalg.norm(mb, axis=1, keepdims=True) + 1e-8)
            md = lam * s_dino[c][idx1] + (1-lam) * s_dino[c][idx2]
            md = md / (np.linalg.norm(md, axis=1, keepdims=True) + 1e-8)
            mm = lam * s_morph[c][idx1] + (1-lam) * s_morph[c][idx2]
            augb.append(mb.astype(np.float32))
            augd.append(md.astype(np.float32))
            augm.append(mm.astype(np.float32))
        aug_bclip[c] = np.concatenate(augb)
        aug_dino[c] = np.concatenate(augd)
        aug_morph[c] = np.concatenate(augm)
    
    return cls_dual_bb(q_bclip, q_dino, q_morph, q_labels,
                       aug_bclip, aug_dino, aug_morph, s_labels_dict,
                       cids, bw, dw, mw, k)


def cls_bclip_only_dual(q_bclip, q_morph, q_labels,
                         s_bclip, s_morph, cids, vw=0.65, mw=0.35, k=7):
    """BiomedCLIP-only dual-space (baseline)."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        scores = []
        for c in cids:
            vs = s_bclip[c] @ q_bclip[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = vw*vs + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


# ========== Main ==========

def main():
    print("Loading cached features...")
    bclip_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bclip_val, morph_val, labels_val = load_cache("biomedclip", "val")
    dino_train, _, _ = load_cache("dinov2_s", "train")
    dino_val, _, _ = load_cache("dinov2_s", "val")
    
    print(f"Train: {len(labels_train)} | Val: {len(labels_val)}")
    print(f"BiomedCLIP: {bclip_train.shape[1]}-dim | DINOv2-S: {dino_train.shape[1]}-dim | Morph: {morph_train.shape[1]}-dim")
    
    cids = sorted(CLASS_NAMES.keys())
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})
    
    for seed in SEEDS:
        support_idx = select_support(labels_train, seed, cids)
        
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}
        
        # ========== STRATEGY SWEEP ==========
        
        # 1. BiomedCLIP-only baseline (for comparison)
        for vw, mw, k in [(0.65, 0.35, 7), (0.60, 0.40, 5)]:
            name = f"bclip:{vw:.0%}v_{mw:.0%}m_k{k}"
            m = cls_bclip_only_dual(bclip_val, morph_val, labels_val, s_bclip, s_morph, cids, vw, mw, k)
            all_results[name]["acc"].append(m["acc"]); all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])
        
        # 2. Dual-backbone sweep
        for bw in [0.30, 0.35, 0.40, 0.45]:
            for dw in [0.15, 0.20, 0.25, 0.30]:
                mw = round(1.0 - bw - dw, 2)
                if mw < 0.15 or mw > 0.50:
                    continue
                for k in [5, 7, 9]:
                    name = f"db:{bw}b_{dw}d_{mw}m_k{k}"
                    m = cls_dual_bb(bclip_val, dino_val, morph_val, labels_val,
                                    s_bclip, s_dino, s_morph, None, cids, bw, dw, mw, k)
                    all_results[name]["acc"].append(m["acc"]); all_results[name]["mf1"].append(m["mf1"])
                    for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])
        
        # 3. Cascade variants
        for thr in [0.010, 0.015, 0.020, 0.025, 0.030]:
            name = f"cascade:thr{thr}"
            m = cls_cascade_v2(bclip_val, dino_val, morph_val, labels_val,
                               s_bclip, s_dino, s_morph, None, cids, thr)
            all_results[name]["acc"].append(m["acc"]); all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])
        
        # 4. ProKeR variants
        for sigma in [0.3, 0.5, 0.7, 1.0]:
            for lam in [0.01, 0.1, 0.5]:
                name = f"proker:s{sigma}_l{lam}"
                m = cls_proker(bclip_val, dino_val, morph_val, labels_val,
                               s_bclip, s_dino, s_morph, None, cids, sigma, lam)
                all_results[name]["acc"].append(m["acc"]); all_results[name]["mf1"].append(m["mf1"])
                for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])
        
        # 5. Augmented support
        np.random.seed(seed)
        for n_aug in [3, 5, 10]:
            name = f"aug:naug{n_aug}_40_25_35_k7"
            m = cls_augmented_support(bclip_val, dino_val, morph_val, labels_val,
                                      s_bclip, s_dino, s_morph, None, cids, 0.40, 0.25, 0.35, 7, n_aug)
            all_results[name]["acc"].append(m["acc"]); all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])
    
    # Print TOP results
    print(f"\n{'='*120}")
    print("FAST CLASSIFY: All Strategies (5 seeds)")
    print(f"{'='*120}")
    header = f"{'Strategy':<45} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 120)
    
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for i, (name, v) in enumerate(sorted_r):
        if i >= 30:
            break
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<45} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    # Eos-focused results
    print(f"\n{'='*120}")
    print("TOP 10 by Eosinophil F1")
    print(f"{'='*120}")
    print(header)
    print("-" * 120)
    sorted_eos = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["pc"][3]))
    for i, (name, v) in enumerate(sorted_eos[:10]):
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<45} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    # Save
    result_file = Path(__file__).parent / "fast_classify_results.json"
    with open(result_file, "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "per_class": {str(c): float(np.mean(v["pc"][c])) for c in cids}}
                   for n, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to {result_file}")


if __name__ == "__main__":
    main()
