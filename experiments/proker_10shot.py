#!/usr/bin/env python3
"""
ProKeR-inspired + TIMO-inspired 10-shot classification.

ProKeR (CVPR 2025): Kernel perspective on few-shot adaptation.
- Uses global regularization in RKHS
- Replaces linear logits with kernel-based scoring

TIMO (AAAI 2025): Text-Image mutual guidance.
- Rectify text prototypes using visual features (IGT)
- Use text features to mitigate anomalous visual matches (TGI)

New ideas:
1. RBF kernel-based prototype matching (ProKeR)
2. Query-adaptive weight fusion based on prediction entropy
3. Soft kNN with distance-weighted voting (all neighbors)
4. Relative representation: express queries relative to support centroid
5. Feature rectification via support-set mean subtraction
"""
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.special import softmax

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
        pr = tp/pp if pp else 0.0
        rc = tp/gp if gp else 0.0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
        pc[c] = {"p": pr, "r": rc, "f1": f1, "n": gp}
        f1s.append(f1)
    return {"acc": correct/total if total else 0, "mf1": float(np.mean(f1s)), "pc": pc}


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels): pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


# ===== Strategy 1: ProKeR - RBF Kernel Scoring =====
def cls_proker(q_feats, q_labels, s_feats, s_morph, q_morph,
               cids, sigma=0.5, lam=0.1, morph_w=0.3):
    """ProKeR-inspired: RBF kernel with global regularization."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    gt, pred = [], []
    for i in range(len(q_labels)):
        qf = q_feats[i]
        qm = (q_morph[i] - gm) / gs
        scores = []
        for c in cids:
            dists = np.linalg.norm(s_feats[c] - qf, axis=1)
            kernel_scores = np.exp(-dists**2 / (2 * sigma**2))
            visual_score = float(np.mean(kernel_scores))

            mdists = np.linalg.norm(((s_morph[c] - gm) / gs) - qm, axis=1)
            morph_kernel = np.exp(-mdists**2 / 2.0)
            morph_score = float(np.mean(morph_kernel))

            scores.append((1 - morph_w) * visual_score + morph_w * morph_score)
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


# ===== Strategy 2: Feature Rectification + kNN =====
def cls_rectified(q_feats, q_labels, s_feats, s_morph, q_morph,
                  cids, vw=0.65, mw=0.35, k=7):
    """Subtract support-set mean from all features before matching."""
    all_s = np.concatenate([s_feats[c] for c in cids])
    mean_feat = all_s.mean(0)

    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    s_rect = {}
    for c in cids:
        sf = s_feats[c] - mean_feat
        norms = np.linalg.norm(sf, axis=1, keepdims=True) + 1e-8
        s_rect[c] = sf / norms
    snm = {c: (s_morph[c] - gm) / gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qf = q_feats[i] - mean_feat
        qf = qf / (np.linalg.norm(qf) + 1e-8)
        qm = (q_morph[i] - gm) / gs
        scores = []
        for c in cids:
            vs = s_rect[c] @ qf
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            comb = vw * vs + mw * ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


# ===== Strategy 3: Soft kNN with distance-weighted voting =====
def cls_soft_knn(q_feats, q_labels, s_feats, s_morph, q_morph,
                 cids, temp=0.1, morph_w=0.3):
    """All-neighbor weighted voting with temperature-scaled softmax."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    all_sf = np.concatenate([s_feats[c] for c in cids])
    all_sm = np.concatenate([(s_morph[c] - gm) / gs for c in cids])
    all_labels = np.concatenate([np.full(len(s_feats[c]), c) for c in cids])

    gt, pred = [], []
    for i in range(len(q_labels)):
        vs = all_sf @ q_feats[i]
        qm = (q_morph[i] - gm) / gs
        md = np.linalg.norm(all_sm - qm, axis=1)
        ms = 1.0 / (1.0 + md)
        combined = (1 - morph_w) * vs + morph_w * ms
        weights = softmax(combined / temp)
        class_scores = {}
        for c in cids:
            mask = all_labels == c
            class_scores[c] = float(np.sum(weights[mask]))
        gt.append(int(q_labels[i]))
        pred.append(max(class_scores, key=class_scores.get))
    return metrics(gt, pred, cids)


# ===== Strategy 4: Query-Adaptive Weight Fusion =====
def cls_adaptive_fusion(q_bclip, q_dino, q_labels, s_bclip, s_dino,
                        s_morph, q_morph, cids, k=7, base_vw=0.5):
    """Adapt visual/morph weights per query based on prediction entropy."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    snm = {c: (s_morph[c] - gm) / gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm) / gs

        vis_scores = []
        for c in cids:
            vb = s_bclip[c] @ q_bclip[i]
            vd = s_dino[c] @ q_dino[i]
            vis_scores.append(float(np.sort(0.7*vb + 0.3*vd)[::-1][:k].mean()))
        vis_arr = np.array(vis_scores)
        vis_prob = softmax(vis_arr * 20)
        vis_entropy = -np.sum(vis_prob * np.log(vis_prob + 1e-10))

        morph_scores = []
        for c in cids:
            md = np.linalg.norm(qm - snm[c], axis=1)
            morph_scores.append(float(np.mean(1.0 / (1.0 + np.sort(md)[:k]))))
        morph_arr = np.array(morph_scores)

        adapt_mw = min(0.6, 0.2 + vis_entropy * 0.3)
        adapt_vw = 1.0 - adapt_mw
        final = adapt_vw * vis_arr + adapt_mw * morph_arr

        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(final))])
    return metrics(gt, pred, cids)


# ===== Strategy 5: Relative Representation =====
def cls_relative(q_feats, q_labels, s_feats, s_morph, q_morph, cids, k=7):
    """Express features relative to class prototypes, then classify."""
    protos = {c: s_feats[c].mean(0) for c in cids}
    proto_stack = np.stack([protos[c] for c in cids])
    proto_stack = proto_stack / (np.linalg.norm(proto_stack, axis=1, keepdims=True) + 1e-8)

    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    s_rel = {}
    for c in cids:
        rel = s_feats[c] @ proto_stack.T
        s_rel[c] = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-8)
    snm = {c: (s_morph[c] - gm) / gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        q_rel = q_feats[i] @ proto_stack.T
        q_rel = q_rel / (np.linalg.norm(q_rel) + 1e-8)
        qm = (q_morph[i] - gm) / gs

        scores = []
        for c in cids:
            vs = s_rel[c] @ q_rel
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            comb = 0.65 * vs + 0.35 * ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


# ===== Strategy 6: Transductive + all best methods =====
def cls_best_transductive(q_bclip, q_dino, q_morph, q_labels,
                           s_bclip_init, s_dino_init, s_morph_init,
                           cids, fisher_w,
                           vw=0.50, dw=0.15, mw=0.35, k=7,
                           n_iter=2, top_k=5, conf_thr=0.025, cascade_thr=0.010,
                           use_rectify=False, use_kernel=False, sigma=0.5):
    """Optimized transductive + cascade with optional rectification and kernel."""
    sm_all = np.concatenate([s_morph_init[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    if use_rectify:
        all_s_b = np.concatenate([s_bclip_init[c] for c in cids])
        mean_b = all_s_b.mean(0)
    else:
        mean_b = None

    s_b = {c: s_bclip_init[c].copy() for c in cids}
    s_d = {c: s_dino_init[c].copy() for c in cids}
    s_m = {c: s_morph_init[c].copy() for c in cids}

    for _t in range(n_iter):
        snm = {c: (s_m[c] - gm) / gs for c in cids}
        preds, margins = [], []
        for i in range(len(q_labels)):
            qb = q_bclip[i]
            qd = q_dino[i]
            if use_rectify and mean_b is not None:
                qb = qb - mean_b
                qb = qb / (np.linalg.norm(qb) + 1e-8)
            qm = (q_morph[i] - gm) / gs
            scores = []
            for c in cids:
                sb = s_b[c]
                if use_rectify and mean_b is not None:
                    sb = sb - mean_b
                    sb = sb / (np.linalg.norm(sb, axis=1, keepdims=True) + 1e-8)
                if use_kernel:
                    dists = np.linalg.norm(sb - qb, axis=1)
                    vs_b = np.exp(-dists**2 / (2*sigma**2))
                else:
                    vs_b = sb @ qb
                vs_d = s_d[c] @ qd
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                comb = vw * vs_b + dw * vs_d + mw * ms
                scores.append(float(np.sort(comb)[::-1][:k].mean()))
            s_arr = np.array(scores)
            preds.append(cids[int(np.argmax(s_arr))])
            sorted_s = np.sort(s_arr)[::-1]
            margins.append(sorted_s[0] - sorted_s[1])

        preds = np.array(preds)
        margins = np.array(margins)
        for c in cids:
            c_mask = (preds == c) & (margins > conf_thr)
            c_idx = np.where(c_mask)[0]
            if len(c_idx) == 0: continue
            sorted_idx = c_idx[np.argsort(margins[c_idx])[::-1][:top_k]]
            s_b[c] = np.concatenate([s_bclip_init[c], q_bclip[sorted_idx] * 0.5])
            s_d[c] = np.concatenate([s_dino_init[c], q_dino[sorted_idx] * 0.5])
            s_m[c] = np.concatenate([s_morph_init[c], q_morph[sorted_idx]])

    snm = {c: (s_m[c] - gm) / gs for c in cids}
    snm_w = {c: snm[c] * fisher_w for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qb, qd = q_bclip[i], q_dino[i]
        if use_rectify and mean_b is not None:
            qb = qb - mean_b
            qb = qb / (np.linalg.norm(qb) + 1e-8)
        qm = (q_morph[i] - gm) / gs
        qm_w = qm * fisher_w
        scores = {}
        for c in cids:
            sb = s_b[c]
            if use_rectify and mean_b is not None:
                sb = sb - mean_b
                sb = sb / (np.linalg.norm(sb, axis=1, keepdims=True) + 1e-8)
            if use_kernel:
                dists = np.linalg.norm(sb - qb, axis=1)
                vs_b = np.exp(-dists**2 / (2*sigma**2))
            else:
                vs_b = sb @ qb
            vs_d = s_d[c] @ qd
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            comb = vw * vs_b + dw * vs_d + mw * ms
            scores[c] = float(np.sort(comb)[::-1][:k].mean())

        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0] - np.sort(s_arr)[::-1][1]

        if top1 in [3, 4] and margin < cascade_thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0 / (1.0 + np.sort(md_w)[:5])))
                vs_b_s = float(np.sort((s_b[gc] @ qb) if not use_kernel else
                                        np.exp(-np.linalg.norm(s_b[gc]-qb, axis=1)**2/(2*sigma**2)))[::-1][:3].mean())
                vs_d_s = float(np.sort(s_d[gc] @ qd)[::-1][:3].mean())
                scores[gc] = 0.30 * vs_b_s + 0.15 * vs_d_s + 0.55 * mscore
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
    n_dims = morph_train.shape[1]
    eos, neu = morph_train[labels_train == 3], morph_train[labels_train == 4]
    fisher_w = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos[:, d]) - np.mean(neu[:, d]))**2 / (np.var(eos[:, d]) + np.var(neu[:, d]) + 1e-10)
        fisher_w[d] = 1.0 + f * 2.0

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"Seed {seed}...")
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}

        # --- Independent strategies ---
        strats = {}

        # ProKeR kernel variants
        for sigma in [0.3, 0.5, 0.7, 1.0]:
            for mw in [0.2, 0.3, 0.4]:
                name = f"proker_s{sigma}_mw{mw}"
                strats[name] = lambda q=bclip_val, ql=labels_val, sf=s_bclip, sm=s_morph, qm=morph_val, \
                                      s=sigma, m=mw: cls_proker(q, ql, sf, sm, qm, cids, s, 0.1, m)

        # Feature rectification
        for k in [5, 7]:
            for vw, mw in [(0.65, 0.35), (0.60, 0.40), (0.55, 0.45)]:
                name = f"rectified_vw{vw}_k{k}"
                strats[name] = lambda q=bclip_val, ql=labels_val, sf=s_bclip, sm=s_morph, qm=morph_val, \
                                      v=vw, m=mw, kk=k: cls_rectified(q, ql, sf, sm, qm, cids, v, m, kk)

        # Soft kNN
        for temp in [0.05, 0.1, 0.2]:
            for mw in [0.2, 0.3, 0.4]:
                name = f"soft_knn_t{temp}_mw{mw}"
                strats[name] = lambda q=bclip_val, ql=labels_val, sf=s_bclip, sm=s_morph, qm=morph_val, \
                                      t=temp, m=mw: cls_soft_knn(q, ql, sf, sm, qm, cids, t, m)

        # Adaptive fusion
        name = "adaptive_fusion"
        strats[name] = lambda: cls_adaptive_fusion(
            bclip_val, dino_val, labels_val, s_bclip, s_dino, s_morph, morph_val, cids)

        # Relative representation
        for k in [5, 7]:
            name = f"relative_k{k}"
            strats[name] = lambda q=bclip_val, ql=labels_val, sf=s_bclip, sm=s_morph, qm=morph_val, \
                                  kk=k: cls_relative(q, ql, sf, sm, qm, cids, kk)

        for sn, fn in strats.items():
            try:
                m = fn()
                all_results[sn]["acc"].append(m["acc"])
                all_results[sn]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[sn]["pc"][c].append(m["pc"][c]["f1"])
            except Exception as e:
                print(f"    {sn} FAILED: {e}")

        # --- Transductive + cascade variants ---
        trans_configs = [
            ("trans_cas_baseline", {"vw": 0.50, "dw": 0.15, "mw": 0.35}),
            ("trans_cas_rect", {"vw": 0.50, "dw": 0.15, "mw": 0.35, "use_rectify": True}),
            ("trans_cas_kernel", {"vw": 0.50, "dw": 0.15, "mw": 0.35, "use_kernel": True, "sigma": 0.5}),
            ("trans_cas_kern_rect", {"vw": 0.50, "dw": 0.15, "mw": 0.35,
                                     "use_rectify": True, "use_kernel": True, "sigma": 0.5}),
            ("trans_cas_mw40", {"vw": 0.45, "dw": 0.15, "mw": 0.40}),
            ("trans_cas_mw30", {"vw": 0.55, "dw": 0.15, "mw": 0.30}),
            ("trans_cas_k5", {"vw": 0.50, "dw": 0.15, "mw": 0.35, "k": 5}),
            ("trans_cas_ct008", {"vw": 0.50, "dw": 0.15, "mw": 0.35, "cascade_thr": 0.008}),
            ("trans_cas_ct015", {"vw": 0.50, "dw": 0.15, "mw": 0.35, "cascade_thr": 0.015}),
            ("trans_cas_iter3", {"vw": 0.50, "dw": 0.15, "mw": 0.35, "n_iter": 3}),
        ]
        for tc_name, tc_kwargs in trans_configs:
            try:
                m = cls_best_transductive(
                    bclip_val, dino_val, morph_val, labels_val,
                    s_bclip, s_dino, s_morph, cids, fisher_w, **tc_kwargs)
                all_results[tc_name]["acc"].append(m["acc"])
                all_results[tc_name]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[tc_name]["pc"][c].append(m["pc"][c]["f1"])
            except Exception as e:
                print(f"    {tc_name} FAILED: {e}")

    # Print results
    print(f"\n{'='*130}")
    print("ProKeR/TIMO-INSPIRED RESULTS (5 seeds)")
    print(f"{'='*130}")
    header = f"{'Strategy':<45} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 130)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:25]:
        if len(v["acc"]) < 3:
            continue
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<45} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}, "
          f"Eos={np.mean(best[1]['pc'][3]):.4f}, Acc={np.mean(best[1]['acc']):.4f}")
    print(f"\nBaseline (trans_cas_baseline): "
          f"mF1={np.mean(all_results['trans_cas_baseline']['mf1']):.4f}, "
          f"Eos={np.mean(all_results['trans_cas_baseline']['pc'][3]):.4f}")


if __name__ == "__main__":
    main()
