#!/usr/bin/env python3
"""
Attribute-guided 10-shot classification inspired by AT-Adapter (2026).
Instead of class-level prototypes, use ATTRIBUTE-level prototypes.

Key insight: While cell classes are hard to distinguish (e.g., Eos vs Neu),
their ATTRIBUTES (granule color, nucleus shape, size) are more discriminative.

Approach:
1. Define cytological attributes for each cell type
2. Build attribute-level scoring functions using morphology + visual features
3. Combine attribute scores for final classification
4. Integrate with transductive + cascade
"""
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]

# Cytological attribute definitions based on the "Thirteen Point Strategy"
# Each attribute maps to morphology feature dimensions that express it
MORPH_DIM_NAMES = [
    "area", "perimeter", "circularity", "mean_r", "mean_g", "mean_b",
    "std_intensity", "eccentricity", "solidity", "extent", "aspect_ratio", "compactness",
    "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std",
    "red_dominance", "rg_ratio", "rb_ratio",
    "texture_contrast", "granule_mean",
    "gabor_mean", "gabor_std",
    "lbp_mean", "lbp_std",
    "n_granules", "mean_granule_size", "std_granule_size",
    "hist_entropy", "hist_skewness",
    "nuclear_ratio", "n_lobes", "edge_density",
    "dark_fraction", "red_iqr", "red_gt_green_ratio", "rg_diff_mean", "rg_diff_std",
]

CELL_ATTRIBUTES = {
    3: {  # Eosinophil
        "red_granules": [18, 19, 37, 38],      # red_dominance, rg_ratio, red_gt_green, rg_diff_mean
        "large_granules": [27, 28, 29],         # n_granules, mean_granule_size, std_granule_size
        "bilobed_nucleus": [33, 32],            # n_lobes, nuclear_ratio
        "eosin_staining": [3, 36],              # mean_r, red_iqr
        "uniform_cytoplasm": [6, 21],           # std_intensity, texture_contrast
    },
    4: {  # Neutrophil
        "fine_granules": [27, 28, 29],
        "multilobed_nucleus": [33, 32],
        "pale_cytoplasm": [6, 14, 16],          # std_intensity, s_mean, v_mean
        "low_red_staining": [18, 19, 37],
        "high_texture": [21, 22, 23],           # texture_contrast, granule_mean, gabor_mean
    },
    5: {  # Lymphocyte
        "high_nc_ratio": [32, 0],               # nuclear_ratio, area (small cells)
        "round_shape": [2, 7, 8],               # circularity, eccentricity, solidity
        "dark_nucleus": [35, 31],               # dark_fraction, hist_skewness
        "thin_cytoplasm": [6, 30],              # std_intensity, hist_entropy
        "small_size": [0, 1],                    # area, perimeter
    },
    6: {  # Macrophage
        "large_size": [0, 1],
        "irregular_shape": [2, 7, 8, 10],       # circularity, eccentricity, solidity, aspect_ratio
        "vacuolated_cytoplasm": [21, 30, 6],    # texture_contrast, hist_entropy, std_intensity
        "eccentric_nucleus": [32, 33],
        "phagocytic_inclusions": [22, 27],      # granule_mean, n_granules
    },
}


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


def compute_attribute_scores(morph, support_morph, cids, gm, gs):
    """Compute attribute-level similarity scores."""
    qm = (morph - gm) / gs
    attr_scores = {}

    for c in cids:
        attrs = CELL_ATTRIBUTES[c]
        class_scores = []
        for attr_name, dims in attrs.items():
            valid_dims = [d for d in dims if d < len(qm)]
            if not valid_dims:
                continue
            q_attr = qm[valid_dims]
            s_attr = ((support_morph[c] - gm) / gs)[:, valid_dims]
            s_mean = s_attr.mean(0)
            dist = np.linalg.norm(q_attr - s_mean)
            class_scores.append(1.0 / (1.0 + dist))
        attr_scores[c] = float(np.mean(class_scores)) if class_scores else 0.0

    return attr_scores


def cls_attribute_guided(q_bclip, q_dino, q_morph, q_labels,
                          s_bclip, s_dino, s_morph,
                          cids, vw=0.35, dw=0.15, mw=0.25, aw=0.25, k=7):
    """Attribute-guided dual-backbone kNN."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        attr_scores = compute_attribute_scores(q_morph[i], s_morph, cids, gm, gs)

        scores = []
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = vw*vs_b + dw*vs_d + mw*ms
            visual_score = float(np.sort(comb)[::-1][:k].mean())
            scores.append(visual_score + aw * attr_scores[c])
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_attr_trans_cascade(q_bclip, q_dino, q_morph, q_labels,
                            s_bclip_init, s_dino_init, s_morph_init,
                            cids, morph_weights,
                            vw=0.35, dw=0.15, mw=0.25, aw=0.25, k=7,
                            n_iter=2, top_k=5, conf_thr=0.025, cascade_thr=0.01):
    """Attribute-guided + transductive + cascade."""
    sm_all_init = np.concatenate([s_morph_init[c] for c in cids])
    gm, gs = sm_all_init.mean(0), sm_all_init.std(0)+1e-8

    s_b = {c: s_bclip_init[c].copy() for c in cids}
    s_d = {c: s_dino_init[c].copy() for c in cids}
    s_m = {c: s_morph_init[c].copy() for c in cids}

    for _t in range(n_iter):
        snm = {c: (s_m[c]-gm)/gs for c in cids}
        preds, margins_a = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i]-gm)/gs
            attr_sc = compute_attribute_scores(q_morph[i], s_morph_init, cids, gm, gs)
            scores = []
            for c in cids:
                vs_b = s_b[c] @ q_bclip[i]
                vs_d = s_d[c] @ q_dino[i]
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0/(1.0+md)
                comb = vw*vs_b + dw*vs_d + mw*ms
                scores.append(float(np.sort(comb)[::-1][:k].mean()) + aw * attr_sc[c])
            s_arr = np.array(scores)
            sorted_s = np.sort(s_arr)[::-1]
            preds.append(cids[int(np.argmax(s_arr))])
            margins_a.append(sorted_s[0]-sorted_s[1])
        preds = np.array(preds)
        margins_a = np.array(margins_a)
        for c in cids:
            c_mask = (preds == c) & (margins_a > conf_thr)
            c_idx = np.where(c_mask)[0]
            if len(c_idx) == 0: continue
            sorted_idx = c_idx[np.argsort(margins_a[c_idx])[::-1][:top_k]]
            s_b[c] = np.concatenate([s_bclip_init[c], q_bclip[sorted_idx]*0.5])
            s_d[c] = np.concatenate([s_dino_init[c], q_dino[sorted_idx]*0.5])
            s_m[c] = np.concatenate([s_morph_init[c], q_morph[sorted_idx]])

    sm_all = np.concatenate([s_m[c] for c in cids])
    gm2, gs2 = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_m[c]-gm2)/gs2 for c in cids}
    snm_w = {c: (s_m[c]-gm2)/gs2 * morph_weights for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm2)/gs2
        qm_w = qm * morph_weights
        attr_sc = compute_attribute_scores(q_morph[i], s_morph_init, cids, gm, gs)
        scores = {}
        for c in cids:
            vs_b = s_b[c] @ q_bclip[i]
            vs_d = s_d[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = vw*vs_b + dw*vs_d + mw*ms
            scores[c] = float(np.sort(comb)[::-1][:k].mean()) + aw * attr_sc[c]
        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0]-np.sort(s_arr)[::-1][1]
        if top1 in [3, 4] and margin < cascade_thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(s_b[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(s_d[gc] @ q_dino[i])[::-1][:3].mean())
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

    eos, neu = morph_train[labels_train==3], morph_train[labels_train==4]
    n_dims = morph_train.shape[1]
    fisher_w = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos[:,d])-np.mean(neu[:,d]))**2 / (np.var(eos[:,d])+np.var(neu[:,d])+1e-10)
        fisher_w[d] = 1.0 + f * 2.0

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"Seed {seed}...")
        np.random.seed(seed)
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}

        # Attribute-guided sweep
        for aw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            vw = 0.45 - aw * 0.5
            dw = 0.20 - aw * 0.2
            mw = 1.0 - vw - dw - aw
            if mw < 0.1 or vw < 0.1: continue
            name = f"attr_aw{aw}_vw{vw:.2f}"
            m = cls_attribute_guided(bclip_val, dino_val, morph_val, labels_val,
                                      s_bclip, s_dino, s_morph, cids, vw, dw, mw, aw)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # Attr + transductive + cascade
        for aw in [0.05, 0.10, 0.15, 0.20]:
            for cthr in [0.008, 0.010, 0.012]:
                vw = 0.45 - aw * 0.4
                dw = 0.20 - aw * 0.2
                mw = 1.0 - vw - dw - aw
                if mw < 0.1: continue
                name = f"attr_tc_aw{aw}_ct{cthr}"
                m = cls_attr_trans_cascade(
                    bclip_val, dino_val, morph_val, labels_val,
                    s_bclip, s_dino, s_morph, cids, fisher_w,
                    vw, dw, mw, aw, cascade_thr=cthr)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

    print(f"\n{'='*130}")
    print("ATTRIBUTE-GUIDED RESULTS (5 seeds)")
    print(f"{'='*130}")
    header = f"{'Strategy':<50} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 130)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:20]:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}, Eos F1={np.mean(best[1]['pc'][3]):.4f}")


if __name__ == "__main__":
    main()
