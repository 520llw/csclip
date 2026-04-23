#!/usr/bin/env python3
"""
PCA + Fisher feature selection for 10-shot classification.
Reduces high-dimensional visual features to remove noise and improve kNN.

Approaches:
1. PCA on BiomedCLIP features (512→K)
2. Fisher-weighted kNN on ALL dimensions (visual + morph)
3. PCA + Fisher + transductive + cascade
4. Distribution calibration: Gaussian class modeling
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA

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


def compute_fisher_weights(s_feats_dict, cids, n_dims):
    """Multi-class Fisher discriminant weights for each dimension."""
    weights = np.ones(n_dims, dtype=np.float32)
    for d in range(n_dims):
        all_vals = np.concatenate([s_feats_dict[c][:, d] for c in cids])
        grand_mean = np.mean(all_vals)
        between_var = sum(
            len(s_feats_dict[c]) * (np.mean(s_feats_dict[c][:, d]) - grand_mean)**2
            for c in cids
        ) / len(all_vals)
        within_var = sum(
            np.var(s_feats_dict[c][:, d]) * len(s_feats_dict[c])
            for c in cids
        ) / len(all_vals)
        weights[d] = between_var / (within_var + 1e-10)
    return weights


def cls_pca_knn(q_bclip, q_dino, q_morph, q_labels,
                s_bclip, s_dino, s_morph,
                cids, n_components=64, bw=0.45, dw=0.20, mw=0.35, k=7):
    """PCA-reduced kNN."""
    all_s_b = np.concatenate([s_bclip[c] for c in cids])
    all_s_d = np.concatenate([s_dino[c] for c in cids])

    pca_b = PCA(n_components=min(n_components, len(all_s_b)-1, all_s_b.shape[1]))
    pca_b.fit(all_s_b)
    pca_d = PCA(n_components=min(n_components, len(all_s_d)-1, all_s_d.shape[1]))
    pca_d.fit(all_s_d)

    sb_pca = {c: pca_b.transform(s_bclip[c]) for c in cids}
    sd_pca = {c: pca_d.transform(s_dino[c]) for c in cids}
    qb_pca = pca_b.transform(q_bclip)
    qd_pca = pca_d.transform(q_dino)

    for c in cids:
        sb_pca[c] = sb_pca[c] / (np.linalg.norm(sb_pca[c], axis=1, keepdims=True) + 1e-10)
        sd_pca[c] = sd_pca[c] / (np.linalg.norm(sd_pca[c], axis=1, keepdims=True) + 1e-10)
    qb_pca = qb_pca / (np.linalg.norm(qb_pca, axis=1, keepdims=True) + 1e-10)
    qd_pca = qd_pca / (np.linalg.norm(qd_pca, axis=1, keepdims=True) + 1e-10)

    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        scores = []
        for c in cids:
            vs_b = sb_pca[c] @ qb_pca[i]
            vs_d = sd_pca[c] @ qd_pca[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_fisher_weighted_knn(q_bclip, q_dino, q_morph, q_labels,
                             s_bclip, s_dino, s_morph,
                             cids, k=7, top_pct=0.3):
    """Fisher-weighted kNN on concatenated features."""
    s_concat = {}
    for c in cids:
        sm_all_tmp = np.concatenate([s_morph[cc] for cc in cids])
        gm, gs = sm_all_tmp.mean(0), sm_all_tmp.std(0)+1e-8
        norm_morph = (s_morph[c]-gm)/gs
        s_concat[c] = np.concatenate([s_bclip[c], s_dino[c], norm_morph], axis=1)

    fisher_w = compute_fisher_weights(s_concat, cids, s_concat[cids[0]].shape[1])

    top_n = max(10, int(len(fisher_w) * top_pct))
    top_dims = np.argsort(fisher_w)[::-1][:top_n]
    s_sel = {c: s_concat[c][:, top_dims] for c in cids}

    sm_all_tmp = np.concatenate([s_morph[cc] for cc in cids])
    gm, gs = sm_all_tmp.mean(0), sm_all_tmp.std(0)+1e-8

    gt, pred = [], []
    for i in range(len(q_labels)):
        norm_morph = (q_morph[i]-gm)/gs
        q_full = np.concatenate([q_bclip[i], q_dino[i], norm_morph])
        q_sel = q_full[top_dims]

        scores = []
        for c in cids:
            dists = np.linalg.norm(s_sel[c] - q_sel, axis=1)
            scores.append(-float(np.sort(dists)[:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_gaussian_calibration(q_bclip, q_dino, q_morph, q_labels,
                              s_bclip, s_dino, s_morph,
                              cids, alpha=0.5):
    """Gaussian distribution calibration per class."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8

    s_feats = {}
    for c in cids:
        norm_morph = (s_morph[c]-gm)/gs
        s_feats[c] = np.concatenate([s_bclip[c], s_dino[c], norm_morph * 0.5], axis=1)

    class_means = {c: np.mean(s_feats[c], axis=0) for c in cids}
    global_feats = np.concatenate([s_feats[c] for c in cids])
    global_cov = np.cov(global_feats.T) + alpha * np.eye(global_feats.shape[1])

    class_covs = {}
    for c in cids:
        if len(s_feats[c]) > 1:
            class_cov = np.cov(s_feats[c].T)
        else:
            class_cov = np.zeros_like(global_cov)
        class_covs[c] = 0.5 * class_cov + 0.5 * global_cov + alpha * np.eye(global_feats.shape[1])

    try:
        cov_invs = {c: np.linalg.inv(class_covs[c]) for c in cids}
    except np.linalg.LinAlgError:
        cov_invs = {c: np.eye(global_feats.shape[1]) for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        norm_morph = (q_morph[i]-gm)/gs
        q_feat = np.concatenate([q_bclip[i], q_dino[i], norm_morph * 0.5])

        log_probs = []
        for c in cids:
            diff = q_feat - class_means[c]
            d = float(diff @ cov_invs[c] @ diff)
            log_probs.append(-0.5 * d)
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(log_probs))])
    return metrics(gt, pred, cids)


def cls_pca_trans_cascade(q_bclip, q_dino, q_morph, q_labels,
                           s_bclip, s_dino, s_morph,
                           cids, morph_weights, n_components=64,
                           n_iter=2, top_k=5, conf_thr=0.025, cascade_thr=0.01, k=7):
    """PCA + transductive + cascade."""
    all_s_b = np.concatenate([s_bclip[c] for c in cids])
    all_s_d = np.concatenate([s_dino[c] for c in cids])
    pca_b = PCA(n_components=min(n_components, len(all_s_b)-1, all_s_b.shape[1]))
    pca_b.fit(all_s_b)
    pca_d = PCA(n_components=min(n_components, len(all_s_d)-1, all_s_d.shape[1]))
    pca_d.fit(all_s_d)

    def to_pca(feats_dict):
        return {c: pca_b.transform(feats_dict[c]) / (np.linalg.norm(pca_b.transform(feats_dict[c]),axis=1,keepdims=True)+1e-10) for c in cids}

    def to_pca_d(feats_dict):
        return {c: pca_d.transform(feats_dict[c]) / (np.linalg.norm(pca_d.transform(feats_dict[c]),axis=1,keepdims=True)+1e-10) for c in cids}

    sb_pca = to_pca(s_bclip)
    sd_pca = to_pca_d(s_dino)
    qb_pca = pca_b.transform(q_bclip)
    qb_pca = qb_pca / (np.linalg.norm(qb_pca, axis=1, keepdims=True) + 1e-10)
    qd_pca = pca_d.transform(q_dino)
    qd_pca = qd_pca / (np.linalg.norm(qd_pca, axis=1, keepdims=True) + 1e-10)

    sb_work = {c: sb_pca[c].copy() for c in cids}
    sd_work = {c: sd_pca[c].copy() for c in cids}
    sm_work = {c: s_morph[c].copy() for c in cids}

    bw, dw, mw = 0.45, 0.20, 0.35

    for _t in range(n_iter):
        sm_all = np.concatenate([sm_work[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
        snm = {c: (sm_work[c]-gm)/gs for c in cids}
        preds, margins_a = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i]-gm)/gs
            scores = []
            for c in cids:
                vs_b = sb_work[c] @ qb_pca[i]
                vs_d = sd_work[c] @ qd_pca[i]
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0/(1.0+md)
                comb = bw*vs_b + dw*vs_d + mw*ms
                scores.append(float(np.sort(comb)[::-1][:k].mean()))
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
            sb_work[c] = np.concatenate([sb_pca[c], qb_pca[sorted_idx]*0.5])
            sd_work[c] = np.concatenate([sd_pca[c], qd_pca[sorted_idx]*0.5])
            sm_work[c] = np.concatenate([s_morph[c], q_morph[sorted_idx]])

    sm_all = np.concatenate([sm_work[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (sm_work[c]-gm)/gs for c in cids}
    snm_w = {c: (sm_work[c]-gm)/gs * morph_weights for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        qm_w = qm * morph_weights
        scores = {}
        for c in cids:
            vs_b = sb_work[c] @ qb_pca[i]
            vs_d = sd_work[c] @ qd_pca[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores[c] = float(np.sort(comb)[::-1][:k].mean())
        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0]-np.sort(s_arr)[::-1][1]
        if top1 in [3, 4] and margin < cascade_thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(sb_work[gc] @ qb_pca[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(sd_work[gc] @ qd_pca[i])[::-1][:3].mean())
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

        # 1) PCA sweep
        for nc in [16, 32, 39, 50, 64, 100, 128]:
            name = f"pca_{nc}"
            m = cls_pca_knn(bclip_val, dino_val, morph_val, labels_val,
                             s_bclip, s_dino, s_morph, cids, nc)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 2) Fisher feature selection sweep
        for top_pct in [0.1, 0.2, 0.3, 0.5, 0.7]:
            name = f"fisher_sel_{top_pct}"
            m = cls_fisher_weighted_knn(bclip_val, dino_val, morph_val, labels_val,
                                         s_bclip, s_dino, s_morph, cids, top_pct=top_pct)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 3) Gaussian calibration
        for alpha in [0.01, 0.1, 0.5, 1.0, 5.0]:
            name = f"gauss_cal_a{alpha}"
            m = cls_gaussian_calibration(bclip_val, dino_val, morph_val, labels_val,
                                          s_bclip, s_dino, s_morph, cids, alpha)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 4) PCA + transductive + cascade
        for nc in [32, 39, 50, 64]:
            name = f"pca_tc_{nc}"
            m = cls_pca_trans_cascade(bclip_val, dino_val, morph_val, labels_val,
                                       s_bclip, s_dino, s_morph, cids, fisher_w, nc)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

    print(f"\n{'='*130}")
    print("PCA / FISHER / GAUSSIAN RESULTS (5 seeds)")
    print(f"{'='*130}")
    header = f"{'Strategy':<50} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 130)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}")


if __name__ == "__main__":
    main()
