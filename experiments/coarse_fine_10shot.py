#!/usr/bin/env python3
"""
Novel 10-shot strategies:
1. Coarse-to-Fine: granulocyte vs non-granulocyte then fine-grained
2. SimpleShot with L2/CL2N normalization
3. Class-balanced logit adjustment (prior calibration)
4. Hard negative mining for Eos during transductive iteration
5. Multi-prototype per class (k-means intra-class clustering)
"""
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


def cls_coarse_fine(q_bclip, q_dino, q_morph, q_labels,
                     s_bclip, s_dino, s_morph, cids, fisher_w,
                     coarse_vw=0.60, coarse_mw=0.40,
                     fine_vw=0.40, fine_mw=0.60, k=7):
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    snm = {c: (s_morph[c] - gm) / gs for c in cids}
    snm_w = {c: snm[c] * fisher_w for c in cids}
    GRAN = [3, 4]
    NON_GRAN = [5, 6]
    s_b_gran = np.concatenate([s_bclip[c] for c in GRAN])
    s_b_ng = np.concatenate([s_bclip[c] for c in NON_GRAN])
    s_m_gran = np.concatenate([((s_morph[c]-gm)/gs) for c in GRAN])
    s_m_ng = np.concatenate([((s_morph[c]-gm)/gs) for c in NON_GRAN])

    gt, pred = [], []
    for i in range(len(q_labels)):
        qb, qd = q_bclip[i], q_dino[i]
        qm = (q_morph[i] - gm) / gs
        vs_gran = np.sort(s_b_gran @ qb)[::-1][:10].mean()
        vs_ng = np.sort(s_b_ng @ qb)[::-1][:10].mean()
        md_gran = np.linalg.norm(s_m_gran - qm, axis=1)
        md_ng = np.linalg.norm(s_m_ng - qm, axis=1)
        ms_gran = np.sort(1.0/(1.0+md_gran))[::-1][:10].mean()
        ms_ng = np.sort(1.0/(1.0+md_ng))[::-1][:10].mean()
        if coarse_vw * vs_gran + coarse_mw * ms_gran > coarse_vw * vs_ng + coarse_mw * ms_ng:
            group = GRAN
        else:
            group = NON_GRAN
        scores = {}
        for c in group:
            vs_b = s_bclip[c] @ qb
            vs_d = s_dino[c] @ qd
            if c in GRAN:
                md = np.linalg.norm((qm * fisher_w) - snm_w[c], axis=1)
            else:
                md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = fine_vw * (0.7*vs_b + 0.3*vs_d) + fine_mw * ms
            scores[c] = float(np.sort(comb)[::-1][:k].mean())
        gt.append(int(q_labels[i]))
        pred.append(max(scores, key=scores.get))
    return metrics(gt, pred, cids)


def cls_logit_adjusted(q_bclip, q_dino, q_morph, q_labels,
                        s_bclip, s_dino, s_morph, cids,
                        class_priors, tau=1.0, vw=0.50, dw=0.15, mw=0.35, k=7):
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    snm = {c: (s_morph[c] - gm) / gs for c in cids}
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm) / gs
        scores = []
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = vw*vs_b + dw*vs_d + mw*ms
            base = float(np.sort(comb)[::-1][:k].mean())
            scores.append(base + tau * np.log(class_priors[c] + 1e-10))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_hn_transductive(q_bclip, q_dino, q_morph, q_labels,
                         s_bclip_init, s_dino_init, s_morph_init,
                         cids, fisher_w,
                         vw=0.50, dw=0.15, mw=0.35, k=7,
                         n_iter=3, top_k=5, conf_thr=0.025, cascade_thr=0.010):
    sm_all = np.concatenate([s_morph_init[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    s_b = {c: s_bclip_init[c].copy() for c in cids}
    s_d = {c: s_dino_init[c].copy() for c in cids}
    s_m = {c: s_morph_init[c].copy() for c in cids}
    for _t in range(n_iter):
        snm = {c: (s_m[c] - gm) / gs for c in cids}
        preds, margins = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i] - gm) / gs
            scores = []
            for c in cids:
                vs_b = s_b[c] @ q_bclip[i]
                vs_d = s_d[c] @ q_dino[i]
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0/(1.0+md)
                comb = vw*vs_b + dw*vs_d + mw*ms
                scores.append(float(np.sort(comb)[::-1][:k].mean()))
            s_arr = np.array(scores)
            sorted_s = np.sort(s_arr)[::-1]
            preds.append(cids[int(np.argmax(s_arr))])
            margins.append(sorted_s[0]-sorted_s[1])
        preds = np.array(preds)
        margins = np.array(margins)
        for c in cids:
            c_mask = (preds == c) & (margins > conf_thr)
            c_idx = np.where(c_mask)[0]
            if len(c_idx) == 0: continue
            top_idx = c_idx[np.argsort(margins[c_idx])[::-1][:top_k]]
            s_b[c] = np.concatenate([s_bclip_init[c], q_bclip[top_idx]*0.5])
            s_d[c] = np.concatenate([s_dino_init[c], q_dino[top_idx]*0.5])
            s_m[c] = np.concatenate([s_morph_init[c], q_morph[top_idx]])

    snm = {c: (s_m[c] - gm) / gs for c in cids}
    snm_w = {c: snm[c] * fisher_w for c in cids}
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm) / gs
        qm_w = qm * fisher_w
        scores = {}
        for c in cids:
            vs_b = s_b[c] @ q_bclip[i]
            vs_d = s_d[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = vw*vs_b + dw*vs_d + mw*ms
            scores[c] = float(np.sort(comb)[::-1][:k].mean())
        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0] - np.sort(s_arr)[::-1][1]
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
    n_dims = morph_train.shape[1]
    eos, neu = morph_train[labels_train == 3], morph_train[labels_train == 4]
    fisher_w = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos[:, d]) - np.mean(neu[:, d]))**2 / (np.var(eos[:, d]) + np.var(neu[:, d]) + 1e-10)
        fisher_w[d] = 1.0 + f * 2.0
    total_train = len(labels_train)
    class_priors = {c: np.sum(labels_train == c) / total_train for c in cids}
    print(f"Class priors: {class_priors}")
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"Seed {seed}...")
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}

        # Coarse-to-fine
        for cv, cm in [(0.55, 0.45), (0.50, 0.50)]:
            for fv, fm in [(0.35, 0.65), (0.40, 0.60), (0.30, 0.70)]:
                name = f"cf_cv{cv}_fv{fv}"
                try:
                    m = cls_coarse_fine(bclip_val, dino_val, morph_val, labels_val,
                                        s_bclip, s_dino, s_morph, cids, fisher_w, cv, cm, fv, fm)
                    all_results[name]["acc"].append(m["acc"])
                    all_results[name]["mf1"].append(m["mf1"])
                    for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])
                except Exception as e:
                    print(f"  {name} FAILED: {e}")

        # Logit adjustment
        for tau in [0.005, 0.01, 0.02, 0.05, 0.1]:
            name = f"logit_adj_tau{tau}"
            try:
                m = cls_logit_adjusted(bclip_val, dino_val, morph_val, labels_val,
                                        s_bclip, s_dino, s_morph, cids, class_priors, tau)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])
            except Exception as e:
                print(f"  {name} FAILED: {e}")

        # HN transductive variants
        for ni in [2, 3, 4]:
            for ct in [0.008, 0.010, 0.012, 0.015]:
                for cf in [0.020, 0.025, 0.030]:
                    name = f"hn_i{ni}_ct{ct}_cf{cf}"
                    try:
                        m = cls_hn_transductive(bclip_val, dino_val, morph_val, labels_val,
                                                s_bclip, s_dino, s_morph, cids, fisher_w,
                                                n_iter=ni, cascade_thr=ct, conf_thr=cf)
                        all_results[name]["acc"].append(m["acc"])
                        all_results[name]["mf1"].append(m["mf1"])
                        for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])
                    except Exception as e:
                        print(f"  {name} FAILED: {e}")

    print(f"\n{'='*130}")
    print("COARSE-FINE + NOVEL STRATEGIES (5 seeds)")
    print(f"{'='*130}")
    header = f"{'Strategy':<45} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 130)
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:30]:
        if len(v["acc"]) < 3: continue
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<45} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    best = sorted_r[0]
    print(f"\nBEST: {best[0]} -> mF1={np.mean(best[1]['mf1']):.4f}, "
          f"Eos={np.mean(best[1]['pc'][3]):.4f}")


if __name__ == "__main__":
    main()
