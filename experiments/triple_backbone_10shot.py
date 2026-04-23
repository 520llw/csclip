#!/usr/bin/env python3
"""
Triple-backbone 10-shot: BiomedCLIP + Phikon-v2 + DINOv2 + morphology
+ transductive inference + cascade.

Phikon-v2 (pathology ViT-L, 1024d) provides pathology-specific features
that complement BiomedCLIP (medical VL, 512d) and DINOv2-S (generic, 384d).
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


def triple_trans_cascade(q_bc, q_ph, q_dn, q_morph, q_labels,
                          s_bc_init, s_ph_init, s_dn_init, s_morph_init,
                          cids, fisher_w,
                          bw=0.40, pw=0.25, dw=0.00, mw=0.35, k=7,
                          n_iter=2, top_k=5, conf_thr=0.025, cascade_thr=0.010,
                          pseudo_w=0.5, cascade_mw=0.55):
    sm_all = np.concatenate([s_morph_init[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    s_bc = {c: s_bc_init[c].copy() for c in cids}
    s_ph = {c: s_ph_init[c].copy() for c in cids}
    s_dn = {c: s_dn_init[c].copy() for c in cids} if dw > 0 else None
    s_m = {c: s_morph_init[c].copy() for c in cids}

    for _t in range(n_iter):
        snm = {c: (s_m[c]-gm)/gs for c in cids}
        preds, margins = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i]-gm)/gs
            scores = []
            for c in cids:
                vs_b = s_bc[c] @ q_bc[i]
                vs_p = s_ph[c] @ q_ph[i]
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0/(1.0+md)
                comb = bw*vs_b + pw*vs_p + mw*ms
                if dw > 0 and s_dn is not None:
                    vs_d = s_dn[c] @ q_dn[i]
                    comb += dw*vs_d
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
            s_bc[c] = np.concatenate([s_bc_init[c], q_bc[top_idx]*pseudo_w])
            s_ph[c] = np.concatenate([s_ph_init[c], q_ph[top_idx]*pseudo_w])
            if s_dn is not None:
                s_dn[c] = np.concatenate([s_dn_init[c], q_dn[top_idx]*pseudo_w])
            s_m[c] = np.concatenate([s_morph_init[c], q_morph[top_idx]])

    sm_all2 = np.concatenate([s_m[c] for c in cids])
    gm2, gs2 = sm_all2.mean(0), sm_all2.std(0)+1e-8
    snm = {c: (s_m[c]-gm2)/gs2 for c in cids}
    snm_w = {c: snm[c] * fisher_w for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm2)/gs2
        qm_w = qm * fisher_w
        scores = {}
        for c in cids:
            vs_b = s_bc[c] @ q_bc[i]
            vs_p = s_ph[c] @ q_ph[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + pw*vs_p + mw*ms
            if dw > 0 and s_dn is not None:
                vs_d = s_dn[c] @ q_dn[i]
                comb += dw*vs_d
            scores[c] = float(np.sort(comb)[::-1][:k].mean())
        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0] - np.sort(s_arr)[::-1][1]
        if top1 in [3, 4] and margin < cascade_thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(s_bc[gc] @ q_bc[i])[::-1][:3].mean())
                vs_p_s = float(np.sort(s_ph[gc] @ q_ph[i])[::-1][:3].mean())
                scores[gc] = 0.25*vs_b_s + 0.20*vs_p_s + cascade_mw*mscore
            top1 = 3 if scores[3] > scores[4] else 4
        gt.append(int(q_labels[i]))
        pred.append(top1)
    return metrics(gt, pred, cids)


def main():
    bc_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bc_val, morph_val, labels_val = load_cache("biomedclip", "val")
    ph_train, _, _ = load_cache("phikon_v2", "train")
    ph_val, _, _ = load_cache("phikon_v2", "val")
    dn_train, _, _ = load_cache("dinov2_s", "train")
    dn_val, _, _ = load_cache("dinov2_s", "val")

    cids = sorted(CLASS_NAMES.keys())
    n_dims = morph_train.shape[1]
    eos, neu = morph_train[labels_train == 3], morph_train[labels_train == 4]
    fisher_w = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos[:,d])-np.mean(neu[:,d]))**2 / (np.var(eos[:,d])+np.var(neu[:,d])+1e-10)
        fisher_w[d] = 1.0 + f * 2.0

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"Seed {seed}...")
        support_idx = select_support(labels_train, seed, cids)
        s_bc = {c: bc_train[support_idx[c]] for c in cids}
        s_ph = {c: ph_train[support_idx[c]] for c in cids}
        s_dn = {c: dn_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}

        # BiomedCLIP + Phikon (no DINOv2)
        configs = [
            # (name, bw, pw, dw, mw, k, conf, cthr, cas_mw)
            ("bp_40_25_35_tc", 0.40, 0.25, 0.0, 0.35, 7, 0.025, 0.010, 0.55),
            ("bp_40_25_35_tc_cm45", 0.40, 0.25, 0.0, 0.35, 7, 0.025, 0.010, 0.45),
            ("bp_35_30_35_tc", 0.35, 0.30, 0.0, 0.35, 7, 0.025, 0.010, 0.55),
            ("bp_35_25_40_tc", 0.35, 0.25, 0.0, 0.40, 7, 0.025, 0.010, 0.55),
            ("bp_45_20_35_tc", 0.45, 0.20, 0.0, 0.35, 7, 0.025, 0.010, 0.55),
            ("bp_38_27_35_tc", 0.38, 0.27, 0.0, 0.35, 7, 0.025, 0.010, 0.55),
            ("bp_42_23_35_tc", 0.42, 0.23, 0.0, 0.35, 7, 0.025, 0.010, 0.55),
            ("bp_40_25_35_k5", 0.40, 0.25, 0.0, 0.35, 5, 0.025, 0.010, 0.55),
            ("bp_40_25_35_cf020", 0.40, 0.25, 0.0, 0.35, 7, 0.020, 0.010, 0.55),
            ("bp_40_25_35_cf030", 0.40, 0.25, 0.0, 0.35, 7, 0.030, 0.010, 0.55),
            ("bp_40_25_35_ct008", 0.40, 0.25, 0.0, 0.35, 7, 0.025, 0.008, 0.55),
            ("bp_40_25_35_ct012", 0.40, 0.25, 0.0, 0.35, 7, 0.025, 0.012, 0.55),
            ("bp_40_25_35_i3", 0.40, 0.25, 0.0, 0.35, 7, 0.025, 0.010, 0.55),
            # Triple: BiomedCLIP + Phikon + DINOv2
            ("bpd_35_22_08_35", 0.35, 0.22, 0.08, 0.35, 7, 0.025, 0.010, 0.55),
            ("bpd_38_20_07_35", 0.38, 0.20, 0.07, 0.35, 7, 0.025, 0.010, 0.55),
            ("bpd_35_25_05_35", 0.35, 0.25, 0.05, 0.35, 7, 0.025, 0.010, 0.55),
            ("bpd_40_20_05_35", 0.40, 0.20, 0.05, 0.35, 7, 0.025, 0.010, 0.55),
            ("bpd_30_25_10_35", 0.30, 0.25, 0.10, 0.35, 7, 0.025, 0.010, 0.55),
            # Morph weight variations
            ("bp_40_20_40_tc", 0.40, 0.20, 0.0, 0.40, 7, 0.025, 0.010, 0.55),
            ("bp_37_23_40_tc", 0.37, 0.23, 0.0, 0.40, 7, 0.025, 0.010, 0.55),
            ("bp_42_25_33_tc", 0.42, 0.25, 0.0, 0.33, 7, 0.025, 0.010, 0.55),
        ]

        for name, bw, pw, dw, mw, k, cf, ct, cmw in configs:
            ni = 3 if "i3" in name else 2
            try:
                m = triple_trans_cascade(
                    bc_val, ph_val, dn_val, morph_val, labels_val,
                    s_bc, s_ph, s_dn, s_morph, cids, fisher_w,
                    bw=bw, pw=pw, dw=dw, mw=mw, k=k,
                    n_iter=ni, conf_thr=cf, cascade_thr=ct, cascade_mw=cmw)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[name]["pc"][c].append(m["pc"][c]["f1"])
            except Exception as e:
                print(f"  {name} FAILED: {e}")

    print(f"\n{'='*130}")
    print("TRIPLE BACKBONE + TRANSDUCTIVE + CASCADE (5 seeds)")
    print(f"{'='*130}")
    header = f"{'Strategy':<45} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 130)
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:25]:
        if len(v["acc"]) < 3: continue
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<45} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\n*** BEST: {best[0]} ***")
    print(f"  mF1={np.mean(best[1]['mf1']):.4f}, Eos={np.mean(best[1]['pc'][3]):.4f}, "
          f"Acc={np.mean(best[1]['acc']):.4f}")
    print(f"\nPrevious best (trans+cas dual-bb): mF1=0.7376, Eos=0.3917, Acc=0.8616")
    improvement = np.mean(best[1]['mf1']) - 0.7376
    print(f"Improvement: +{improvement*100:.2f}% mF1")


if __name__ == "__main__":
    main()
