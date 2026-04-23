#!/usr/bin/env python3
"""
MultiCenter classification using its own train set for support selection.

Previous approach: Use data2's train set → severe domain shift → mF1=0.3189
This approach: Use MultiCenter's own train set → eliminate cross-domain bias

Also test on data2 with its own train (should match previous results).
"""
import sys, random
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
    result = {}
    for c in cids:
        available = pc.get(c, [])
        if len(available) == 0:
            result[c] = []
        else:
            result[c] = random.sample(available, min(n_shot, len(available)))
    return result


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
        pc[c] = {"p": pr, "r": rc, "f1": f1, "n": gp}
        f1s.append(f1)
    return {"acc": correct / total, "mf1": np.mean(f1s), "pc": pc}


# ==================== Classification Methods ====================

def knn_classify(q_feats, s_feats_per_class, cids, k=7):
    """Simple kNN prototype matching."""
    preds = []
    for i in range(len(q_feats)):
        scores = {}
        for c in cids:
            if len(s_feats_per_class[c]) == 0:
                scores[c] = -1e9
                continue
            sims = s_feats_per_class[c] @ q_feats[i]
            scores[c] = float(np.sort(sims)[::-1][:min(k, len(sims))].mean())
        preds.append(max(scores, key=scores.get))
    return preds


def multi_backbone_classify(q_bc, q_ph, q_dn, q_morph, s_bc, s_ph, s_dn, s_morph,
                             cids, bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7):
    """Multi-backbone fusion classification with morphology."""
    sm_all = np.concatenate([s_morph[c] for c in cids if len(s_morph[c]) > 0])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    snm = {}
    for c in cids:
        if len(s_morph[c]) > 0:
            snm[c] = (s_morph[c] - gm) / gs
        else:
            snm[c] = np.zeros((0, sm_all.shape[1]))

    preds, margins = [], []
    for i in range(len(q_bc)):
        qm = (q_morph[i] - gm) / gs
        scores = []
        for c in cids:
            if len(s_bc[c]) == 0:
                scores.append(-1e9)
                continue
            vs = bw * (s_bc[c] @ q_bc[i]) + pw * (s_ph[c] @ q_ph[i]) + dw * (s_dn[c] @ q_dn[i])
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            scores.append(float(np.sort(vs + mw * ms)[::-1][:min(k, len(vs))].mean()))
        sa = np.array(scores)
        ss = np.sort(sa)[::-1]
        preds.append(cids[int(np.argmax(sa))])
        margins.append(ss[0] - ss[1])
    return preds, margins


def multi_backbone_atd(q_bc, q_ph, q_dn, q_morph, q_labels,
                       s_bc0, s_ph0, s_dn0, s_morph0, cids,
                       bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7,
                       n_iter=2, top_k_pseudo=5, conf_thr=0.025):
    """Multi-backbone + ATD (from SADC v3, adapted for flexible class counts)."""
    sb = {c: s_bc0[c].copy() for c in cids}
    sp = {c: s_ph0[c].copy() for c in cids}
    sd = {c: s_dn0[c].copy() for c in cids}
    smm = {c: s_morph0[c].copy() for c in cids}

    sb_orig = {c: s_bc0[c].copy() for c in cids}
    sp_orig = {c: s_ph0[c].copy() for c in cids}
    sd_orig = {c: s_dn0[c].copy() for c in cids}
    smm_orig = {c: s_morph0[c].copy() for c in cids}

    for it in range(n_iter):
        sm_all = np.concatenate([smm[c] for c in cids if len(smm[c]) > 0])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        snm = {}
        for c in cids:
            if len(smm[c]) > 0:
                snm[c] = (smm[c] - gm) / gs
            else:
                snm[c] = np.zeros((0, sm_all.shape[1]))

        preds, margins = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i] - gm) / gs
            scores = []
            for c in cids:
                if len(sb[c]) == 0:
                    scores.append(-1e9)
                    continue
                vs = bw * (sb[c] @ q_bc[i]) + pw * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                scores.append(float(np.sort(vs + mw * ms)[::-1][:min(k, len(vs))].mean()))
            sa = np.array(scores)
            ss = np.sort(sa)[::-1]
            preds.append(cids[int(np.argmax(sa))])
            margins.append(ss[0] - ss[1])

        preds = np.array(preds)
        margins = np.array(margins)

        for c in cids:
            if len(sb_orig[c]) == 0:
                continue
            cm = (preds == c) & (margins > conf_thr)
            ci = np.where(cm)[0]
            if len(ci) == 0:
                continue

            proto_c = sb_orig[c].mean(0)
            dists = np.array([np.linalg.norm(q_bc[idx] - proto_c) for idx in ci])
            diversity = margins[ci] * (1.0 + 0.3 * dists / (dists.mean() + 1e-8))
            ti = ci[np.argsort(diversity)[::-1][:top_k_pseudo]]

            sb[c] = np.concatenate([sb_orig[c], q_bc[ti] * 0.5])
            sp[c] = np.concatenate([sp_orig[c], q_ph[ti] * 0.5])
            sd[c] = np.concatenate([sd_orig[c], q_dn[ti] * 0.5])
            smm[c] = np.concatenate([smm_orig[c], q_morph[ti]])

    sm2 = np.concatenate([smm[c] for c in cids if len(smm[c]) > 0])
    gm2, gs2 = sm2.mean(0), sm2.std(0) + 1e-8

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm2) / gs2
        scores = {}
        for c in cids:
            if len(sb[c]) == 0:
                scores[c] = -1e9
                continue
            snm_c = (smm[c] - gm2) / gs2
            vs = bw * (sb[c] @ q_bc[i]) + pw * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
            md = np.linalg.norm(qm - snm_c, axis=1)
            ms = 1.0 / (1.0 + md)
            scores[c] = float(np.sort(vs + mw * ms)[::-1][:min(k, len(vs))].mean())
        sa = np.array([scores[c] for c in cids])
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(sa))])

    return metrics(gt, pred, cids)


def print_row(name, v, cids):
    pc = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    counts = " ".join(f"({v.get('n_support', {}).get(c, '?'):>3})" for c in cids)
    print(f"{name:<55} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc}  "
          f"{np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}  {counts}")


def run_dataset(dataset_name, prefix, cids):
    print(f"\n{'='*140}")
    print(f"Dataset: {dataset_name} (prefix='{prefix}')")
    print(f"{'='*140}", flush=True)

    bc_t, mt, lt = load_cache("biomedclip", "train", prefix)
    bc_v, mv, lv = load_cache("biomedclip", "val", prefix)
    ph_t, _, _ = load_cache("phikon_v2", "train", prefix)
    ph_v, _, _ = load_cache("phikon_v2", "val", prefix)
    dn_t, _, _ = load_cache("dinov2_s", "train", prefix)
    dn_v, _, _ = load_cache("dinov2_s", "val", prefix)

    unique_labels = np.unique(lt)
    print(f"Train labels: {dict(zip(*np.unique(lt, return_counts=True)))}")
    print(f"Val labels:   {dict(zip(*np.unique(lv, return_counts=True)))}")

    active_cids = [c for c in cids if c in unique_labels]
    print(f"Active classes: {[CLASS_NAMES.get(c, c) for c in active_cids]}")
    skip_cids = [c for c in cids if c not in unique_labels]
    if skip_cids:
        print(f"WARNING: Classes {skip_cids} not in train set!")

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list), "n_support": {}})

    for seed in SEEDS:
        si = select_support(lt, seed, active_cids)
        sbc = {c: bc_t[si[c]] if si[c] else np.zeros((0, bc_t.shape[1])) for c in active_cids}
        sph = {c: ph_t[si[c]] if si[c] else np.zeros((0, ph_t.shape[1])) for c in active_cids}
        sdn = {c: dn_t[si[c]] if si[c] else np.zeros((0, dn_t.shape[1])) for c in active_cids}
        sm = {c: mt[si[c]] if si[c] else np.zeros((0, mt.shape[1])) for c in active_cids}

        for c in active_cids:
            all_results["_"]["n_support"][c] = len(si[c])

        # Method 1: Simple kNN on BiomedCLIP
        pred_knn = knn_classify(bc_v, sbc, active_cids)
        m_knn = metrics([int(l) for l in lv], pred_knn, active_cids)
        all_results["kNN_BC"]["acc"].append(m_knn["acc"])
        all_results["kNN_BC"]["mf1"].append(m_knn["mf1"])
        for c in active_cids:
            all_results["kNN_BC"]["pc"][c].append(m_knn["pc"][c]["f1"])

        # Method 2: Multi-backbone fusion
        pred_mb, _ = multi_backbone_classify(bc_v, ph_v, dn_v, mv, sbc, sph, sdn, sm, active_cids)
        m_mb = metrics([int(l) for l in lv], pred_mb, active_cids)
        all_results["MultiBackbone"]["acc"].append(m_mb["acc"])
        all_results["MultiBackbone"]["mf1"].append(m_mb["mf1"])
        for c in active_cids:
            all_results["MultiBackbone"]["pc"][c].append(m_mb["pc"][c]["f1"])

        # Method 3: Multi-backbone + ATD
        m_atd = multi_backbone_atd(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, active_cids)
        all_results["MultiBackbone+ATD"]["acc"].append(m_atd["acc"])
        all_results["MultiBackbone+ATD"]["mf1"].append(m_atd["mf1"])
        for c in active_cids:
            all_results["MultiBackbone+ATD"]["pc"][c].append(m_atd["pc"][c]["f1"])

        print(f"  Seed {seed}: kNN={m_knn['mf1']:.4f} MB={m_mb['mf1']:.4f} ATD={m_atd['mf1']:.4f}", flush=True)

    print(f"\n{'='*140}")
    print(f"RESULTS — {dataset_name}")
    print(f"{'='*140}")
    h = f"{'Strategy':<55} {'Acc':>7} {'mF1':>7} " + " ".join(f"{'F1_'+CLASS_NAMES.get(c,'?')[:3]:>7}" for c in active_cids) + f"  {'As':>5} {'Fs':>5}  Support"
    print(h)
    print("-" * 140)
    for n in ["kNN_BC", "MultiBackbone", "MultiBackbone+ATD"]:
        v = all_results[n]
        if not v["acc"]:
            continue
        print_row(n, v, active_cids)

    return all_results


def main():
    cids = sorted(CLASS_NAMES.keys())

    print("="*80)
    print("MultiCenter Own-Support Classification")
    print("Previous MC mF1 (data2 support): 0.3189")
    print("="*80)

    # MultiCenter with its own support
    mc_results = run_dataset("MultiCenter (own support)", "multicenter_", cids)

    # data2 verification
    d2_results = run_dataset("data2 (verification)", "", cids)

    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Dataset':<30} {'Method':<25} {'mF1':>7} {'Acc':>7}")
    print("-" * 75)
    print(f"{'MC (data2 support, old)':<30} {'MultiBackbone+ATD':<25} {'0.3189':>7} {'0.4402':>7}")
    if mc_results:
        for n in ["kNN_BC", "MultiBackbone", "MultiBackbone+ATD"]:
            v = mc_results[n]
            if v["mf1"]:
                print(f"{'MC (own support, NEW)':<30} {n:<25} {np.mean(v['mf1']):>7.4f} {np.mean(v['acc']):>7.4f}")
    print()
    print(f"{'data2 (old)':<30} {'SADC+ATD':<25} {'0.7518':>7} {'0.8222':>7}")
    if d2_results:
        for n in ["MultiBackbone+ATD"]:
            v = d2_results[n]
            if v["mf1"]:
                print(f"{'data2 (verification)':<30} {n:<25} {np.mean(v['mf1']):>7.4f} {np.mean(v['acc']):>7.4f}")


if __name__ == "__main__":
    main()
