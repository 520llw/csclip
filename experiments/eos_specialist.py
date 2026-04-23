#!/usr/bin/env python3
"""
Eos-Specialist Classification: Target the Eos F1 bottleneck (0.44→0.55+).

Innovations:
1. Eos-Neu binary expert: When main classifier is uncertain between Eos/Neu,
   trigger a specialist that uses morphology-heavy features
2. Morphology hard gating: Use granule features to pre-screen Eos candidates
3. Density-driven prototype rectification: Shift prototypes along sample density
   gradient to better separate Eos from Neu
4. Class-balanced temperature scaling: Different softmax temperatures per class
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


# ==================== Density-driven Prototype Rectification ====================

def rectify_prototypes(s_feats_per_class, q_feats, cids, q_pred, gamma=0.5):
    """
    Shift class prototypes towards high-density query regions.
    For each class, compute mean of confident query predictions and
    interpolate with support prototype.
    """
    rectified = {}
    for c in cids:
        proto = s_feats_per_class[c].mean(0)
        mask = np.array(q_pred) == c
        if mask.sum() < 3:
            rectified[c] = proto
            continue
        q_proto = q_feats[mask].mean(0)
        q_proto /= np.linalg.norm(q_proto) + 1e-8
        proto /= np.linalg.norm(proto) + 1e-8
        rect = (1 - gamma) * proto + gamma * q_proto
        rect /= np.linalg.norm(rect) + 1e-8
        rectified[c] = rect
    return rectified


# ==================== Eos-Neu Binary Expert ====================

def eos_neu_expert(q_bc_i, q_ph_i, q_dn_i, q_morph_i,
                   s_bc, s_ph, s_dn, s_morph, gm, gs,
                   eos_cid=3, neu_cid=4,
                   vis_weight=0.4, morph_weight=0.6, k=5):
    """
    Binary classifier for Eos vs Neu using morphology-heavy scoring.
    When main classifier is uncertain, this expert makes the final call.
    """
    qm = (q_morph_i - gm) / gs

    scores = {}
    for c in [eos_cid, neu_cid]:
        if len(s_bc[c]) == 0:
            scores[c] = -1e9
            continue
        vs_bc = np.sort(s_bc[c] @ q_bc_i)[::-1][:k].mean()
        vs_ph = np.sort(s_ph[c] @ q_ph_i)[::-1][:k].mean()
        vs_dn = np.sort(s_dn[c] @ q_dn_i)[::-1][:k].mean()
        vis_score = 0.5 * vs_bc + 0.3 * vs_ph + 0.2 * vs_dn

        snm = (s_morph[c] - gm) / gs
        md = np.linalg.norm(qm - snm, axis=1)
        morph_score = np.mean(1.0 / (1.0 + np.sort(md)[:k]))

        scores[c] = vis_weight * vis_score + morph_weight * morph_score

    return max(scores, key=scores.get)


# ==================== Full Pipeline ====================

def eos_specialist_classify(q_bc, q_ph, q_dn, q_morph, q_labels,
                            s_bc0, s_ph0, s_dn0, s_morph0, cids,
                            bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7,
                            expert_margin=0.05, expert_morph_w=0.6,
                            use_rectification=True, rect_gamma=0.3,
                            use_atd=True, n_iter=2, top_k_pseudo=5, conf_thr=0.025,
                            use_class_temp=False, eos_temp=0.8):
    """
    Classification with Eos specialist improvements.
    """
    K = len(cids)
    EOS, NEU = 3, 4

    sb = {c: s_bc0[c].copy() for c in cids}
    sp = {c: s_ph0[c].copy() for c in cids}
    sd = {c: s_dn0[c].copy() for c in cids}
    smm = {c: s_morph0[c].copy() for c in cids}

    sb_orig = {c: s_bc0[c].copy() for c in cids}
    sp_orig = {c: s_ph0[c].copy() for c in cids}
    sd_orig = {c: s_dn0[c].copy() for c in cids}
    smm_orig = {c: s_morph0[c].copy() for c in cids}

    for it in range(n_iter if use_atd else 1):
        sm_all = np.concatenate([smm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        snm = {c: (smm[c] - gm) / gs for c in cids}

        preds, margins, all_scores = [], [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i] - gm) / gs
            scores = []
            for c in cids:
                vs = bw * (sb[c] @ q_bc[i]) + pw * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                scores.append(float(np.sort(vs + mw * ms)[::-1][:k].mean()))
            sa = np.array(scores)
            ss = np.sort(sa)[::-1]
            preds.append(cids[int(np.argmax(sa))])
            margins.append(ss[0] - ss[1])
            all_scores.append(sa)

        preds = np.array(preds)
        margins = np.array(margins)

        if use_atd and it < n_iter - 1:
            for c in cids:
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

    if use_rectification:
        rect_proto = rectify_prototypes(
            {c: sb_orig[c] for c in cids}, q_bc, cids, preds.tolist(), rect_gamma)

    sm_all = np.concatenate([smm[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    gt, final_pred = [], []
    expert_triggered = 0
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm) / gs
        scores = {}
        for c in cids:
            snm_c = (smm[c] - gm) / gs
            vs = bw * (sb[c] @ q_bc[i]) + pw * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
            md = np.linalg.norm(qm - snm_c, axis=1)
            ms = 1.0 / (1.0 + md)
            scores[c] = float(np.sort(vs + mw * ms)[::-1][:k].mean())

        sa = np.array([scores[c] for c in cids])

        if use_class_temp:
            eos_idx = cids.index(EOS)
            sa_temp = sa.copy()
            sa_temp[eos_idx] *= (1.0 / eos_temp)
            t1 = cids[int(np.argmax(sa_temp))]
        else:
            t1 = cids[int(np.argmax(sa))]

        ss = np.sort(sa)[::-1]
        mg = ss[0] - ss[1]

        if mg < expert_margin:
            top2 = set(cids[j] for j in np.argsort(sa)[::-1][:2])
            if EOS in top2 and NEU in top2:
                t1 = eos_neu_expert(q_bc[i], q_ph[i], q_dn[i], q_morph[i],
                                    sb, sp, sd, smm, gm, gs,
                                    vis_weight=1.0 - expert_morph_w,
                                    morph_weight=expert_morph_w)
                expert_triggered += 1

        gt.append(int(q_labels[i]))
        final_pred.append(t1)

    m = calc_metrics(gt, final_pred, cids)
    m["expert_triggered"] = expert_triggered
    return m


def print_row(name, v, cids):
    pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    et = int(np.mean(v.get('expert', [0])))
    print(f"{name:<55} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  "
          f"{np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f} {et:>4}")


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
        ("SADC_ATD_baseline",            dict(expert_margin=0, use_rectification=False, use_class_temp=False)),
        # Expert margin sweep
        ("expert_m003",                  dict(expert_margin=0.03, expert_morph_w=0.6)),
        ("expert_m005",                  dict(expert_margin=0.05, expert_morph_w=0.6)),
        ("expert_m008",                  dict(expert_margin=0.08, expert_morph_w=0.6)),
        ("expert_m010",                  dict(expert_margin=0.10, expert_morph_w=0.6)),
        ("expert_m015",                  dict(expert_margin=0.15, expert_morph_w=0.6)),
        # Expert morph weight sweep
        ("expert_m005_mw04",             dict(expert_margin=0.05, expert_morph_w=0.4)),
        ("expert_m005_mw08",             dict(expert_margin=0.05, expert_morph_w=0.8)),
        ("expert_m005_mw10",             dict(expert_margin=0.05, expert_morph_w=1.0)),
        # With prototype rectification
        ("expert_m005+rect03",           dict(expert_margin=0.05, expert_morph_w=0.6, use_rectification=True, rect_gamma=0.3)),
        ("expert_m005+rect05",           dict(expert_margin=0.05, expert_morph_w=0.6, use_rectification=True, rect_gamma=0.5)),
        ("expert_m010+rect03",           dict(expert_margin=0.10, expert_morph_w=0.6, use_rectification=True, rect_gamma=0.3)),
        # Class temperature scaling
        ("eos_temp08",                   dict(expert_margin=0, use_class_temp=True, eos_temp=0.8)),
        ("eos_temp06",                   dict(expert_margin=0, use_class_temp=True, eos_temp=0.6)),
        ("eos_temp10",                   dict(expert_margin=0, use_class_temp=True, eos_temp=1.0)),
        # Combined
        ("expert_m005+rect03+temp08",    dict(expert_margin=0.05, expert_morph_w=0.6, use_rectification=True, rect_gamma=0.3, use_class_temp=True, eos_temp=0.8)),
        ("expert_m010+rect03+temp06",    dict(expert_margin=0.10, expert_morph_w=0.6, use_rectification=True, rect_gamma=0.3, use_class_temp=True, eos_temp=0.6)),
    ]

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list), "expert": []})

    for seed in SEEDS:
        print(f"\n{'='*80}\nSeed {seed}\n{'='*80}", flush=True)
        si = select_support(lt, seed, cids)
        sbc = {c: bc_t[si[c]] for c in cids}
        sph = {c: ph_t[si[c]] for c in cids}
        sdn = {c: dn_t[si[c]] for c in cids}
        sm = {c: mt[si[c]] for c in cids}

        for name, cfg in configs:
            m = eos_specialist_classify(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, cids, **cfg)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            all_results[name]["expert"].append(m.get("expert_triggered", 0))
            for c in cids:
                all_results[name]["pc"][c].append(m["pc"][c]["f1"])
            print(f"  {name:<50} mf1={m['mf1']:.4f} Eos={m['pc'][3]['f1']:.4f} "
                  f"expert={m.get('expert_triggered',0)}", flush=True)

    print(f"\n{'='*150}")
    print("EOS SPECIALIST RESULTS (5 seeds, data2_organized)")
    print(f"{'='*150}")
    h = f"{'Strategy':<55} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5} {'Exp':>4}"
    print(h)
    print("-" * 150)
    sr = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for n, v in sr:
        print_row(n, v, cids)

    baseline_eos = np.mean(all_results["SADC_ATD_baseline"]["pc"][3])
    best = sr[0]
    best_eos = np.mean(best[1]["pc"][3])
    print(f"\n*** BEST: {best[0]} ***")
    print(f"    mF1={np.mean(best[1]['mf1']):.4f}, Eos F1={best_eos:.4f}")
    print(f"    Eos improvement: {best_eos - baseline_eos:+.4f}")


if __name__ == "__main__":
    main()
