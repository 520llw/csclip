#!/usr/bin/env python3
"""
Support-Anchored Discriminative Classification (SADC)
=====================================================
Innovations:
  1. All-Pair Regularized Discriminant Weighting (APRDW) — support-only, zero leakage
  2. Cross-Backbone Consistency Gated Transduction (CBCGT)
  3. Query-Adaptive Visual-Morphology Fusion (QAVMF)
  4. Generalized Pairwise Cascade for all confusing class pairs
"""
import random, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


def load_cache(m, s, cache_dir=None):
    d = np.load((cache_dir or CACHE_DIR) / f"{m}_{s}.npz")
    return d["feats"], d["morphs"], d["labels"]


def metrics(gt, pred, cids):
    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pred))
    pc, f1s = {}, []
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


def select_support(labels, seed, cids, n_shot=N_SHOT):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels):
        pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(n_shot, len(pc[c]))) for c in cids}


# ==================== Innovation 1: APRDW ====================

def compute_pairwise_discriminant_weights(s_morph, cids, alpha=0.5):
    """All-Pair Regularized Discriminant Weighting using ONLY support samples.

    Uses Ledoit-Wolf-style shrinkage to stabilize variance estimates from
    small samples (10 per class). Returns pairwise weight dict.
    """
    ndim = next(iter(s_morph.values())).shape[1]
    pw_weights = {}

    global_var = np.concatenate([s_morph[c] for c in cids]).var(0)

    for ci, cj in itertools.combinations(cids, 2):
        mu_i = s_morph[ci].mean(0)
        mu_j = s_morph[cj].mean(0)
        var_i = s_morph[ci].var(0)
        var_j = s_morph[cj].var(0)

        var_pooled = (var_i + var_j) / 2.0
        var_reg = alpha * var_pooled + (1.0 - alpha) * global_var
        fisher = (mu_i - mu_j) ** 2 / (var_reg + 1e-10)

        w = 1.0 + fisher * 2.0
        pw_weights[(ci, cj)] = w.astype(np.float32)
        pw_weights[(cj, ci)] = w.astype(np.float32)

    return pw_weights


# ==================== Innovation 2: CBCGT ====================

def score_per_backbone(sb, sp, sd, snm, q_bc, q_ph, q_dn, qm, cids, k):
    """Compute independent per-backbone predictions for consistency gating."""
    bc_scores, ph_scores, dn_scores = [], [], []
    for c in cids:
        bc_scores.append(float(np.sort(sb[c] @ q_bc)[::-1][:k].mean()))
        ph_scores.append(float(np.sort(sp[c] @ q_ph)[::-1][:k].mean()))
        dn_morph = np.linalg.norm(qm - snm[c], axis=1)
        dn_scores.append(float(np.sort(sd[c] @ q_dn + 0.35 / (1.0 + dn_morph))[::-1][:k].mean()))

    return (cids[int(np.argmax(bc_scores))],
            cids[int(np.argmax(ph_scores))],
            cids[int(np.argmax(dn_scores))])


# ==================== Innovation 3: QAVMF ====================

def adaptive_morph_weight(visual_margin, mw_base, beta=2.0, tau=0.02):
    """Query-adaptive morphology weight: increase when visual confidence is low."""
    return mw_base * (1.0 + beta * np.exp(-visual_margin / tau))


# ==================== Full SADC Pipeline ====================

def sadc_pipeline(q_bc, q_ph, q_dn, q_morph, q_labels,
                  s_bc0, s_ph0, s_dn0, s_morph0,
                  cids, pw_weights,
                  bw=0.42, pw_=0.18, dw=0.07, mw=0.33,
                  k=7, n_iter=2, top_k_pseudo=5, conf_thr=0.025,
                  cascade_thr=0.012, cascade_mw=0.45,
                  use_cbcgt=True, use_qavmf=True, use_gen_cascade=True,
                  beta=2.0, tau=0.02):
    """Full SADC classification pipeline."""

    sm_all = np.concatenate([s_morph0[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    sb = {c: s_bc0[c].copy() for c in cids}
    sp = {c: s_ph0[c].copy() for c in cids}
    sd = {c: s_dn0[c].copy() for c in cids}
    smm = {c: s_morph0[c].copy() for c in cids}

    # --- Transductive iterations with CBCGT ---
    for it in range(n_iter):
        snm = {c: (smm[c] - gm) / gs for c in cids}
        preds, margins = [], []
        pred_bc_list, pred_ph_list, pred_dn_list = [], [], []

        for i in range(len(q_labels)):
            qm = (q_morph[i] - gm) / gs
            scores = []
            for c in cids:
                vs = bw * (sb[c] @ q_bc[i]) + pw_ * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                scores.append(float(np.sort(vs + mw * ms)[::-1][:k].mean()))
            sa = np.array(scores)
            ss = np.sort(sa)[::-1]
            preds.append(cids[int(np.argmax(sa))])
            margins.append(ss[0] - ss[1])

            if use_cbcgt:
                pbc, pph, pdn = score_per_backbone(
                    sb, sp, sd, snm, q_bc[i], q_ph[i], q_dn[i], qm, cids, k)
                pred_bc_list.append(pbc)
                pred_ph_list.append(pph)
                pred_dn_list.append(pdn)

        preds = np.array(preds)
        margins = np.array(margins)

        for c in cids:
            if use_cbcgt:
                candidates = []
                for i in range(len(q_labels)):
                    if preds[i] != c:
                        continue
                    pbc = pred_bc_list[i]
                    pph = pred_ph_list[i]
                    pdn = pred_dn_list[i]
                    fused = preds[i]
                    n_agree = sum([pbc == fused, pph == fused, pdn == fused])
                    if n_agree >= 3 and margins[i] > conf_thr:
                        candidates.append((i, 1.0, margins[i]))
                    elif n_agree >= 2 and margins[i] > conf_thr * 1.5:
                        candidates.append((i, 0.3, margins[i]))
                candidates.sort(key=lambda x: -x[2])
                ti = [c_[0] for c_ in candidates[:top_k_pseudo]]
                tw = [c_[1] for c_ in candidates[:top_k_pseudo]]
            else:
                cm = (preds == c) & (margins > conf_thr)
                ci = np.where(cm)[0]
                if len(ci) == 0:
                    continue
                ti = ci[np.argsort(margins[ci])[::-1][:top_k_pseudo]].tolist()
                tw = [0.5] * len(ti)

            if not ti:
                continue
            tw_arr = np.array(tw, dtype=np.float32).reshape(-1, 1)
            sb[c] = np.concatenate([s_bc0[c], q_bc[ti] * tw_arr])
            sp[c] = np.concatenate([s_ph0[c], q_ph[ti] * tw_arr])
            sd[c] = np.concatenate([s_dn0[c], q_dn[ti] * tw_arr])
            smm[c] = np.concatenate([s_morph0[c], q_morph[ti]])

    # --- Final classification pass ---
    sm2 = np.concatenate([smm[c] for c in cids])
    gm2, gs2 = sm2.mean(0), sm2.std(0) + 1e-8
    snm = {c: (smm[c] - gm2) / gs2 for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm2) / gs2

        # Visual scores
        v_scores = {}
        for c in cids:
            vs = bw * (sb[c] @ q_bc[i]) + pw_ * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
            v_scores[c] = float(np.sort(vs)[::-1][:k].mean())

        visual_arr = np.array([v_scores[c] for c in cids])
        v_sorted = np.sort(visual_arr)[::-1]
        visual_margin = v_sorted[0] - v_sorted[1]

        # Morphology scores
        m_scores = {}
        for c in cids:
            md = np.linalg.norm(qm - snm[c], axis=1)
            m_scores[c] = float(np.sort(1.0 / (1.0 + md))[::-1][:k].mean())

        # QAVMF: adaptive morphology weight
        if use_qavmf:
            amw = adaptive_morph_weight(visual_margin, mw, beta, tau)
        else:
            amw = mw

        scores = {}
        for c in cids:
            scores[c] = v_scores[c] + amw * m_scores[c]

        sa = np.array([scores[c] for c in cids])
        t1 = cids[int(np.argmax(sa))]
        mg = np.sort(sa)[::-1][0] - np.sort(sa)[::-1][1]

        # Generalized pairwise cascade
        if use_gen_cascade and mg < cascade_thr:
            top2_idx = np.argsort(sa)[::-1][:2]
            c1, c2 = cids[top2_idx[0]], cids[top2_idx[1]]
            pw_key = (c1, c2) if (c1, c2) in pw_weights else (c2, c1)
            if pw_key in pw_weights:
                fw_pair = pw_weights[pw_key]
                qmw = qm * fw_pair
                for gc in [c1, c2]:
                    snmw_g = snm[gc] * fw_pair
                    mdw = np.linalg.norm(qmw - snmw_g, axis=1)
                    msc = float(np.mean(1.0 / (1.0 + np.sort(mdw)[:5])))
                    vbs = float(np.sort(sb[gc] @ q_bc[i])[::-1][:3].mean())
                    vps = float(np.sort(sp[gc] @ q_ph[i])[::-1][:3].mean())
                    scores[gc] = 0.25 * vbs + 0.20 * vps + cascade_mw * msc
                t1 = c1 if scores[c1] > scores[c2] else c2

        gt.append(int(q_labels[i]))
        pred.append(t1)

    return metrics(gt, pred, cids)


def run_old_leaky_baseline(q_bc, q_ph, q_dn, q_morph, q_labels,
                           s_bc, s_ph, s_dn, s_morph,
                           morph_train_all, labels_train_all,
                           cids, bw=0.42, pw_=0.18, dw=0.07, mw=0.33,
                           k=7, ct=0.012, cmw=0.45):
    """Old pipeline WITH data leakage (Fisher from all data). For comparison only."""
    ndim = morph_train_all.shape[1]
    eos_all = morph_train_all[labels_train_all == 3]
    neu_all = morph_train_all[labels_train_all == 4]
    fw = np.ones(ndim, np.float32)
    for d in range(ndim):
        f = (np.mean(eos_all[:, d]) - np.mean(neu_all[:, d])) ** 2 / \
            (np.var(eos_all[:, d]) + np.var(neu_all[:, d]) + 1e-10)
        fw[d] = 1.0 + f * 2.0

    sm = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm.mean(0), sm.std(0) + 1e-8

    sb, sp, sd, smm = (
        {c: v[c].copy() for c in cids}
        for v in [s_bc, s_ph, s_dn, s_morph]
    )
    for _ in range(2):
        snm = {c: (smm[c] - gm) / gs for c in cids}
        preds, margins = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i] - gm) / gs
            scores = []
            for c in cids:
                vs = bw * (sb[c] @ q_bc[i]) + pw_ * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                scores.append(float(np.sort(vs + mw * ms)[::-1][:k].mean()))
            sa = np.array(scores)
            ss = np.sort(sa)[::-1]
            preds.append(cids[int(np.argmax(sa))])
            margins.append(ss[0] - ss[1])
        preds, margins = np.array(preds), np.array(margins)
        for c in cids:
            cm = (preds == c) & (margins > 0.025)
            ci = np.where(cm)[0]
            if len(ci) == 0:
                continue
            ti = ci[np.argsort(margins[ci])[::-1][:5]]
            sb[c] = np.concatenate([s_bc[c], q_bc[ti] * 0.5])
            sp[c] = np.concatenate([s_ph[c], q_ph[ti] * 0.5])
            sd[c] = np.concatenate([s_dn[c], q_dn[ti] * 0.5])
            smm[c] = np.concatenate([s_morph[c], q_morph[ti]])

    sm2 = np.concatenate([smm[c] for c in cids])
    gm2, gs2 = sm2.mean(0), sm2.std(0) + 1e-8
    snm = {c: (smm[c] - gm2) / gs2 for c in cids}
    snmw = {c: snm[c] * fw for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm2) / gs2
        qmw = qm * fw
        scores = {}
        for c in cids:
            vs = bw * (sb[c] @ q_bc[i]) + pw_ * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            scores[c] = float(np.sort(vs + mw * ms)[::-1][:k].mean())
        sa = np.array([scores[c] for c in cids])
        t1 = cids[int(np.argmax(sa))]
        mg = np.sort(sa)[::-1][0] - np.sort(sa)[::-1][1]
        if t1 in [3, 4] and mg < ct:
            for gc in [3, 4]:
                mdw = np.linalg.norm(qmw - snmw[gc], axis=1)
                msc = float(np.mean(1.0 / (1.0 + np.sort(mdw)[:5])))
                vbs = float(np.sort(sb[gc] @ q_bc[i])[::-1][:3].mean())
                vps = float(np.sort(sp[gc] @ q_ph[i])[::-1][:3].mean())
                scores[gc] = 0.25 * vbs + 0.20 * vps + cmw * msc
            t1 = 3 if scores[3] > scores[4] else 4
        gt.append(int(q_labels[i]))
        pred.append(t1)
    return metrics(gt, pred, cids)


def print_row(name, v, cids):
    pc = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    print(f"{name:<55} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc}  "
          f"{np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")


def main():
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")

    cids = sorted(CLASS_NAMES.keys())

    configs = {
        "old_leaky_baseline":       dict(use_cbcgt=False, use_qavmf=False, use_gen_cascade=False),
        "fix_leak_only":            dict(use_cbcgt=False, use_qavmf=False, use_gen_cascade=False),
        "fix_leak+gen_cascade":     dict(use_cbcgt=False, use_qavmf=False, use_gen_cascade=True),
        "fix_leak+CBCGT":           dict(use_cbcgt=True,  use_qavmf=False, use_gen_cascade=False),
        "fix_leak+QAVMF":           dict(use_cbcgt=False, use_qavmf=True,  use_gen_cascade=False),
        "fix_leak+CBCGT+gen_cas":   dict(use_cbcgt=True,  use_qavmf=False, use_gen_cascade=True),
        "SADC_full":                dict(use_cbcgt=True,  use_qavmf=True,  use_gen_cascade=True),
    }

    beta_tau_grid = [(1.5, 0.015), (2.0, 0.02), (2.5, 0.025), (3.0, 0.03)]

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"\nSeed {seed}...")
        si = select_support(lt, seed, cids)
        sbc = {c: bc_t[si[c]] for c in cids}
        sph = {c: ph_t[si[c]] for c in cids}
        sdn = {c: dn_t[si[c]] for c in cids}
        sm = {c: mt[si[c]] for c in cids}

        pw_w = compute_pairwise_discriminant_weights(sm, cids, alpha=0.5)

        # Old leaky baseline for comparison
        m = run_old_leaky_baseline(
            bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, mt, lt, cids)
        all_results["old_leaky_baseline"]["acc"].append(m["acc"])
        all_results["old_leaky_baseline"]["mf1"].append(m["mf1"])
        for c in cids:
            all_results["old_leaky_baseline"]["pc"][c].append(m["pc"][c]["f1"])

        for name, cfg in configs.items():
            if name == "old_leaky_baseline":
                continue

            if cfg.get("use_qavmf"):
                for beta, tau in beta_tau_grid:
                    tag = f"{name}_b{beta}_t{tau}"
                    m = sadc_pipeline(
                        bc_v, ph_v, dn_v, mv, lv,
                        sbc, sph, sdn, sm, cids, pw_w,
                        beta=beta, tau=tau, **cfg)
                    all_results[tag]["acc"].append(m["acc"])
                    all_results[tag]["mf1"].append(m["mf1"])
                    for c in cids:
                        all_results[tag]["pc"][c].append(m["pc"][c]["f1"])
            else:
                m = sadc_pipeline(
                    bc_v, ph_v, dn_v, mv, lv,
                    sbc, sph, sdn, sm, cids, pw_w, **cfg)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[name]["pc"][c].append(m["pc"][c]["f1"])

    print(f"\n{'=' * 140}")
    print("SADC ABLATION STUDY (5 seeds, data2_organized)")
    print(f"{'=' * 140}")
    h = f"{'Strategy':<55} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h)
    print("-" * 140)
    sr = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for n, v in sr:
        if len(v["acc"]) < 3:
            continue
        print_row(n, v, cids)

    best = sr[0]
    print(f"\n*** BEST: {best[0]} -> Acc={np.mean(best[1]['acc']):.4f}, "
          f"mF1={np.mean(best[1]['mf1']):.4f}, "
          f"Eos={np.mean(best[1]['pc'][3]):.4f} ***")


if __name__ == "__main__":
    main()
