#!/usr/bin/env python3
"""
SADC v2: Refined innovations after ablation feedback.
Key changes:
  - Better shrinkage alpha sweep for APRDW
  - Softer CBCGT: 2-of-3 agreement sufficient for full weight
  - QAVMF only activates for uncertain queries (margin < threshold)
  - Cascade restricted to actually-confused pairs (detected from support LOO)
  - Support-only LOO confusion detection to identify which pairs need cascade
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


def compute_pairwise_weights(s_morph, cids, alpha=0.3):
    """Support-only pairwise discriminant weights with Ledoit-Wolf shrinkage."""
    ndim = next(iter(s_morph.values())).shape[1]
    all_morph = np.concatenate([s_morph[c] for c in cids])
    global_var = all_morph.var(0)
    pw_weights = {}

    for ci, cj in itertools.combinations(cids, 2):
        mu_i, mu_j = s_morph[ci].mean(0), s_morph[cj].mean(0)
        var_i, var_j = s_morph[ci].var(0), s_morph[cj].var(0)
        var_pooled = (var_i + var_j) / 2.0
        var_reg = alpha * var_pooled + (1.0 - alpha) * global_var
        fisher = (mu_i - mu_j) ** 2 / (var_reg + 1e-10)
        w = 1.0 + np.clip(fisher, 0, 10) * 2.0
        pw_weights[(ci, cj)] = w.astype(np.float32)
        pw_weights[(cj, ci)] = w.astype(np.float32)

    return pw_weights


def detect_confused_pairs_loo(s_bc, s_ph, s_dn, s_morph, cids, k=5):
    """Leave-one-out on support to detect which class pairs are confused.
    Returns set of (ci, cj) pairs where LOO errors occur."""
    confused = set()
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    for c_true in cids:
        n = len(s_bc[c_true])
        for leave_idx in range(n):
            q_bc_i = s_bc[c_true][leave_idx]
            q_ph_i = s_ph[c_true][leave_idx]
            q_dn_i = s_dn[c_true][leave_idx]
            q_m_i = (s_morph[c_true][leave_idx] - gm) / gs

            scores = {}
            for c in cids:
                if c == c_true:
                    mask = list(range(n))
                    mask.remove(leave_idx)
                    bc_s = s_bc[c_true][mask]
                    ph_s = s_ph[c_true][mask]
                    dn_s = s_dn[c_true][mask]
                    m_s = (s_morph[c_true][mask] - gm) / gs
                else:
                    bc_s = s_bc[c]
                    ph_s = s_ph[c]
                    dn_s = s_dn[c]
                    m_s = (s_morph[c] - gm) / gs

                vs = 0.42 * (bc_s @ q_bc_i) + 0.18 * (ph_s @ q_ph_i) + 0.07 * (dn_s @ q_dn_i)
                md = np.linalg.norm(q_m_i - m_s, axis=1)
                ms = 1.0 / (1.0 + md)
                kk = min(k, len(vs))
                scores[c] = float(np.sort(vs + 0.33 * ms)[::-1][:kk].mean())

            pred = max(scores, key=scores.get)
            if pred != c_true:
                pair = tuple(sorted([c_true, pred]))
                confused.add(pair)

    return confused


def sadc_v2(q_bc, q_ph, q_dn, q_morph, q_labels,
            s_bc0, s_ph0, s_dn0, s_morph0,
            cids, pw_weights, confused_pairs,
            bw=0.42, pw_=0.18, dw=0.07, mw=0.33,
            k=7, n_iter=2, top_k_pseudo=5, conf_thr=0.025,
            cascade_thr=0.012, cascade_mw=0.45,
            use_cbcgt=True, use_qavmf=True, use_cascade=True,
            beta=2.0, tau=0.02, qavmf_activate_thr=0.03):

    sm_all = np.concatenate([s_morph0[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    sb = {c: s_bc0[c].copy() for c in cids}
    sp = {c: s_ph0[c].copy() for c in cids}
    sd = {c: s_dn0[c].copy() for c in cids}
    smm = {c: s_morph0[c].copy() for c in cids}

    for it in range(n_iter):
        snm = {c: (smm[c] - gm) / gs for c in cids}
        preds, margins = [], []
        ind_preds = []

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
            fused_pred = cids[int(np.argmax(sa))]
            preds.append(fused_pred)
            margins.append(ss[0] - ss[1])

            if use_cbcgt:
                bc_s = [float(np.sort(sb[c] @ q_bc[i])[::-1][:k].mean()) for c in cids]
                ph_s = [float(np.sort(sp[c] @ q_ph[i])[::-1][:k].mean()) for c in cids]
                ind_preds.append((cids[int(np.argmax(bc_s))],
                                  cids[int(np.argmax(ph_s))]))

        preds = np.array(preds)
        margins = np.array(margins)

        for c in cids:
            candidates = []
            for i in range(len(q_labels)):
                if preds[i] != c or margins[i] <= conf_thr:
                    continue
                if use_cbcgt:
                    pbc, pph = ind_preds[i]
                    n_agree = sum([pbc == c, pph == c])
                    if n_agree >= 2:
                        candidates.append((i, 1.0, margins[i]))
                    elif n_agree >= 1:
                        candidates.append((i, 0.3, margins[i]))
                else:
                    candidates.append((i, 0.5, margins[i]))

            candidates.sort(key=lambda x: -x[2])
            ti = [c_[0] for c_ in candidates[:top_k_pseudo]]
            tw = [c_[1] for c_ in candidates[:top_k_pseudo]]
            if not ti:
                continue
            tw_arr = np.array(tw, dtype=np.float32).reshape(-1, 1)
            sb[c] = np.concatenate([s_bc0[c], q_bc[ti] * tw_arr])
            sp[c] = np.concatenate([s_ph0[c], q_ph[ti] * tw_arr])
            sd[c] = np.concatenate([s_dn0[c], q_dn[ti] * tw_arr])
            smm[c] = np.concatenate([s_morph0[c], q_morph[ti]])

    sm2 = np.concatenate([smm[c] for c in cids])
    gm2, gs2 = sm2.mean(0), sm2.std(0) + 1e-8
    snm = {c: (smm[c] - gm2) / gs2 for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm2) / gs2

        v_scores = {}
        for c in cids:
            vs = bw * (sb[c] @ q_bc[i]) + pw_ * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
            v_scores[c] = float(np.sort(vs)[::-1][:k].mean())

        visual_arr = np.array([v_scores[c] for c in cids])
        v_sorted = np.sort(visual_arr)[::-1]
        visual_margin = v_sorted[0] - v_sorted[1]

        m_scores = {}
        for c in cids:
            md = np.linalg.norm(qm - snm[c], axis=1)
            m_scores[c] = float(np.sort(1.0 / (1.0 + md))[::-1][:k].mean())

        if use_qavmf and visual_margin < qavmf_activate_thr:
            amw = mw * (1.0 + beta * np.exp(-visual_margin / tau))
        else:
            amw = mw

        scores = {}
        for c in cids:
            scores[c] = v_scores[c] + amw * m_scores[c]

        sa = np.array([scores[c] for c in cids])
        t1 = cids[int(np.argmax(sa))]
        mg = np.sort(sa)[::-1][0] - np.sort(sa)[::-1][1]

        if use_cascade and mg < cascade_thr:
            top2_idx = np.argsort(sa)[::-1][:2]
            c1, c2 = cids[top2_idx[0]], cids[top2_idx[1]]
            pair = tuple(sorted([c1, c2]))
            if pair in confused_pairs and pair in pw_weights:
                fw_pair = pw_weights[pair]
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


def main():
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")
    cids = sorted(CLASS_NAMES.keys())

    ablation_configs = [
        ("A: old_leaky_baseline",    False, False, False, False),
        ("B: fix_leak_only",         True,  False, False, False),
        ("C: +confused_cascade",     True,  False, False, True),
        ("D: +CBCGT_soft",           True,  True,  False, False),
        ("E: +QAVMF_selective",      True,  False, True,  False),
        ("F: +CBCGT+cascade",        True,  True,  False, True),
        ("G: +CBCGT+QAVMF",         True,  True,  True,  False),
        ("H: SADC_full",             True,  True,  True,  True),
    ]

    alpha_vals = [0.2, 0.3, 0.5]
    beta_tau_vals = [(2.0, 0.02), (2.5, 0.025)]
    cascade_thr_vals = [0.010, 0.012, 0.015]

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"\nSeed {seed}...")
        si = select_support(lt, seed, cids)
        sbc = {c: bc_t[si[c]] for c in cids}
        sph = {c: ph_t[si[c]] for c in cids}
        sdn = {c: dn_t[si[c]] for c in cids}
        sm = {c: mt[si[c]] for c in cids}

        confused = detect_confused_pairs_loo(sbc, sph, sdn, sm, cids)
        print(f"  Confused pairs (LOO): {confused}")

        # A: old leaky baseline
        from sadc_classify import run_old_leaky_baseline
        m = run_old_leaky_baseline(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, mt, lt, cids)
        all_results["A: old_leaky_baseline"]["acc"].append(m["acc"])
        all_results["A: old_leaky_baseline"]["mf1"].append(m["mf1"])
        for c in cids:
            all_results["A: old_leaky_baseline"]["pc"][c].append(m["pc"][c]["f1"])

        for alpha in alpha_vals:
            pw_w = compute_pairwise_weights(sm, cids, alpha=alpha)

            for beta, tau in beta_tau_vals:
                for ct in cascade_thr_vals:
                    for label, fix, cbcgt, qavmf, cascade in ablation_configs:
                        if not fix:
                            continue
                        tag = f"{label}_a{alpha}"
                        if qavmf:
                            tag += f"_b{beta}_t{tau}"
                        if cascade:
                            tag += f"_ct{ct}"

                        m = sadc_v2(
                            bc_v, ph_v, dn_v, mv, lv,
                            sbc, sph, sdn, sm, cids, pw_w, confused,
                            use_cbcgt=cbcgt, use_qavmf=qavmf, use_cascade=cascade,
                            beta=beta, tau=tau, cascade_thr=ct)
                        all_results[tag]["acc"].append(m["acc"])
                        all_results[tag]["mf1"].append(m["mf1"])
                        for c in cids:
                            all_results[tag]["pc"][c].append(m["pc"][c]["f1"])

    print(f"\n{'=' * 150}")
    print("SADC v2 COMPREHENSIVE ABLATION (5 seeds)")
    print(f"{'=' * 150}")
    h = f"{'Strategy':<70} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h)
    print("-" * 150)
    sr = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for n, v in sr[:30]:
        if len(v["acc"]) < 3:
            continue
        pc = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{n:<70} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc}  "
              f"{np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sr[0]
    print(f"\n*** BEST: {best[0]} ***")
    print(f"    Acc={np.mean(best[1]['acc']):.4f}, mF1={np.mean(best[1]['mf1']):.4f}, "
          f"Eos={np.mean(best[1]['pc'][3]):.4f}")

    print(f"\n--- Best by Eos F1 ---")
    se = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["pc"][3]))
    for n, v in se[:10]:
        if len(v["acc"]) < 3:
            continue
        pc = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{n:<70} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc}")


if __name__ == "__main__":
    main()
