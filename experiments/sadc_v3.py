#!/usr/bin/env python3
"""
SADC v3: Fundamentally different approach to compensate for data leakage fix.

Key innovations (all zero-leakage):
  1. Support Feature Augmentation (SFA): mixup in feature space, 10→30 per class
  2. Backbone-Disagreement Cascade (BDC): use cross-backbone disagreement
     as cascade trigger instead of Fisher weights
  3. Morphology Top-K Selection (MTKS): select most discriminative morph dims
     from support LOO, not Fisher from all data
  4. Adaptive Transduction with Diversity (ATD): pseudo-labels weighted by
     diversity relative to existing support
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


# ==================== Innovation 1: Support Feature Augmentation ====================

def augment_support_mixup(s_feats, n_aug=20, alpha=0.6):
    """Generate synthetic support samples via feature-space mixup.
    For N samples, generate n_aug additional samples by interpolating random pairs.
    """
    n = len(s_feats)
    if n < 2:
        return s_feats
    aug = []
    for _ in range(n_aug):
        i, j = random.sample(range(n), 2)
        lam = np.random.beta(alpha, alpha)
        mixed = lam * s_feats[i] + (1.0 - lam) * s_feats[j]
        mixed = mixed / (np.linalg.norm(mixed) + 1e-8)
        aug.append(mixed)
    return np.concatenate([s_feats, np.stack(aug)])


def augment_support_all(s_bc, s_ph, s_dn, s_morph, cids, n_aug=20, seed=0):
    """Augment all feature spaces consistently."""
    rng_state = random.getstate()
    np_state = np.random.get_state()

    a_bc, a_ph, a_dn, a_morph = {}, {}, {}, {}
    for c in cids:
        random.seed(seed + c)
        np.random.seed(seed + c)
        n = len(s_bc[c])
        pairs = [(random.sample(range(n), 2), np.random.beta(0.6, 0.6))
                 for _ in range(n_aug)]

        bc_aug, ph_aug, dn_aug, m_aug = [], [], [], []
        for (i, j), lam in pairs:
            bc_mix = lam * s_bc[c][i] + (1 - lam) * s_bc[c][j]
            bc_mix /= np.linalg.norm(bc_mix) + 1e-8
            bc_aug.append(bc_mix)

            ph_mix = lam * s_ph[c][i] + (1 - lam) * s_ph[c][j]
            ph_mix /= np.linalg.norm(ph_mix) + 1e-8
            ph_aug.append(ph_mix)

            dn_mix = lam * s_dn[c][i] + (1 - lam) * s_dn[c][j]
            dn_mix /= np.linalg.norm(dn_mix) + 1e-8
            dn_aug.append(dn_mix)

            m_aug.append(lam * s_morph[c][i] + (1 - lam) * s_morph[c][j])

        a_bc[c] = np.concatenate([s_bc[c], np.stack(bc_aug)])
        a_ph[c] = np.concatenate([s_ph[c], np.stack(ph_aug)])
        a_dn[c] = np.concatenate([s_dn[c], np.stack(dn_aug)])
        a_morph[c] = np.concatenate([s_morph[c], np.stack(m_aug)])

    random.setstate(rng_state)
    np.random.set_state(np_state)
    return a_bc, a_ph, a_dn, a_morph


# ==================== Innovation 2: Backbone-Disagreement Cascade ====================

def backbone_disagreement_cascade(
        sb, sp, sd, snm, q_bc, q_ph, q_dn, qm,
        fused_pred, fused_margin, cids, k,
        cascade_mw=0.50):
    """When backbones disagree on top-1, use morphology-heavy re-scoring
    to disambiguate between the disagreed classes. No Fisher weights needed."""

    bc_scores = {c: float(np.sort(sb[c] @ q_bc)[::-1][:k].mean()) for c in cids}
    ph_scores = {c: float(np.sort(sp[c] @ q_ph)[::-1][:k].mean()) for c in cids}

    pred_bc = max(bc_scores, key=bc_scores.get)
    pred_ph = max(ph_scores, key=ph_scores.get)

    if pred_bc == pred_ph == fused_pred:
        return fused_pred

    candidates = {fused_pred, pred_bc, pred_ph}
    if len(candidates) < 2:
        return fused_pred

    best_c, best_score = fused_pred, -1e9
    for c in candidates:
        vs_bc = float(np.sort(sb[c] @ q_bc)[::-1][:3].mean())
        vs_ph = float(np.sort(sp[c] @ q_ph)[::-1][:3].mean())
        vs_dn = float(np.sort(sd[c] @ q_dn)[::-1][:3].mean())
        md = np.linalg.norm(qm - snm[c], axis=1)
        ms = float(np.mean(1.0 / (1.0 + np.sort(md)[:5])))
        score = 0.20 * vs_bc + 0.15 * vs_ph + 0.10 * vs_dn + cascade_mw * ms
        if score > best_score:
            best_score = score
            best_c = c

    return best_c


# ==================== Innovation 3: Morphology Top-K Selection ====================

def select_top_morph_dims(s_morph, cids, top_k=15):
    """Select most discriminative morphology dimensions via support-only
    pairwise Fisher analysis. Returns boolean mask of selected dims."""
    ndim = next(iter(s_morph.values())).shape[1]
    all_morph = np.concatenate([s_morph[c] for c in cids])
    global_var = all_morph.var(0) + 1e-10

    fisher_sum = np.zeros(ndim)
    for ci, cj in itertools.combinations(cids, 2):
        mu_diff = (s_morph[ci].mean(0) - s_morph[cj].mean(0)) ** 2
        fisher_sum += mu_diff / global_var

    top_dims = np.argsort(fisher_sum)[::-1][:top_k]
    return top_dims


# ==================== Full SADC v3 Pipeline ====================

def sadc_v3(q_bc, q_ph, q_dn, q_morph, q_labels,
            s_bc0, s_ph0, s_dn0, s_morph0,
            cids,
            bw=0.42, pw_=0.18, dw=0.07, mw=0.33,
            k=7, n_iter=2, top_k_pseudo=5, conf_thr=0.025,
            use_sfa=True, n_aug=20,
            use_bdc=True, cascade_mw=0.50,
            use_mtks=True, top_k_dims=15,
            use_atd=True):

    if use_sfa:
        sb, sp, sd, smm = augment_support_all(s_bc0, s_ph0, s_dn0, s_morph0, cids, n_aug)
    else:
        sb = {c: s_bc0[c].copy() for c in cids}
        sp = {c: s_ph0[c].copy() for c in cids}
        sd = {c: s_dn0[c].copy() for c in cids}
        smm = {c: s_morph0[c].copy() for c in cids}

    if use_mtks:
        top_dims = select_top_morph_dims(s_morph0, cids, top_k_dims)
    else:
        top_dims = np.arange(next(iter(s_morph0.values())).shape[1])

    sm_all = np.concatenate([smm[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8

    sb_orig = {c: s_bc0[c].copy() for c in cids}
    sp_orig = {c: s_ph0[c].copy() for c in cids}
    sd_orig = {c: s_dn0[c].copy() for c in cids}
    smm_orig = {c: s_morph0[c].copy() for c in cids}

    for it in range(n_iter):
        snm = {c: (smm[c] - gm) / gs for c in cids}
        preds, margins = [], []

        for i in range(len(q_labels)):
            qm = (q_morph[i] - gm) / gs
            scores = []
            for c in cids:
                vs = bw * (sb[c] @ q_bc[i]) + pw_ * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
                qm_sel = qm[top_dims] if use_mtks else qm
                snm_sel = snm[c][:, top_dims] if use_mtks else snm[c]
                md = np.linalg.norm(qm_sel - snm_sel, axis=1)
                ms = 1.0 / (1.0 + md)
                scores.append(float(np.sort(vs + mw * ms)[::-1][:k].mean()))
            sa = np.array(scores)
            ss = np.sort(sa)[::-1]
            preds.append(cids[int(np.argmax(sa))])
            margins.append(ss[0] - ss[1])

        preds = np.array(preds)
        margins = np.array(margins)

        for c in cids:
            cm = (preds == c) & (margins > conf_thr)
            ci = np.where(cm)[0]
            if len(ci) == 0:
                continue

            if use_atd:
                proto_c = sb_orig[c].mean(0)
                dists = np.array([np.linalg.norm(q_bc[idx] - proto_c) for idx in ci])
                diversity_scores = margins[ci] * (1.0 + 0.3 * dists / (dists.mean() + 1e-8))
                ti = ci[np.argsort(diversity_scores)[::-1][:top_k_pseudo]]
            else:
                ti = ci[np.argsort(margins[ci])[::-1][:top_k_pseudo]]

            sb[c] = np.concatenate([sb_orig[c] if not use_sfa else sb[c][:len(s_bc0[c]) + n_aug],
                                    q_bc[ti] * 0.5])
            sp[c] = np.concatenate([sp_orig[c] if not use_sfa else sp[c][:len(s_ph0[c]) + n_aug],
                                    q_ph[ti] * 0.5])
            sd[c] = np.concatenate([sd_orig[c] if not use_sfa else sd[c][:len(s_dn0[c]) + n_aug],
                                    q_dn[ti] * 0.5])
            smm[c] = np.concatenate([smm_orig[c] if not use_sfa else smm[c][:len(s_morph0[c]) + n_aug],
                                     q_morph[ti]])

    sm2 = np.concatenate([smm[c] for c in cids])
    gm2, gs2 = sm2.mean(0), sm2.std(0) + 1e-8
    snm = {c: (smm[c] - gm2) / gs2 for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm2) / gs2

        scores = {}
        for c in cids:
            vs = bw * (sb[c] @ q_bc[i]) + pw_ * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
            qm_sel = qm[top_dims] if use_mtks else qm
            snm_sel = snm[c][:, top_dims] if use_mtks else snm[c]
            md = np.linalg.norm(qm_sel - snm_sel, axis=1)
            ms = 1.0 / (1.0 + md)
            scores[c] = float(np.sort(vs + mw * ms)[::-1][:k].mean())

        sa = np.array([scores[c] for c in cids])
        t1 = cids[int(np.argmax(sa))]
        mg = np.sort(sa)[::-1][0] - np.sort(sa)[::-1][1]

        if use_bdc:
            t1 = backbone_disagreement_cascade(
                sb, sp, sd, snm, q_bc[i], q_ph[i], q_dn[i], qm,
                t1, mg, cids, k, cascade_mw)

        gt.append(int(q_labels[i]))
        pred.append(t1)

    return metrics(gt, pred, cids)


def run_old_leaky(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, mt, lt, cids):
    from sadc_classify import run_old_leaky_baseline
    return run_old_leaky_baseline(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, mt, lt, cids)


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

    configs = [
        ("old_leaky_baseline",           None),
        ("fix_leak_baseline",            dict(use_sfa=False, use_bdc=False, use_mtks=False, use_atd=False)),
        ("+SFA_aug10",                   dict(use_sfa=True,  n_aug=10,  use_bdc=False, use_mtks=False, use_atd=False)),
        ("+SFA_aug20",                   dict(use_sfa=True,  n_aug=20,  use_bdc=False, use_mtks=False, use_atd=False)),
        ("+SFA_aug30",                   dict(use_sfa=True,  n_aug=30,  use_bdc=False, use_mtks=False, use_atd=False)),
        ("+BDC_cm40",                    dict(use_sfa=False, use_bdc=True, cascade_mw=0.40, use_mtks=False, use_atd=False)),
        ("+BDC_cm50",                    dict(use_sfa=False, use_bdc=True, cascade_mw=0.50, use_mtks=False, use_atd=False)),
        ("+BDC_cm55",                    dict(use_sfa=False, use_bdc=True, cascade_mw=0.55, use_mtks=False, use_atd=False)),
        ("+MTKS_k10",                    dict(use_sfa=False, use_bdc=False, use_mtks=True, top_k_dims=10, use_atd=False)),
        ("+MTKS_k15",                    dict(use_sfa=False, use_bdc=False, use_mtks=True, top_k_dims=15, use_atd=False)),
        ("+MTKS_k20",                    dict(use_sfa=False, use_bdc=False, use_mtks=True, top_k_dims=20, use_atd=False)),
        ("+ATD",                         dict(use_sfa=False, use_bdc=False, use_mtks=False, use_atd=True)),
        ("+SFA20+BDC50",                 dict(use_sfa=True, n_aug=20, use_bdc=True, cascade_mw=0.50, use_mtks=False, use_atd=False)),
        ("+SFA20+MTKS15",               dict(use_sfa=True, n_aug=20, use_bdc=False, use_mtks=True, top_k_dims=15, use_atd=False)),
        ("+SFA20+ATD",                   dict(use_sfa=True, n_aug=20, use_bdc=False, use_mtks=False, use_atd=True)),
        ("+SFA20+BDC50+MTKS15",         dict(use_sfa=True, n_aug=20, use_bdc=True, cascade_mw=0.50, use_mtks=True, top_k_dims=15, use_atd=False)),
        ("+SFA20+BDC50+ATD",            dict(use_sfa=True, n_aug=20, use_bdc=True, cascade_mw=0.50, use_mtks=False, use_atd=True)),
        ("SADC_full_v3",                 dict(use_sfa=True, n_aug=20, use_bdc=True, cascade_mw=0.50, use_mtks=True, top_k_dims=15, use_atd=True)),
        ("+SFA30+BDC50+MTKS15+ATD",     dict(use_sfa=True, n_aug=30, use_bdc=True, cascade_mw=0.50, use_mtks=True, top_k_dims=15, use_atd=True)),
        ("+SFA20+BDC55+MTKS15+ATD",     dict(use_sfa=True, n_aug=20, use_bdc=True, cascade_mw=0.55, use_mtks=True, top_k_dims=15, use_atd=True)),
        ("+SFA20+BDC40+MTKS10+ATD",     dict(use_sfa=True, n_aug=20, use_bdc=True, cascade_mw=0.40, use_mtks=True, top_k_dims=10, use_atd=True)),
    ]

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"\nSeed {seed}...")
        si = select_support(lt, seed, cids)
        sbc = {c: bc_t[si[c]] for c in cids}
        sph = {c: ph_t[si[c]] for c in cids}
        sdn = {c: dn_t[si[c]] for c in cids}
        sm = {c: mt[si[c]] for c in cids}

        for name, cfg in configs:
            if cfg is None:
                m = run_old_leaky(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, mt, lt, cids)
            else:
                m = sadc_v3(bc_v, ph_v, dn_v, mv, lv, sbc, sph, sdn, sm, cids, **cfg)

            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids:
                all_results[name]["pc"][c].append(m["pc"][c]["f1"])

    print(f"\n{'=' * 150}")
    print("SADC v3 ABLATION (5 seeds, data2_organized)")
    print(f"{'=' * 150}")
    h = f"{'Strategy':<55} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h)
    print("-" * 150)
    sr = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for n, v in sr:
        if len(v["acc"]) < 3:
            continue
        print_row(n, v, cids)

    best_no_leak = [(n, v) for n, v in sr if "leaky" not in n][0]
    print(f"\n*** BEST (no-leak): {best_no_leak[0]} ***")
    print(f"    Acc={np.mean(best_no_leak[1]['acc']):.4f}, "
          f"mF1={np.mean(best_no_leak[1]['mf1']):.4f}, "
          f"Eos={np.mean(best_no_leak[1]['pc'][3]):.4f}")


if __name__ == "__main__":
    main()
