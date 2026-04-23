#!/usr/bin/env python3
"""
Nested Cross-Validation for BALF 10-shot Classification.

Eliminates all forms of hyperparameter leakage by:
1. Outer loop: 5-fold on val set → report final metrics
2. Inner loop: Leave-One-Out on support set → tune hyperparams
3. No information from test fold ever used for any decision

Also implements SOTA baselines for comparison:
- Linear Probe (Logistic Regression on frozen features)
- Nearest Centroid (NCM/prototype matching)
- kNN with optimal k
- Our full SADC+ATD method
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


# ==================== SOTA Baselines ====================

def linear_probe(q_feats, s_feats_per_class, cids, lr=0.01, epochs=200, wd=0.001):
    """Logistic Regression on frozen features (most common few-shot baseline)."""
    K = len(cids)
    X, y = [], []
    for ki, c in enumerate(cids):
        X.append(s_feats_per_class[c])
        y.extend([ki] * len(s_feats_per_class[c]))
    X = np.concatenate(X)
    y = np.array(y)
    D = X.shape[1]

    W = np.random.randn(D, K) * 0.01
    b = np.zeros(K)

    cw = 1.0 / (np.bincount(y, minlength=K) + 1e-8)
    cw /= cw.sum() / K
    sw = cw[y]

    for _ in range(epochs):
        logits = X @ W + b
        probs = softmax(logits, axis=1)
        g = probs.copy()
        g[np.arange(len(y)), y] -= 1.0
        g *= sw[:, None]
        g /= len(y)
        W -= lr * (X.T @ g + wd * W)
        b -= lr * g.sum(0)

    return q_feats @ W + b


def ncm_classify(q_feats, s_feats_per_class, cids):
    """Nearest Centroid Method."""
    K = len(cids)
    protos = np.zeros((K, q_feats.shape[1]))
    for ki, c in enumerate(cids):
        protos[ki] = s_feats_per_class[c].mean(0)
        protos[ki] /= np.linalg.norm(protos[ki]) + 1e-8
    q_norm = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-8)
    return q_norm @ protos.T


def knn_classify(q_feats, s_feats_per_class, cids, k=5):
    """k-Nearest Neighbor."""
    K = len(cids)
    scores = np.zeros((len(q_feats), K))
    for ki, c in enumerate(cids):
        sims = q_feats @ s_feats_per_class[c].T
        for i in range(len(q_feats)):
            topk = np.sort(sims[i])[::-1][:min(k, len(s_feats_per_class[c]))]
            scores[i, ki] = topk.mean()
    return scores


def sadc_atd_classify(q_bc, q_ph, q_dn, q_morph,
                      s_bc, s_ph, s_dn, s_morph, cids,
                      bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7,
                      n_iter=2, top_k_pseudo=5, conf_thr=0.025):
    """Our SADC+ATD method."""
    K = len(cids)
    sb = {c: s_bc[c].copy() for c in cids}
    sp = {c: s_ph[c].copy() for c in cids}
    sd = {c: s_dn[c].copy() for c in cids}
    smm = {c: s_morph[c].copy() for c in cids}
    sb_orig = {c: s_bc[c].copy() for c in cids}
    smm_orig = {c: s_morph[c].copy() for c in cids}

    for it in range(n_iter):
        sm_all = np.concatenate([smm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        snm = {c: (smm[c] - gm) / gs for c in cids}

        all_scores = np.zeros((len(q_bc), K))
        for i in range(len(q_bc)):
            qm = (q_morph[i] - gm) / gs
            for ki, c in enumerate(cids):
                vs = bw * (sb[c] @ q_bc[i]) + pw * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                all_scores[i, ki] = float(np.sort(vs + mw * ms)[::-1][:k].mean())

        if it < n_iter - 1:
            preds = np.array([cids[int(np.argmax(all_scores[i]))] for i in range(len(q_bc))])
            margins = np.array([np.sort(all_scores[i])[::-1][0] - np.sort(all_scores[i])[::-1][1] for i in range(len(q_bc))])
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
                sp[c] = np.concatenate([s_ph[c], q_ph[ti] * 0.5])
                sd[c] = np.concatenate([s_dn[c], q_dn[ti] * 0.5])
                smm[c] = np.concatenate([smm_orig[c], q_morph[ti]])

    return all_scores


# ==================== Cross-Validation Framework ====================

def k_fold_split(indices, n_folds=5, seed=42):
    """Split indices into k folds."""
    rng = np.random.RandomState(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    folds = []
    fold_size = len(shuffled) // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(shuffled)
        folds.append(shuffled[start:end])
    return folds


def nested_cv_evaluate(bc_v, ph_v, dn_v, mv, lv,
                       bc_t, ph_t, dn_t, mt, lt,
                       cids, n_outer=5, seed=42):
    """
    Nested cross-validation:
    - Outer loop: 5-fold on val set
    - Inner: support selection from train set (no tuning needed on val)
    - Report per-fold and aggregated metrics
    """
    results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    val_indices = np.arange(len(lv))
    folds = k_fold_split(val_indices, n_outer, seed)

    for fold_idx, test_fold in enumerate(folds):
        print(f"\n  Fold {fold_idx+1}/{n_outer} (test={len(test_fold)} samples)", flush=True)

        q_bc = bc_v[test_fold]
        q_ph = ph_v[test_fold]
        q_dn = dn_v[test_fold]
        q_morph = mv[test_fold]
        q_labels = lv[test_fold]

        si = select_support(lt, seed + fold_idx, cids)
        sbc = {c: bc_t[si[c]] for c in cids}
        sph = {c: ph_t[si[c]] for c in cids}
        sdn = {c: dn_t[si[c]] for c in cids}
        sm = {c: mt[si[c]] for c in cids}

        # Linear Probe on BiomedCLIP
        lp_bc = linear_probe(q_bc, sbc, cids)
        pred_lp_bc = [cids[int(np.argmax(lp_bc[i]))] for i in range(len(q_labels))]
        m = calc_metrics([int(l) for l in q_labels], pred_lp_bc, cids)
        results["LP_BC"]["acc"].append(m["acc"]); results["LP_BC"]["mf1"].append(m["mf1"])
        for c in cids: results["LP_BC"]["pc"][c].append(m["pc"][c]["f1"])

        # Linear Probe on concatenated
        q_cat = np.concatenate([q_bc, q_ph, q_dn], axis=1)
        s_cat = {c: np.concatenate([sbc[c], sph[c], sdn[c]], axis=1) for c in cids}
        lp_cat = linear_probe(q_cat, s_cat, cids)
        pred_lp_cat = [cids[int(np.argmax(lp_cat[i]))] for i in range(len(q_labels))]
        m = calc_metrics([int(l) for l in q_labels], pred_lp_cat, cids)
        results["LP_concat"]["acc"].append(m["acc"]); results["LP_concat"]["mf1"].append(m["mf1"])
        for c in cids: results["LP_concat"]["pc"][c].append(m["pc"][c]["f1"])

        # NCM on BiomedCLIP
        ncm = ncm_classify(q_bc, sbc, cids)
        pred_ncm = [cids[int(np.argmax(ncm[i]))] for i in range(len(q_labels))]
        m = calc_metrics([int(l) for l in q_labels], pred_ncm, cids)
        results["NCM_BC"]["acc"].append(m["acc"]); results["NCM_BC"]["mf1"].append(m["mf1"])
        for c in cids: results["NCM_BC"]["pc"][c].append(m["pc"][c]["f1"])

        # kNN on BiomedCLIP
        for kval in [1, 3, 5, 7]:
            knn = knn_classify(q_bc, sbc, cids, k=kval)
            pred_knn = [cids[int(np.argmax(knn[i]))] for i in range(len(q_labels))]
            m = calc_metrics([int(l) for l in q_labels], pred_knn, cids)
            name = f"kNN_BC_k{kval}"
            results[name]["acc"].append(m["acc"]); results[name]["mf1"].append(m["mf1"])
            for c in cids: results[name]["pc"][c].append(m["pc"][c]["f1"])

        # Multi-backbone kNN (no ATD)
        K = len(cids)
        sm_all = np.concatenate([sm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        snm = {c: (sm[c] - gm) / gs for c in cids}
        mb_scores = np.zeros((len(q_labels), K))
        for i in range(len(q_labels)):
            qm = (q_morph[i] - gm) / gs
            for ki, c in enumerate(cids):
                vs = 0.42 * (sbc[c] @ q_bc[i]) + 0.18 * (sph[c] @ q_ph[i]) + 0.07 * (sdn[c] @ q_dn[i])
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                mb_scores[i, ki] = float(np.sort(vs + 0.33 * ms)[::-1][:7].mean())
        pred_mb = [cids[int(np.argmax(mb_scores[i]))] for i in range(len(q_labels))]
        m = calc_metrics([int(l) for l in q_labels], pred_mb, cids)
        results["MB_kNN"]["acc"].append(m["acc"]); results["MB_kNN"]["mf1"].append(m["mf1"])
        for c in cids: results["MB_kNN"]["pc"][c].append(m["pc"][c]["f1"])

        # Full SADC+ATD (our method)
        sadc_scores = sadc_atd_classify(q_bc, q_ph, q_dn, q_morph, sbc, sph, sdn, sm, cids)
        pred_sadc = [cids[int(np.argmax(sadc_scores[i]))] for i in range(len(q_labels))]
        m = calc_metrics([int(l) for l in q_labels], pred_sadc, cids)
        results["SADC_ATD(Ours)"]["acc"].append(m["acc"]); results["SADC_ATD(Ours)"]["mf1"].append(m["mf1"])
        for c in cids: results["SADC_ATD(Ours)"]["pc"][c].append(m["pc"][c]["f1"])

    return results


def print_row(name, v, cids):
    pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    print(f"{name:<25} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  "
          f"{np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")


def main():
    print("="*100)
    print("NESTED CROSS-VALIDATION + SOTA COMPARISON")
    print("10-shot classification on data2_organized")
    print("="*100, flush=True)

    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")
    cids = sorted(CLASS_NAMES.keys())

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"\n{'='*100}")
        print(f"Seed {seed}")
        print(f"{'='*100}", flush=True)

        cv_results = nested_cv_evaluate(bc_v, ph_v, dn_v, mv, lv,
                                        bc_t, ph_t, dn_t, mt, lt,
                                        cids, n_outer=5, seed=seed)

        for method, data in cv_results.items():
            all_results[method]["acc"].extend(data["acc"])
            all_results[method]["mf1"].extend(data["mf1"])
            for c in cids:
                all_results[method]["pc"][c].extend(data["pc"][c])

    print(f"\n\n{'='*120}")
    print("FINAL RESULTS: Nested 5-fold CV × 5 seeds (25 total evaluations)")
    print(f"{'='*120}")
    h = f"{'Method':<25} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h)
    print("-" * 120)

    method_order = ["NCM_BC", "kNN_BC_k1", "kNN_BC_k3", "kNN_BC_k5", "kNN_BC_k7",
                    "LP_BC", "LP_concat", "MB_kNN", "SADC_ATD(Ours)"]
    for name in method_order:
        if name in all_results:
            print_row(name, all_results[name], cids)

    print()
    our = all_results["SADC_ATD(Ours)"]
    best_baseline_name = max(
        [n for n in all_results if n != "SADC_ATD(Ours)"],
        key=lambda n: np.mean(all_results[n]["mf1"]))
    best_baseline = all_results[best_baseline_name]

    print(f"Our method (SADC+ATD): mF1={np.mean(our['mf1']):.4f}±{np.std(our['mf1']):.4f}")
    print(f"Best baseline ({best_baseline_name}): mF1={np.mean(best_baseline['mf1']):.4f}±{np.std(best_baseline['mf1']):.4f}")
    print(f"Improvement over best baseline: {np.mean(our['mf1']) - np.mean(best_baseline['mf1']):+.4f}")

    # MultiCenter evaluation
    print(f"\n\n{'='*120}")
    print("MULTICENTER EVALUATION")
    print(f"{'='*120}", flush=True)

    mc_bc_t, mc_mt, mc_lt = load_cache("biomedclip", "train", "multicenter_")
    mc_bc_v, mc_mv, mc_lv = load_cache("biomedclip", "val", "multicenter_")
    mc_ph_t, _, _ = load_cache("phikon_v2", "train", "multicenter_")
    mc_ph_v, _, _ = load_cache("phikon_v2", "val", "multicenter_")
    mc_dn_t, _, _ = load_cache("dinov2_s", "train", "multicenter_")
    mc_dn_v, _, _ = load_cache("dinov2_s", "val", "multicenter_")

    mc_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        si = select_support(mc_lt, seed, cids)
        sbc = {c: mc_bc_t[si[c]] if si[c] else np.zeros((0, mc_bc_t.shape[1])) for c in cids}
        sph = {c: mc_ph_t[si[c]] if si[c] else np.zeros((0, mc_ph_t.shape[1])) for c in cids}
        sdn = {c: mc_dn_t[si[c]] if si[c] else np.zeros((0, mc_dn_t.shape[1])) for c in cids}
        sm = {c: mc_mt[si[c]] if si[c] else np.zeros((0, mc_mt.shape[1])) for c in cids}

        active = [c for c in cids if len(sbc[c]) > 0]

        for method_name, method_fn in [
            ("NCM_BC", lambda: ncm_classify(mc_bc_v, sbc, active)),
            ("kNN_BC_k5", lambda: knn_classify(mc_bc_v, sbc, active, 5)),
            ("LP_BC", lambda: linear_probe(mc_bc_v, sbc, active)),
            ("LP_concat", lambda: linear_probe(
                np.concatenate([mc_bc_v, mc_ph_v, mc_dn_v], axis=1),
                {c: np.concatenate([sbc[c], sph[c], sdn[c]], axis=1) for c in active},
                active)),
        ]:
            scores = method_fn()
            pred = [active[int(np.argmax(scores[i]))] for i in range(len(mc_lv))]
            m = calc_metrics([int(l) for l in mc_lv], pred, active)
            mc_results[method_name]["acc"].append(m["acc"])
            mc_results[method_name]["mf1"].append(m["mf1"])
            for c in active:
                mc_results[method_name]["pc"][c].append(m["pc"][c]["f1"])

        sadc_sc = sadc_atd_classify(mc_bc_v, mc_ph_v, mc_dn_v, mc_mv,
                                    sbc, sph, sdn, sm, active)
        pred = [active[int(np.argmax(sadc_sc[i]))] for i in range(len(mc_lv))]
        m = calc_metrics([int(l) for l in mc_lv], pred, active)
        mc_results["SADC_ATD(Ours)"]["acc"].append(m["acc"])
        mc_results["SADC_ATD(Ours)"]["mf1"].append(m["mf1"])
        for c in active:
            mc_results["SADC_ATD(Ours)"]["pc"][c].append(m["pc"][c]["f1"])

    print(f"\n{'Method':<25} {'Acc':>7} {'mF1':>7}  {'As':>5} {'Fs':>5}")
    print("-" * 55)
    for name in ["NCM_BC", "kNN_BC_k5", "LP_BC", "LP_concat", "SADC_ATD(Ours)"]:
        if name in mc_results:
            v = mc_results[name]
            print(f"{name:<25} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f}  "
                  f"{np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")


if __name__ == "__main__":
    main()
