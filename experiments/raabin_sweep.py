#!/usr/bin/env python3
"""
Raabin-WBC-top3 hyperparameter grid search.
Sweep alpha in {0.05, 0.10, 0.15, 0.20} x tau in {0.10, 0.15, 0.20}.
Compare against MB-kNN baseline using 5-fold CV x 5 seeds (25 evals).
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/xut/csclip/experiments")

from afpod_classify import afpod_classify, mb_knn_classify, calc_metrics
from raabin_classify import (
    CIDS, CLASS_NAMES, N_SHOT, SEEDS, N_FOLDS,
    load_all, split_folds, select_support_indices, build_support,
)


def main():
    print("=" * 110)
    print("Raabin-WBC-top3 hyperparameter sweep: alpha x tau")
    print(f"Grid: alpha in [0.05, 0.10, 0.15, 0.20], tau in [0.10, 0.15, 0.20]")
    print("=" * 110, flush=True)

    bc, _, lb = load_all("biomedclip")
    ph, _, lp = load_all("phikon_v2")
    dn, mo, ld = load_all("dinov2_s")
    labels = lb
    morphs = mo
    N = len(labels)
    print(f"Loaded {N} cells. Classes: {CLASS_NAMES}", flush=True)

    alphas = [0.05, 0.10, 0.15, 0.20]
    taus = [0.10, 0.15, 0.20]
    all_idx = list(range(N))

    results = {}  # (alpha, tau) -> defaultdict of lists
    baseline_res = {"acc": [], "mf1": [], "pc": {c: [] for c in CIDS}}

    for (alpha, tau) in [(a, t) for a in alphas for t in taus]:
        results[(alpha, tau)] = {"acc": [], "mf1": [], "pc": {c: [] for c in CIDS}}

    eval_count = 0
    total = len(SEEDS) * N_FOLDS
    for seed in SEEDS:
        folds = split_folds(all_idx, seed=seed)
        for fold_i in range(N_FOLDS):
            eval_count += 1
            query_idx = folds[fold_i]
            train_idx = [i for f_j, f in enumerate(folds) if f_j != fold_i for i in f]
            support_idx = select_support_indices(labels, train_idx, seed=seed * 10 + fold_i)

            q_bc, q_ph, q_dn = bc[query_idx], ph[query_idx], dn[query_idx]
            q_m = morphs[query_idx]
            q_labels = labels[query_idx]
            s_bc, s_ph, s_dn, s_m = build_support(bc, ph, dn, morphs, labels, support_idx)

            # Baseline (once per fold)
            sc = mb_knn_classify(q_bc, q_ph, q_dn, q_m, s_bc, s_ph, s_dn, s_m, CIDS)
            pred = [CIDS[int(np.argmax(sc[i]))] for i in range(len(q_labels))]
            m = calc_metrics([int(l) for l in q_labels], pred, CIDS)
            baseline_res["acc"].append(m["acc"])
            baseline_res["mf1"].append(m["mf1"])
            for c in CIDS:
                baseline_res["pc"][c].append(m["pc"][c]["f1"])

            # Grid
            for (alpha, tau) in [(a, t) for a in alphas for t in taus]:
                sc, _ = afpod_classify(
                    q_bc, q_ph, q_dn, q_m, s_bc, s_ph, s_dn, s_m, CIDS,
                    alpha=alpha, conf_thresh=tau,
                    method="lw", alpha_blend=0.0,
                    detection_mode="dualview_union")
                pred = [CIDS[int(np.argmax(sc[i]))] for i in range(len(q_labels))]
                m = calc_metrics([int(l) for l in q_labels], pred, CIDS)
                r = results[(alpha, tau)]
                r["acc"].append(m["acc"])
                r["mf1"].append(m["mf1"])
                for c in CIDS:
                    r["pc"][c].append(m["pc"][c]["f1"])

            if eval_count % 5 == 0 or eval_count == total:
                print(f"  [{eval_count}/{total}] done", flush=True)

    print("\n" + "=" * 110)
    print(f"{'Config':<18} {'mF1':>8} {'Eos F1':>8} {'Lym F1':>8} {'Neu F1':>8} {'ΔmF1':>8} {'ΔEos':>8}")
    print("-" * 110)
    bl_mf1 = np.mean(baseline_res["mf1"])
    bl_eos = np.mean(baseline_res["pc"][0])
    print(f"{'MB-kNN (baseline)':<18} {bl_mf1:>8.4f} {bl_eos:>8.4f} "
          f"{np.mean(baseline_res['pc'][1]):>8.4f} {np.mean(baseline_res['pc'][2]):>8.4f} "
          f"{'—':>8} {'—':>8}")

    best = None
    for alpha in alphas:
        for tau in taus:
            r = results[(alpha, tau)]
            mf1 = np.mean(r["mf1"])
            eos = np.mean(r["pc"][0])
            lym = np.mean(r["pc"][1])
            neu = np.mean(r["pc"][2])
            tag = f"a={alpha:.2f}, τ={tau:.2f}"
            print(f"{tag:<18} {mf1:>8.4f} {eos:>8.4f} {lym:>8.4f} {neu:>8.4f} "
                  f"{mf1 - bl_mf1:>+8.4f} {eos - bl_eos:>+8.4f}")
            if best is None or mf1 > best[0]:
                best = (mf1, alpha, tau, eos, lym, neu)

    print("-" * 110)
    print(f"Best config: alpha={best[1]:.2f}, tau={best[2]:.2f}")
    print(f"  mF1 = {best[0]:.4f} (Δ {best[0]-bl_mf1:+.4f})")
    print(f"  Eos = {best[3]:.4f} (Δ {best[3]-bl_eos:+.4f})")
    print(f"  Lym = {best[4]:.4f}, Neu = {best[5]:.4f}")


if __name__ == "__main__":
    main()
