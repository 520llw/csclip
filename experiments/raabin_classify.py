#!/usr/bin/env python3
"""
Raabin-WBC-top3 evaluation: MB-kNN baseline vs. AFP-OD P3c.

Data: /home/xut/csclip/experiments/feature_cache/raabin_{biomedclip,phikon_v2,dinov2_s}_all.npz
Classes: 0=Eosinophil, 1=Lymphocyte, 2=Neutrophil
Protocol: 5-fold CV x 5 seeds; 10-shot support drawn from the other 4 folds.
"""
from __future__ import annotations
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/xut/csclip/experiments")

from afpod_classify import (
    _ledoit_wolf_shrinkage, fisher_direction,
    find_confusion_pairs_loo, find_confusion_pairs_dualview,
    amplify_separation, afpod_classify, mb_knn_classify,
    calc_metrics, print_row,
)

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {0: "Eosinophil", 1: "Lymphocyte", 2: "Neutrophil"}
CIDS = [0, 1, 2]
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]
N_FOLDS = 5


def load_all(name):
    d = np.load(CACHE_DIR / f"raabin_{name}_all.npz")
    return d["feats"], d["morphs"], d["labels"]


def group_by_class(feats, morphs, labels, indices):
    per_bc = {}
    per_m = {}
    per_lbl = {}
    for c in CIDS:
        idx = [i for i in indices if int(labels[i]) == c]
        per_bc[c] = feats[idx] if idx else np.zeros((0, feats.shape[1]), dtype=feats.dtype)
        per_m[c] = morphs[idx] if idx else np.zeros((0, morphs.shape[1]), dtype=morphs.dtype)
        per_lbl[c] = idx
    return per_bc, per_m, per_lbl


def select_support_indices(labels, train_idx, seed, n_shot=N_SHOT):
    random.seed(seed)
    pc = defaultdict(list)
    for i in train_idx:
        pc[int(labels[i])].append(i)
    out = {}
    for c in CIDS:
        pool = pc[c]
        if len(pool) <= n_shot:
            out[c] = list(pool)
        else:
            out[c] = random.sample(pool, n_shot)
    return out


def split_folds(all_indices, seed, n_folds=N_FOLDS):
    rng = np.random.RandomState(seed)
    shuf = list(all_indices)
    rng.shuffle(shuf)
    fold_size = len(shuf) // n_folds
    folds = []
    for i in range(n_folds):
        s = i * fold_size
        e = s + fold_size if i < n_folds - 1 else len(shuf)
        folds.append(shuf[s:e])
    return folds


def build_support(bc_f, ph_f, dn_f, mo_f, labels, support_idx_per_class):
    s_bc, s_ph, s_dn, s_m = {}, {}, {}, {}
    for c in CIDS:
        idxs = support_idx_per_class[c]
        s_bc[c] = bc_f[idxs]
        s_ph[c] = ph_f[idxs]
        s_dn[c] = dn_f[idxs]
        s_m[c] = mo_f[idxs]
    return s_bc, s_ph, s_dn, s_m


def main():
    print("=" * 110)
    print("Raabin-WBC-top3 evaluation: MB-kNN baseline vs AFP-OD P3c")
    print(f"Protocol: {N_FOLDS}-fold CV x {len(SEEDS)} seeds = {N_FOLDS * len(SEEDS)} evaluations")
    print(f"Classes: {CLASS_NAMES}")
    print(f"N-shot = {N_SHOT}")
    print("=" * 110, flush=True)

    print("\nLoading feature caches...")
    bc, mb, lb = load_all("biomedclip")
    ph, mp, lp = load_all("phikon_v2")
    dn, md, ld = load_all("dinov2_s")

    assert np.array_equal(lb, lp) and np.array_equal(lp, ld), "Label mismatch"
    assert np.array_equal(mb, mp) and np.array_equal(mp, md), "Morph mismatch"

    labels = lb
    morphs = mb
    N = len(labels)

    from collections import Counter
    print(f"  Total cells: {N}")
    print(f"  Class counts: {dict(Counter(int(l) for l in labels))}")
    print(f"  BC {bc.shape}, PH {ph.shape}, DN {dn.shape}, morph {morphs.shape}")

    all_idx = list(range(N))
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": {c: [] for c in CIDS}})

    configs = [
        # (method, shrink, alpha, alpha_blend, detection_mode, phase)
        ("lw", None, 0.10, 0.0, "feature_only", "p2"),
        ("lw", None, 0.10, 0.0, "dualview_union", "p3c"),
    ]

    eval_count = 0
    total_evals = len(SEEDS) * N_FOLDS
    for seed in SEEDS:
        folds = split_folds(all_idx, seed=seed)
        for fold_i in range(N_FOLDS):
            eval_count += 1
            query_idx = folds[fold_i]
            train_idx = [i for f_j, f in enumerate(folds) if f_j != fold_i for i in f]

            support_idx = select_support_indices(labels, train_idx, seed=seed * 10 + fold_i)

            q_bc = bc[query_idx]; q_ph = ph[query_idx]; q_dn = dn[query_idx]
            q_m = morphs[query_idx]
            q_labels = labels[query_idx]

            s_bc, s_ph, s_dn, s_m = build_support(bc, ph, dn, morphs, labels, support_idx)

            # MB-kNN baseline
            mb_scores = mb_knn_classify(q_bc, q_ph, q_dn, q_m,
                                        s_bc, s_ph, s_dn, s_m, CIDS)
            mb_pred = [CIDS[int(np.argmax(mb_scores[i]))] for i in range(len(q_labels))]
            mb_m = calc_metrics([int(l) for l in q_labels], mb_pred, CIDS)
            all_results["MB_kNN"]["acc"].append(mb_m["acc"])
            all_results["MB_kNN"]["mf1"].append(mb_m["mf1"])
            for c in CIDS:
                all_results["MB_kNN"]["pc"][c].append(mb_m["pc"][c]["f1"])

            # AFP-OD variants
            for (method, shrink, alpha, alpha_blend, det_mode, phase) in configs:
                scores, _ = afpod_classify(
                    q_bc, q_ph, q_dn, q_m, s_bc, s_ph, s_dn, s_m, CIDS,
                    alpha=alpha, conf_thresh=0.15,
                    method=method, shrink=shrink if shrink is not None else 0.3,
                    alpha_blend=alpha_blend,
                    detection_mode=det_mode)
                pred = [CIDS[int(np.argmax(scores[i]))] for i in range(len(q_labels))]
                m = calc_metrics([int(l) for l in q_labels], pred, CIDS)
                if phase == "p2":
                    name = "AFPOD_p2_lw_a0.10"
                elif phase == "p3c":
                    name = "AFPOD_p3c_dv_union_a0.10"
                else:
                    name = f"AFPOD_{phase}_a{alpha:.2f}"
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in CIDS:
                    all_results[name]["pc"][c].append(m["pc"][c]["f1"])

            if eval_count % 5 == 0 or eval_count == total_evals:
                v = all_results["MB_kNN"]
                p2 = all_results["AFPOD_p2_lw_a0.10"]
                p3c = all_results["AFPOD_p3c_dv_union_a0.10"]
                eos_idx = 0  # class 0 is Eosinophil in Raabin
                print(f"  [{eval_count}/{total_evals}] seed={seed} fold={fold_i}  "
                      f"MB: mF1={np.mean(v['mf1']):.4f}, Eos={np.mean(v['pc'][eos_idx]):.4f}  | "
                      f"P2: mF1={np.mean(p2['mf1']):.4f}, Eos={np.mean(p2['pc'][eos_idx]):.4f}  | "
                      f"P3c: mF1={np.mean(p3c['mf1']):.4f}, Eos={np.mean(p3c['pc'][eos_idx]):.4f}",
                      flush=True)

    print("\n" + "=" * 110)
    print("Final Results — Raabin-WBC-top3")
    print("=" * 110)
    header_classes = "  ".join(f"{CLASS_NAMES[c][:6]:>7}" for c in CIDS)
    print(f"{'Method':<28} {'Acc':>7} {'mF1':>7} {header_classes}   std_a  std_f1")
    print("-" * 110)
    print_row("MB_kNN (baseline)", all_results["MB_kNN"], CIDS)
    print_row("AFPOD_p2_lw_a0.10", all_results["AFPOD_p2_lw_a0.10"], CIDS)
    print_row("AFPOD_p3c_dv_union_a0.10", all_results["AFPOD_p3c_dv_union_a0.10"], CIDS)

    bl_mf1 = np.mean(all_results["MB_kNN"]["mf1"])
    bl_eos = np.mean(all_results["MB_kNN"]["pc"][0])
    print("\nImprovement over MB_kNN baseline:")
    for name in ["AFPOD_p2_lw_a0.10", "AFPOD_p3c_dv_union_a0.10"]:
        v = all_results[name]
        d_mf1 = np.mean(v["mf1"]) - bl_mf1
        d_eos = np.mean(v["pc"][0]) - bl_eos
        print(f"  {name}: ΔmF1 = {d_mf1:+.4f}, ΔEos F1 = {d_eos:+.4f}")


if __name__ == "__main__":
    main()
