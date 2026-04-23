#!/usr/bin/env python3
"""
N-shot ablation: 1/3/5/10/20-shot classification performance.
Also: backbone ablation (which backbone combinations matter).
Two key experiments for paper:
1. How classification improves with more shots → justifies 10-shot
2. Which backbone combinations contribute most → justifies 3-backbone fusion
"""
import sys, random
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
SEEDS = [42, 123, 456, 789, 2026]


def load_cache(model, split):
    d = np.load(CACHE_DIR / f"{model}_{split}.npz")
    return d["feats"], d["morphs"], d["labels"]


def select_support(labels, seed, cids, n_shot):
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


def classify_multi_backbone(q_feats_dict, s_feats_dict, q_morph, s_morph,
                            cids, backbones, weights, mw=0.33, k=7,
                            use_atd=True, n_iter=2, top_k_pseudo=5, conf_thr=0.025):
    """Flexible multi-backbone classifier. backbones = list of backbone names."""
    K = len(cids)
    N_q = len(q_morph)
    q_labels_dummy = np.zeros(N_q)

    sb = {c: {bb: s_feats_dict[bb][c].copy() for bb in backbones} for c in cids}
    smm = {c: s_morph[c].copy() for c in cids}
    sb_orig = {c: {bb: s_feats_dict[bb][c].copy() for bb in backbones} for c in cids}
    smm_orig = {c: s_morph[c].copy() for c in cids}

    for it in range(n_iter if use_atd else 1):
        sm_all = np.concatenate([smm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        snm = {c: (smm[c] - gm) / gs for c in cids}

        all_scores = np.zeros((N_q, K))
        for i in range(N_q):
            qm = (q_morph[i] - gm) / gs
            for ki, c in enumerate(cids):
                vis_score = 0.0
                for bb, w in zip(backbones, weights):
                    sims = sb[c][bb] @ q_feats_dict[bb][i]
                    vis_score += w * float(np.sort(sims)[::-1][:min(k, len(sims))].mean())
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = float(np.mean(1.0 / (1.0 + np.sort(md)[:min(5, len(md))])))
                all_scores[i, ki] = vis_score + mw * ms

        if use_atd and it < n_iter - 1:
            preds = np.array([cids[int(np.argmax(all_scores[i]))] for i in range(N_q)])
            margins = np.array([np.sort(all_scores[i])[::-1][0] - np.sort(all_scores[i])[::-1][1] for i in range(N_q)])
            for c in cids:
                cm = (preds == c) & (margins > conf_thr)
                ci = np.where(cm)[0]
                if len(ci) == 0:
                    continue
                bb0 = backbones[0]
                proto_c = sb_orig[c][bb0].mean(0)
                dists = np.array([np.linalg.norm(q_feats_dict[bb0][idx] - proto_c) for idx in ci])
                diversity = margins[ci] * (1.0 + 0.3 * dists / (dists.mean() + 1e-8))
                ti = ci[np.argsort(diversity)[::-1][:top_k_pseudo]]
                for bb in backbones:
                    sb[c][bb] = np.concatenate([sb_orig[c][bb], q_feats_dict[bb][ti] * 0.5])
                smm[c] = np.concatenate([smm_orig[c], q_morph[ti]])

    preds = [cids[int(np.argmax(all_scores[i]))] for i in range(N_q)]
    return preds


def main():
    print("Loading features...", flush=True)
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")
    cids = sorted(CLASS_NAMES.keys())

    q_feats = {"bc": bc_v, "ph": ph_v, "dn": dn_v}
    t_feats = {"bc": bc_t, "ph": ph_t, "dn": dn_t}

    # ==================== Experiment 1: N-shot ablation ====================
    print("\n" + "="*120)
    print("EXPERIMENT 1: N-shot Ablation (full 3-backbone + ATD)")
    print("="*120, flush=True)

    n_shots = [1, 3, 5, 10, 20]
    nshot_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for ns in n_shots:
        for seed in SEEDS:
            si = select_support(lt, seed, cids, ns)
            s_feats = {
                "bc": {c: bc_t[si[c]] for c in cids},
                "ph": {c: ph_t[si[c]] for c in cids},
                "dn": {c: dn_t[si[c]] for c in cids},
            }
            sm = {c: mt[si[c]] for c in cids}

            preds = classify_multi_backbone(
                q_feats, s_feats, mv, sm, cids,
                backbones=["bc", "ph", "dn"],
                weights=[0.42, 0.18, 0.07],
                mw=0.33, use_atd=(ns >= 3))

            m = calc_metrics([int(l) for l in lv], preds, cids)
            key = f"{ns}-shot"
            nshot_results[key]["acc"].append(m["acc"])
            nshot_results[key]["mf1"].append(m["mf1"])
            for c in cids:
                nshot_results[key]["pc"][c].append(m["pc"][c]["f1"])

        v = nshot_results[f"{ns}-shot"]
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"  {ns:>2}-shot: Acc={np.mean(v['acc']):.4f}±{np.std(v['acc']):.3f} "
              f"mF1={np.mean(v['mf1']):.4f}±{np.std(v['mf1']):.3f} "
              f"[Eos Neu Lym Mac]: {pc_str}", flush=True)

    # ==================== Experiment 2: Backbone ablation ====================
    print("\n" + "="*120)
    print("EXPERIMENT 2: Backbone Ablation (10-shot)")
    print("="*120, flush=True)

    bb_configs = [
        ("BC only",           ["bc"],             [1.0],              0.33),
        ("PH only",           ["ph"],             [1.0],              0.33),
        ("DN only",           ["dn"],             [1.0],              0.33),
        ("BC+PH",             ["bc", "ph"],       [0.65, 0.35],       0.33),
        ("BC+DN",             ["bc", "dn"],       [0.80, 0.20],       0.33),
        ("PH+DN",             ["ph", "dn"],       [0.70, 0.30],       0.33),
        ("BC+PH+DN",          ["bc", "ph", "dn"], [0.42, 0.18, 0.07], 0.33),
        ("BC+PH+DN (no morph)", ["bc", "ph", "dn"], [0.50, 0.30, 0.20], 0.00),
        ("BC+PH+DN+ATD",     ["bc", "ph", "dn"], [0.42, 0.18, 0.07], 0.33),
    ]

    bb_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for name, bbs, ws, mw in bb_configs:
        use_atd = "+ATD" in name
        for seed in SEEDS:
            si = select_support(lt, seed, cids, 10)
            s_feats = {}
            for bb in ["bc", "ph", "dn"]:
                s_feats[bb] = {c: t_feats[bb][si[c]] for c in cids}
            sm = {c: mt[si[c]] for c in cids}

            preds = classify_multi_backbone(
                q_feats, s_feats, mv, sm, cids,
                backbones=bbs, weights=ws, mw=mw, use_atd=use_atd)

            m = calc_metrics([int(l) for l in lv], preds, cids)
            bb_results[name]["acc"].append(m["acc"])
            bb_results[name]["mf1"].append(m["mf1"])
            for c in cids:
                bb_results[name]["pc"][c].append(m["pc"][c]["f1"])

        v = bb_results[name]
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"  {name:<25} Acc={np.mean(v['acc']):.4f}±{np.std(v['acc']):.3f} "
              f"mF1={np.mean(v['mf1']):.4f}±{np.std(v['mf1']):.3f} "
              f"[Eos Neu Lym Mac]: {pc_str}", flush=True)

    # ==================== Summary Tables ====================
    print("\n\n" + "="*120)
    print("SUMMARY TABLE: N-shot Ablation")
    print("="*120)
    print(f"{'N-shot':<10} {'Acc':>12} {'mF1':>12} {'Eos F1':>10} {'Neu F1':>10} {'Lym F1':>10} {'Mac F1':>10} {'Annotations':>12}")
    print("-" * 90)
    for ns in n_shots:
        v = nshot_results[f"{ns}-shot"]
        n_ann = ns * len(cids)
        pct = (1 - n_ann / 5315) * 100
        print(f"{ns:<10} {np.mean(v['acc']):>7.4f}±{np.std(v['acc']):.3f} "
              f"{np.mean(v['mf1']):>7.4f}±{np.std(v['mf1']):.3f} "
              f"{np.mean(v['pc'][3]):>10.4f} {np.mean(v['pc'][4]):>10.4f} "
              f"{np.mean(v['pc'][5]):>10.4f} {np.mean(v['pc'][6]):>10.4f} "
              f"{n_ann:>5} ({pct:.1f}%)")

    print("\n" + "="*120)
    print("SUMMARY TABLE: Backbone Ablation")
    print("="*120)
    print(f"{'Configuration':<25} {'Acc':>12} {'mF1':>12} {'Eos F1':>10} {'Neu F1':>10} {'Lym F1':>10} {'Mac F1':>10}")
    print("-" * 95)
    for name, _, _, _ in bb_configs:
        v = bb_results[name]
        print(f"{name:<25} {np.mean(v['acc']):>7.4f}±{np.std(v['acc']):.3f} "
              f"{np.mean(v['mf1']):>7.4f}±{np.std(v['mf1']):.3f} "
              f"{np.mean(v['pc'][3]):>10.4f} {np.mean(v['pc'][4]):>10.4f} "
              f"{np.mean(v['pc'][5]):>10.4f} {np.mean(v['pc'][6]):>10.4f}")


if __name__ == "__main__":
    main()
