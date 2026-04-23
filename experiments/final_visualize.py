#!/usr/bin/env python3
"""
Final visualization: segmentation comparison, classification confusion matrix,
ablation charts, and per-class performance bars.
"""
import sys, random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(line_buffering=True)

CACHE = Path("/home/xut/csclip/experiments/feature_cache")
OUT = Path("/home/xut/csclip/experiments/figures")
CLASSES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
cids = sorted(CLASSES.keys())
SEEDS = [42, 123, 456, 789, 2026]


def run_atd(q_bc, q_ph, q_dn, q_morph, q_labels, s_bc, s_ph, s_dn, s_morph, cids,
            bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7, n_iter=2):
    sb = {c: s_bc[c].copy() for c in cids}
    sp = {c: s_ph[c].copy() for c in cids}
    sd = {c: s_dn[c].copy() for c in cids}
    sm = {c: s_morph[c].copy() for c in cids}
    sb_orig = {c: s_bc[c].copy() for c in cids}

    for it in range(n_iter):
        sm_all = np.concatenate([sm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        snm = {c: (sm[c] - gm) / gs for c in cids}
        preds, margins = [], []
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
        preds = np.array(preds)
        margins = np.array(margins)
        for c in cids:
            cm = (preds == c) & (margins > 0.025)
            ci = np.where(cm)[0]
            if len(ci) == 0: continue
            proto_c = sb_orig[c].mean(0)
            dists = np.array([np.linalg.norm(q_bc[idx] - proto_c) for idx in ci])
            div_scores = margins[ci] * (1.0 + 0.3 * dists / (dists.mean() + 1e-8))
            ti = ci[np.argsort(div_scores)[::-1][:5]]
            sb[c] = np.concatenate([sb_orig[c], q_bc[ti] * 0.5])
            sp[c] = np.concatenate([{c: s_ph[c].copy() for c in cids}[c], q_ph[ti] * 0.5])
            sd[c] = np.concatenate([{c: s_dn[c].copy() for c in cids}[c], q_dn[ti] * 0.5])
            sm[c] = np.concatenate([{c: s_morph[c].copy() for c in cids}[c], q_morph[ti]])

    sm2 = np.concatenate([sm[c] for c in cids])
    gm2, gs2 = sm2.mean(0), sm2.std(0) + 1e-8
    snm = {c: (sm[c] - gm2) / gs2 for c in cids}
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm2) / gs2
        scores = {}
        for c in cids:
            vs = bw * (sb[c] @ q_bc[i]) + pw * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            scores[c] = float(np.sort(vs + mw * ms)[::-1][:k].mean())
        gt.append(int(q_labels[i]))
        pred.append(max(scores, key=scores.get))
    return gt, pred


def plot_confusion_matrix(gt, pred, title, path):
    names = [CLASSES[c] for c in cids]
    n = len(cids)
    mat = np.zeros((n, n), dtype=int)
    for g, p in zip(gt, pred):
        gi = cids.index(g)
        pi = cids.index(p)
        mat[gi, pi] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap='Blues')
    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
            ax.text(j, i, str(mat[i, j]), ha='center', va='center', color=color, fontsize=14)
    ax.set_xticks(range(n))
    ax.set_xticklabels([n[:3] for n in names], fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels([n[:3] for n in names], fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_ablation_bars():
    methods = ['Baseline\n(no leak)', '+ATD', 'Leaky\nbaseline']
    mf1 = [0.7510, 0.7518, 0.7550]
    eos = [0.4388, 0.4404, 0.4658]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colors = ['#4C72B0', '#DD8452', '#C44E52']
    x = np.arange(len(methods))

    bars1 = ax1.bar(x, mf1, 0.5, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylim(0.70, 0.77)
    ax1.set_ylabel('Macro-F1', fontsize=12)
    ax1.set_title('Classification: Macro-F1', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    for bar, val in zip(bars1, mf1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.axhline(y=0.7510, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    bars2 = ax2.bar(x, eos, 0.5, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylim(0.35, 0.52)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Classification: Eosinophil F1', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    for bar, val in zip(bars2, eos):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = OUT / "ablation_classification.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_segmentation_comparison():
    methods = ['Default\nCellposeSAM', 'Optimized\n(d=50,cp=-3)', 'PAMSR']

    d2_f1 = [0.5148, 0.7261, 0.7276]
    mc_f1 = [0.4122, 0.5517, 0.5490]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, d2_f1, width, label='data2', color='#4C72B0',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, mc_f1, width, label='MultiCenter', color='#DD8452',
                   edgecolor='black', linewidth=0.5)

    ax.set_ylim(0, 0.85)
    ax.set_ylabel('Segmentation F1', fontsize=13)
    ax.set_title('Cell Segmentation: Cross-Dataset Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                   f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.annotate('+41.3%', xy=(1, 0.73), fontsize=11, color='#4C72B0', fontweight='bold',
                ha='center')
    ax.annotate('+33.8%', xy=(1.35, 0.56), fontsize=11, color='#DD8452', fontweight='bold',
                ha='center')

    plt.tight_layout()
    path = OUT / "segmentation_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_class_f1():
    names = ['Eos', 'Neu', 'Lym', 'Mac']
    d2_f1 = [0.4404, 0.7523, 0.9334, 0.8811]
    mc_f1 = [0.0139, 0.5661, 0.2196, 0.4759]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, d2_f1, width, label='data2', color='#4C72B0',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, mc_f1, width, label='MultiCenter', color='#DD8452',
                   edgecolor='black', linewidth=0.5)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('Per-Class F1: Cross-Dataset Comparison (10-shot ATD)', fontsize=14,
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend(fontsize=11)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                       f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    path = OUT / "per_class_f1_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_pipeline_overview():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    stages = [
        (1.0, 2.0, 2.4, 1.6, "CellposeSAM\n+PAMSR", "#E8D5B7", "Segmentation\nF1=0.73"),
        (4.2, 2.0, 2.4, 1.6, "Triple Backbone\nFeature\nExtraction", "#B7D5E8", "BiomedCLIP\nPhikon-v2\nDINOv2"),
        (7.4, 2.0, 2.4, 1.6, "SADC+ATD\nClassification", "#D5E8B7", "10-shot\nmF1=0.75"),
        (10.6, 2.0, 2.4, 1.6, "Output", "#E8B7D5", "Cell Types\n+ Masks"),
    ]

    for x, y, w, h, label, color, sub in stages:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
               fontsize=10, fontweight='bold')
        ax.text(x + w/2, y - 0.3, sub, ha='center', va='top', fontsize=8, color='gray')

    for i in range(3):
        x1 = stages[i][0] + stages[i][2]
        x2 = stages[i+1][0]
        y = stages[i][1] + stages[i][3] / 2
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.text(0.2, 3.5, "BALF Cells", fontsize=9, color='gray')
    ax.annotate('', xy=(1.0, 2.8), xytext=(0.2, 3.2),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.set_title('BALF Cell Analysis Pipeline: Segmentation → Feature Extraction → Classification',
                fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    path = OUT / "pipeline_overview.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Generating confusion matrices ===")
    for ds_name, prefix in [("data2", ""), ("multicenter", "multicenter_")]:
        bc_t = np.load(CACHE / f"{prefix}biomedclip_train.npz")
        bc_v = np.load(CACHE / f"{prefix}biomedclip_val.npz")
        ph_t = np.load(CACHE / f"{prefix}phikon_v2_train.npz")
        ph_v = np.load(CACHE / f"{prefix}phikon_v2_val.npz")
        dn_t = np.load(CACHE / f"{prefix}dinov2_s_train.npz")
        dn_v = np.load(CACHE / f"{prefix}dinov2_s_val.npz")

        all_gt, all_pred = [], []
        for seed in SEEDS:
            random.seed(seed)
            pc = defaultdict(list)
            for i, l in enumerate(bc_t["labels"]):
                pc[int(l)].append(i)
            si = {c: random.sample(pc[c], min(10, len(pc[c]))) for c in cids}

            gt, pred = run_atd(
                bc_v["feats"], ph_v["feats"], dn_v["feats"], bc_v["morphs"], bc_v["labels"],
                {c: bc_t["feats"][si[c]] for c in cids},
                {c: ph_t["feats"][si[c]] for c in cids},
                {c: dn_t["feats"][si[c]] for c in cids},
                {c: bc_t["morphs"][si[c]] for c in cids}, cids)
            all_gt.extend(gt)
            all_pred.extend(pred)

        plot_confusion_matrix(all_gt, all_pred,
                            f"Confusion Matrix: {ds_name} (10-shot ATD, 5 seeds)",
                            OUT / f"confusion_{ds_name}.png")

    print("\n=== Generating comparison charts ===")
    plot_ablation_bars()
    plot_segmentation_comparison()
    plot_per_class_f1()
    plot_pipeline_overview()

    print("\nAll visualizations saved to", OUT)


if __name__ == "__main__":
    main()
