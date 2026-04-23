#!/usr/bin/env python3
"""
Comprehensive visualization of 10-shot classification and CellposeSAM results.
Generates: confusion matrix, per-class F1 bars, t-SNE feature space,
cell crop gallery with predictions, CellposeSAM segmentation overlay.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
OUT_DIR = Path("/home/xut/csclip/experiments/visualizations")
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
CLASS_COLORS = {3: '#e74c3c', 4: '#3498db', 5: '#2ecc71', 6: '#f39c12'}
CLASS_COLORS_BGR = {3: (60, 76, 231), 4: (219, 152, 52), 5: (113, 204, 46), 6: (18, 156, 243)}
N_SHOT = 10
SEED = 42


def load_cache(m, s):
    d = np.load(CACHE_DIR / f"{m}_{s}.npz")
    return d["feats"], d["morphs"], d["labels"]


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels):
        pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def load_yolo(lp, h, w):
    anns = []
    if not lp.exists():
        return anns
    for line in open(lp):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        if cid in CLASS_NAMES:
            pts = [float(x) for x in parts[1:]]
            xs = [pts[i] * w for i in range(0, len(pts), 2)]
            ys = [pts[i] * h for i in range(1, len(pts), 2)]
            anns.append({"class_id": cid, "xs": xs, "ys": ys})
    return anns


def run_10shot_with_predictions(seed=42):
    """Run the triple backbone + transductive + cascade, return (gt, pred) lists."""
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")

    cids = sorted(CLASS_NAMES.keys())
    support_idx = select_support(lt, seed, cids)

    s_bc = {c: bc_t[support_idx[c]] for c in cids}
    s_ph = {c: ph_t[support_idx[c]] for c in cids}
    s_dn = {c: dn_t[support_idx[c]] for c in cids}
    s_morph = {c: mt[support_idx[c]] for c in cids}

    sm = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm.mean(0), sm.std(0) + 1e-8

    # Fisher weights from support only (10 Eos + 10 Neu)
    eos_m, neu_m = s_morph[3], s_morph[4]
    n_dims = mt.shape[1]
    fw = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        mu_diff = (np.mean(eos_m[:, d]) - np.mean(neu_m[:, d])) ** 2
        var_sum = np.var(eos_m[:, d]) + np.var(neu_m[:, d]) + 1e-10
        fw[d] = 1.0 + (mu_diff / var_sum) * 2.0

    bw, pw, dw, mw = 0.42, 0.18, 0.07, 0.33
    k, n_iter, conf_thr, cascade_thr, cascade_mw = 7, 2, 0.025, 0.012, 0.45

    sb, sp, sd, smm = (
        {c: v[c].copy() for c in cids}
        for v in [s_bc, s_ph, s_dn, s_morph]
    )

    for _ in range(n_iter):
        snm = {c: (smm[c] - gm) / gs for c in cids}
        preds, margins = [], []
        for i in range(len(lv)):
            qm = (mv[i] - gm) / gs
            scores = []
            for c in cids:
                vs = bw * (sb[c] @ bc_v[i]) + pw * (sp[c] @ ph_v[i]) + dw * (sd[c] @ dn_v[i])
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                scores.append(float(np.sort(vs + mw * ms)[::-1][:k].mean()))
            sa = np.array(scores)
            ss = np.sort(sa)[::-1]
            preds.append(cids[int(np.argmax(sa))])
            margins.append(ss[0] - ss[1])
        preds, margins = np.array(preds), np.array(margins)
        for c in cids:
            cm = (preds == c) & (margins > conf_thr)
            ci = np.where(cm)[0]
            if len(ci) == 0:
                continue
            ti = ci[np.argsort(margins[ci])[::-1][:5]]
            sb[c] = np.concatenate([s_bc[c], bc_v[ti] * 0.5])
            sp[c] = np.concatenate([s_ph[c], ph_v[ti] * 0.5])
            sd[c] = np.concatenate([s_dn[c], dn_v[ti] * 0.5])
            smm[c] = np.concatenate([s_morph[c], mv[ti]])

    sm2 = np.concatenate([smm[c] for c in cids])
    gm2, gs2 = sm2.mean(0), sm2.std(0) + 1e-8
    snm = {c: (smm[c] - gm2) / gs2 for c in cids}
    snmw = {c: snm[c] * fw for c in cids}

    gt_all, pred_all, margin_all = [], [], []
    for i in range(len(lv)):
        qm = (mv[i] - gm2) / gs2
        qmw = qm * fw
        scores = {}
        for c in cids:
            vs = bw * (sb[c] @ bc_v[i]) + pw * (sp[c] @ ph_v[i]) + dw * (sd[c] @ dn_v[i])
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            scores[c] = float(np.sort(vs + mw * ms)[::-1][:k].mean())
        sa = np.array([scores[c] for c in cids])
        t1 = cids[int(np.argmax(sa))]
        mg = np.sort(sa)[::-1][0] - np.sort(sa)[::-1][1]

        if t1 in [3, 4] and mg < cascade_thr:
            for gc in [3, 4]:
                mdw = np.linalg.norm(qmw - snmw[gc], axis=1)
                msc = float(np.mean(1.0 / (1.0 + np.sort(mdw)[:5])))
                vbs = float(np.sort(sb[gc] @ bc_v[i])[::-1][:3].mean())
                vps = float(np.sort(sp[gc] @ ph_v[i])[::-1][:3].mean())
                scores[gc] = 0.25 * vbs + 0.20 * vps + cascade_mw * msc
            t1 = 3 if scores[3] > scores[4] else 4

        gt_all.append(int(lv[i]))
        pred_all.append(t1)
        margin_all.append(mg)

    return np.array(gt_all), np.array(pred_all), np.array(margin_all)


def run_baseline_prototype(seed=42):
    """Simple prototype baseline (BiomedCLIP only, no transduction)."""
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    cids = sorted(CLASS_NAMES.keys())
    support_idx = select_support(lt, seed, cids)

    prototypes = {}
    for c in cids:
        prototypes[c] = bc_t[support_idx[c]].mean(axis=0)
        prototypes[c] /= np.linalg.norm(prototypes[c])

    gt, pred = [], []
    for i in range(len(lv)):
        q = bc_v[i]
        scores = {c: float(prototypes[c] @ q) for c in cids}
        gt.append(int(lv[i]))
        pred.append(max(scores, key=scores.get))
    return np.array(gt), np.array(pred)


def plot_confusion_matrix(gt, pred, cids, title, save_path):
    """Plot a normalized confusion matrix."""
    n = len(cids)
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gt, pred):
        gi, pi = cids.index(g), cids.index(p)
        cm[gi][pi] += 1

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    names = [CLASS_NAMES[c] for c in cids]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                    ha='center', va='center', fontsize=11, color=color, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=11, rotation=30, ha='right')
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Proportion')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_performance_comparison(save_path):
    """Bar chart comparing methods across multiple metrics."""
    methods = [
        'BiomedCLIP\nPrototype\n(baseline)',
        'BiomedCLIP\n+Morph\nkNN',
        'Dual Backbone\n+Cascade',
        'Triple Backbone\n+Trans+Cascade\n(ours)',
    ]
    acc = [0.8245, 0.8356, 0.8567, 0.8713]
    mf1 = [0.6267, 0.7114, 0.7376, 0.7550]
    eos = [0.1803, 0.3072, 0.3917, 0.4658]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, acc, width, label='Accuracy', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, mf1, width, label='Macro-F1', color='#2ecc71', alpha=0.85)
    bars3 = ax.bar(x + width, eos, width, label='Eos F1', color='#e74c3c', alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('10-Shot Classification Performance Comparison\n(4 classes, 5-seed average)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_f1(save_path):
    """Per-class F1 comparison between baseline and best method."""
    classes = ['Eosinophil', 'Neutrophil', 'Lymphocyte', 'Macrophage']
    baseline_f1 = [0.1803, 0.7139, 0.8761, 0.7364]
    best_f1 = [0.4658, 0.7521, 0.8965, 0.8058]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline_f1, width, label='Baseline (Prototype)',
                   color='#95a5a6', alpha=0.8, edgecolor='#7f8c8d', linewidth=1.5)
    bars2 = ax.bar(x + width / 2, best_f1, width, label='Ours (Triple+Trans+Cascade)',
                   color='#e74c3c', alpha=0.8, edgecolor='#c0392b', linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    improvements = [best_f1[i] - baseline_f1[i] for i in range(len(classes))]
    for i, imp in enumerate(improvements):
        ax.annotate(f'+{imp:.3f}', xy=(x[i] + width / 2, best_f1[i] + 0.04),
                    fontsize=9, color='#27ae60', fontweight='bold', ha='center')

    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class F1 Score Comparison (10-Shot)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne(save_path):
    """t-SNE visualization of feature space."""
    from sklearn.manifold import TSNE

    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_v, _, _ = load_cache("phikon_v2", "val")

    bc_norm = bc_v / (np.linalg.norm(bc_v, axis=1, keepdims=True) + 1e-8)
    ph_norm = ph_v / (np.linalg.norm(ph_v, axis=1, keepdims=True) + 1e-8)
    combined = np.concatenate([bc_norm * 0.6, ph_norm * 0.4], axis=1)

    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb = tsne.fit_transform(combined)

    cids = sorted(CLASS_NAMES.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (feats, title) in enumerate([
        (bc_v, 'BiomedCLIP Features'),
        (combined, 'BiomedCLIP + Phikon-v2 Fused Features')
    ]):
        if ax_idx == 0:
            tsne2 = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
            emb2 = tsne2.fit_transform(feats)
        else:
            emb2 = emb

        ax = axes[ax_idx]
        for c in cids:
            mask = lv == c
            ax.scatter(emb2[mask, 0], emb2[mask, 1], c=CLASS_COLORS[c],
                       label=CLASS_NAMES[c], alpha=0.5, s=15, edgecolors='none')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.suptitle('t-SNE Feature Space Visualization (Validation Set)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cell_gallery(gt, pred, save_path):
    """Show example cell crops with GT and predicted labels."""
    bc_v, mv, lv = load_cache("biomedclip", "val")
    cids = sorted(CLASS_NAMES.keys())

    val_img_dir = DATA_ROOT / "images" / "val"
    val_lbl_dir = DATA_ROOT / "labels_polygon" / "val"

    cell_index = []
    for ip in sorted(val_img_dir.glob("*.png")):
        lp = val_lbl_dir / (ip.stem + ".txt")
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        anns = load_yolo(lp, h, w)
        for ann in anns:
            xs, ys = ann["xs"], ann["ys"]
            x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
            x2, y2 = min(w, int(max(xs)) + 1), min(h, int(max(ys)) + 1)
            bw_, bh_ = x2 - x1, y2 - y1
            mx, my = int(bw_ * 0.15), int(bh_ * 0.15)
            crop = img[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
            if crop.size > 0:
                cell_index.append({
                    "crop": crop, "class_id": ann["class_id"],
                    "img_name": ip.stem
                })

    correct_examples = defaultdict(list)
    wrong_examples = defaultdict(list)

    for i, cell in enumerate(cell_index):
        if i >= len(gt):
            break
        g, p = gt[i], pred[i]
        entry = {"crop": cell["crop"], "gt": g, "pred": p}
        if g == p:
            correct_examples[g].append(entry)
        else:
            wrong_examples[g].append(entry)

    fig, axes = plt.subplots(4, 8, figsize=(20, 12))
    fig.suptitle('10-Shot Classification Results Gallery\n'
                 'Green border = Correct, Red border = Wrong',
                 fontsize=14, fontweight='bold')

    for row, c in enumerate(cids):
        correct = correct_examples.get(c, [])[:4]
        wrong = wrong_examples.get(c, [])[:4]

        for col in range(4):
            ax = axes[row][col]
            if col < len(correct):
                crop = cv2.resize(correct[col]["crop"], (80, 80))
                ax.imshow(crop)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2ecc71')
                    spine.set_linewidth(4)
                if col == 0:
                    ax.set_ylabel(CLASS_NAMES[c], fontsize=11, fontweight='bold',
                                  color=CLASS_COLORS[c])
            else:
                ax.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        for col in range(4):
            ax = axes[row][4 + col]
            if col < len(wrong):
                crop = cv2.resize(wrong[col]["crop"], (80, 80))
                ax.imshow(crop)
                pred_name = CLASS_NAMES.get(wrong[col]["pred"], "?")
                ax.set_title(f'→{pred_name}', fontsize=8, color='#e74c3c')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#e74c3c')
                    spine.set_linewidth(4)
            else:
                ax.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

    col_titles = ['Correct ×4', '', '', '', 'Misclassified ×4', '', '', '']
    for col, t in enumerate(col_titles):
        if t:
            axes[0][col].set_title(t, fontsize=11, fontweight='bold',
                                   pad=10, color='#2ecc71' if col == 0 else '#e74c3c')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_segmentation_overlay(save_path):
    """Show CellposeSAM segmentation results overlaid on images."""
    img_name = "2022-06-10-14-34-55-71733_2048-1536"
    img_path = DATA_ROOT / "images" / "val" / f"{img_name}.png"
    lbl_path = DATA_ROOT / "labels_polygon" / "val" / f"{img_name}.txt"

    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    anns = load_yolo(lbl_path, h, w)

    gt_overlay = img.copy()
    for ann in anns:
        xs, ys = np.array(ann["xs"]), np.array(ann["ys"])
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        color = CLASS_COLORS_BGR[ann["class_id"]]
        cv2.polylines(gt_overlay, [pts], True, color, 2)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(gt_overlay, CLASS_NAMES[ann["class_id"]][:3],
                    (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Ground Truth Annotations ({len(anns)} cells)', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in sorted(CLASS_NAMES.keys())]
    axes[1].legend(handles=patches, loc='upper right', fontsize=10)

    plt.suptitle(f'BALF Cell Image: {img_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cellpose_comparison(save_path):
    """Compare CellposeSAM default vs optimized parameters."""
    metrics_default = {'F1': 0.5716, 'Precision': 0.4602, 'Recall': 0.7541}
    metrics_optimized = {'F1': 0.7575, 'Precision': 0.6795, 'Recall': 0.8556}

    labels = list(metrics_default.keys())
    default_vals = list(metrics_default.values())
    optimized_vals = list(metrics_optimized.values())

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, default_vals, width, label='Default (cellprob=0, auto-diameter)',
                   color='#95a5a6', alpha=0.85, edgecolor='#7f8c8d', linewidth=1.5)
    bars2 = ax.bar(x + width / 2, optimized_vals, width, label='Optimized (cellprob=-3.0, d=50)',
                   color='#e67e22', alpha=0.85, edgecolor='#d35400', linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h_ = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h_ + 0.01,
                    f'{h_:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    improvements = [optimized_vals[i] - default_vals[i] for i in range(len(labels))]
    for i, imp in enumerate(improvements):
        ax.annotate(f'+{imp:.3f}\n(+{imp / default_vals[i] * 100:.1f}%)',
                    xy=(x[i] + width / 2, optimized_vals[i] + 0.04),
                    fontsize=9, color='#27ae60', fontweight='bold', ha='center')

    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('CellposeSAM Segmentation: Default vs Optimized Parameters',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_method_ablation(save_path):
    """Ablation study: show contribution of each component."""
    components = [
        'BiomedCLIP\nPrototype',
        '+Morphology\nkNN',
        '+DINOv2\nDual-Backbone',
        '+Cascade\n(Eos/Neu)',
        '+Transductive\nInference',
        '+Phikon-v2\nTriple-Backbone',
    ]
    mf1_values = [0.6267, 0.7114, 0.7313, 0.7356, 0.7376, 0.7550]
    eos_values = [0.1803, 0.3072, 0.3530, 0.3860, 0.3917, 0.4658]

    fig, ax1 = plt.subplots(figsize=(13, 6))

    x = np.arange(len(components))
    color1 = '#3498db'
    color2 = '#e74c3c'

    ax1.plot(x, mf1_values, 'o-', color=color1, linewidth=2.5, markersize=10,
             label='Macro-F1', zorder=5)
    ax1.fill_between(x, mf1_values[0], mf1_values, alpha=0.1, color=color1)
    ax1.set_ylabel('Macro-F1', fontsize=13, color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.55, 0.80)

    ax2 = ax1.twinx()
    ax2.plot(x, eos_values, 's-', color=color2, linewidth=2.5, markersize=10,
             label='Eos F1', zorder=5)
    ax2.fill_between(x, eos_values[0], eos_values, alpha=0.1, color=color2)
    ax2.set_ylabel('Eosinophil F1', fontsize=13, color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.10, 0.55)

    for i in range(len(components)):
        ax1.annotate(f'{mf1_values[i]:.4f}', (x[i], mf1_values[i]),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=9, color=color1, fontweight='bold')
        ax2.annotate(f'{eos_values[i]:.4f}', (x[i], eos_values[i]),
                     textcoords="offset points", xytext=(0, -18),
                     ha='center', fontsize=9, color=color2, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(components, fontsize=9)
    ax1.set_title('Ablation Study: Incremental Component Contributions (10-Shot)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_support_samples(save_path):
    """Show the 40 support samples (10 per class) used for classification."""
    bc_t, mt, lt = load_cache("biomedclip", "train")
    cids = sorted(CLASS_NAMES.keys())
    support_idx = select_support(lt, SEED, cids)

    val_img_dir = DATA_ROOT / "images" / "train"
    val_lbl_dir = DATA_ROOT / "labels_polygon" / "train"

    all_cells = []
    for ip in sorted(val_img_dir.glob("*.png")):
        lp = val_lbl_dir / (ip.stem + ".txt")
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        anns = load_yolo(lp, h, w)
        for ann in anns:
            xs, ys = ann["xs"], ann["ys"]
            x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
            x2, y2 = min(w, int(max(xs)) + 1), min(h, int(max(ys)) + 1)
            bw_, bh_ = x2 - x1, y2 - y1
            mx, my = int(bw_ * 0.15), int(bh_ * 0.15)
            crop = img[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
            if crop.size > 0:
                all_cells.append({"crop": crop, "class_id": ann["class_id"]})

    fig, axes = plt.subplots(4, 10, figsize=(18, 8))
    fig.suptitle('10-Shot Support Set: 10 Labeled Samples Per Class\n'
                 '(These are ALL the labeled data used for classification)',
                 fontsize=14, fontweight='bold')

    for row, c in enumerate(cids):
        for col, idx in enumerate(support_idx[c]):
            ax = axes[row][col]
            if idx < len(all_cells):
                crop = cv2.resize(all_cells[idx]["crop"], (80, 80))
                ax.imshow(crop)
                for spine in ax.spines.values():
                    spine.set_edgecolor(CLASS_COLORS[c])
                    spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[c], fontsize=11, fontweight='bold',
                              color=CLASS_COLORS[c])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 80)

    print("\n[1/7] Running 10-shot classification...")
    gt, pred, margins = run_10shot_with_predictions(seed=SEED)
    acc = np.mean(gt == pred)
    print(f"  Accuracy: {acc:.4f}")

    gt_base, pred_base = run_baseline_prototype(seed=SEED)
    acc_base = np.mean(gt_base == pred_base)
    print(f"  Baseline accuracy: {acc_base:.4f}")

    cids = sorted(CLASS_NAMES.keys())

    print("\n[2/7] Confusion matrices...")
    plot_confusion_matrix(gt_base, pred_base, cids,
                          'Baseline: BiomedCLIP Prototype (10-Shot)',
                          OUT_DIR / "confusion_baseline.png")
    plot_confusion_matrix(gt, pred, cids,
                          'Ours: Triple Backbone + Trans + Cascade (10-Shot)',
                          OUT_DIR / "confusion_ours.png")

    print("\n[3/7] Performance comparison...")
    plot_performance_comparison(OUT_DIR / "performance_comparison.png")

    print("\n[4/7] Per-class F1...")
    plot_per_class_f1(OUT_DIR / "per_class_f1.png")

    print("\n[5/7] Ablation study...")
    plot_method_ablation(OUT_DIR / "ablation_study.png")

    print("\n[6/7] CellposeSAM comparison...")
    plot_cellpose_comparison(OUT_DIR / "cellpose_comparison.png")
    plot_segmentation_overlay(OUT_DIR / "segmentation_overlay.png")

    print("\n[7/7] t-SNE feature space...")
    plot_tsne(OUT_DIR / "tsne_features.png")

    print(f"\n{'='*80}")
    print(f"ALL VISUALIZATIONS SAVED TO: {OUT_DIR}")
    print(f"{'='*80}")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
