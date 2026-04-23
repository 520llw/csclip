#!/usr/bin/env python3
"""
Real-image visualization: CellposeSAM segmentation + 10-shot classification
on actual BALF cell images.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys, random, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
OUT_DIR = Path("/home/xut/csclip/experiments/visualizations")
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {3: "Eos", 4: "Neu", 5: "Lym", 6: "Mac"}
CLASS_FULL = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
COLORS = {
    3: (231, 76, 60),   # red - Eos
    4: (52, 152, 219),  # blue - Neu
    5: (46, 204, 113),  # green - Lym
    6: (243, 156, 18),  # orange - Mac
}
COLORS_HEX = {3: '#e74c3c', 4: '#3498db', 5: '#2ecc71', 6: '#f39c12'}
N_SHOT = 10
SEED = 42


def load_yolo_polygon(lp, h, w):
    anns = []
    if not lp.exists():
        return anns
    for line in open(lp):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        if cid not in CLASS_NAMES:
            continue
        pts = [float(x) for x in parts[1:]]
        xs = [pts[i] * w for i in range(0, len(pts), 2)]
        ys = [pts[i] * h for i in range(1, len(pts), 2)]
        anns.append({"class_id": cid, "xs": xs, "ys": ys})
    return anns


def ann_to_bbox(ann, margin=0.15):
    xs, ys = ann["xs"], ann["ys"]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)) + 1, int(max(ys)) + 1
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    return x1, y1, x2, y2, mx, my


def draw_cell_box(img, x1, y1, x2, y2, color, label, thickness=2, font_scale=0.5):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_polygon(img, xs, ys, color, thickness=2):
    pts = np.array(list(zip(xs, ys)), dtype=np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


# ===================== SEGMENTATION VISUALIZATION =====================

def run_cellpose_on_image(img_np, cellprob_threshold=0.0, diameter=None):
    from cellpose import models
    model = models.CellposeModel(gpu=True)
    masks, flows, styles = model.eval(
        img_np, diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        channels=[0, 0],
    )
    del model
    gc.collect()
    return masks


def masks_to_contours(masks):
    contours_list = []
    for cell_id in range(1, masks.max() + 1):
        cell_mask = (masks == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours_list.append(contours[0])
    return contours_list


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def visualize_segmentation(img_name, save_path):
    """Run CellposeSAM with default vs optimized params, overlay on real image."""
    img_path = DATA_ROOT / "images" / "val" / f"{img_name}.png"
    lbl_path = DATA_ROOT / "labels_polygon" / "val" / f"{img_name}.txt"

    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    anns = load_yolo_polygon(lbl_path, h, w)

    print(f"  Running CellposeSAM default (cellprob=0, auto-diameter)...")
    masks_default = run_cellpose_on_image(img, cellprob_threshold=0.0, diameter=None)
    contours_default = masks_to_contours(masks_default)

    print(f"  Running CellposeSAM optimized (cellprob=-3.0, d=50)...")
    masks_optimized = run_cellpose_on_image(img, cellprob_threshold=-3.0, diameter=50.0)
    contours_optimized = masks_to_contours(masks_optimized)

    gt_masks = []
    for ann in anns:
        rr, cc = sk_polygon(ann["ys"], ann["xs"], shape=(h, w))
        m = np.zeros((h, w), dtype=bool)
        if len(rr) > 0:
            m[rr, cc] = True
        gt_masks.append(m)

    def match_and_count(pred_masks_full, gt_masks, iou_thr=0.5):
        tp, matched_gt = 0, set()
        for pid in range(1, pred_masks_full.max() + 1):
            pm = pred_masks_full == pid
            best_iou, best_gi = 0, -1
            for gi, gm in enumerate(gt_masks):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pm, gm)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thr and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)
        fp = pred_masks_full.max() - tp
        fn = len(gt_masks) - tp
        return tp, fp, fn

    tp_d, fp_d, fn_d = match_and_count(masks_default, gt_masks)
    tp_o, fp_o, fn_o = match_and_count(masks_optimized, gt_masks)

    prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) else 0
    rec_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) else 0
    f1_d = 2 * prec_d * rec_d / (prec_d + rec_d) if (prec_d + rec_d) else 0

    prec_o = tp_o / (tp_o + fp_o) if (tp_o + fp_o) else 0
    rec_o = tp_o / (tp_o + fn_o) if (tp_o + fn_o) else 0
    f1_o = 2 * prec_o * rec_o / (prec_o + rec_o) if (prec_o + rec_o) else 0

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # GT
    gt_img = img.copy()
    for ann in anns:
        color = COLORS[ann["class_id"]]
        draw_polygon(gt_img, ann["xs"], ann["ys"], color, 2)
        cx, cy = int(np.mean(ann["xs"])), int(np.mean(ann["ys"]))
        cv2.putText(gt_img, CLASS_NAMES[ann["class_id"]], (cx - 10, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    axes[0].imshow(gt_img)
    axes[0].set_title(f'Ground Truth\n{len(anns)} cells', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    patches = [mpatches.Patch(color=COLORS_HEX[c], label=CLASS_FULL[c]) for c in sorted(CLASS_NAMES.keys())]
    axes[0].legend(handles=patches, loc='upper right', fontsize=9)

    # Default
    def_img = img.copy()
    for cnt in contours_default:
        cv2.drawContours(def_img, [cnt], -1, (0, 200, 255), 2, cv2.LINE_AA)
    axes[1].imshow(def_img)
    axes[1].set_title(f'CellposeSAM Default\ncellprob=0, auto-diameter\n'
                      f'Detected: {len(contours_default)} | '
                      f'TP={tp_d} FP={fp_d} FN={fn_d}\n'
                      f'P={prec_d:.3f} R={rec_d:.3f} F1={f1_d:.3f}',
                      fontsize=12, fontweight='bold', color='#e67e22')
    axes[1].axis('off')

    # Optimized
    opt_img = img.copy()
    for cnt in contours_optimized:
        cv2.drawContours(opt_img, [cnt], -1, (50, 255, 50), 2, cv2.LINE_AA)
    axes[2].imshow(opt_img)
    axes[2].set_title(f'CellposeSAM Optimized\ncellprob=-3.0, diameter=50\n'
                      f'Detected: {len(contours_optimized)} | '
                      f'TP={tp_o} FP={fp_o} FN={fn_o}\n'
                      f'P={prec_o:.3f} R={rec_o:.3f} F1={f1_o:.3f}',
                      fontsize=12, fontweight='bold', color='#27ae60')
    axes[2].axis('off')

    plt.suptitle(f'CellposeSAM Segmentation Comparison on Real BALF Image',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    del masks_default, masks_optimized
    gc.collect()
    print(f"  Saved: {save_path}")
    return f1_d, f1_o


# ===================== CLASSIFICATION VISUALIZATION =====================

def load_cache(m, s):
    d = np.load(CACHE_DIR / f"{m}_{s}.npz")
    return d["feats"], d["morphs"], d["labels"]


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels):
        pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def run_classification_per_cell():
    """Run triple backbone classification, return per-cell (gt, pred, confidence)."""
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")

    cids = sorted(CLASS_NAMES.keys())
    support_idx = select_support(lt, SEED, cids)
    s_bc = {c: bc_t[support_idx[c]] for c in cids}
    s_ph = {c: ph_t[support_idx[c]] for c in cids}
    s_dn = {c: dn_t[support_idx[c]] for c in cids}
    s_morph = {c: mt[support_idx[c]] for c in cids}

    sm = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm.mean(0), sm.std(0) + 1e-8

    eos_m, neu_m = s_morph[3], s_morph[4]
    n_dims = mt.shape[1]
    fw = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        mu_diff = (np.mean(eos_m[:, d]) - np.mean(neu_m[:, d])) ** 2
        var_sum = np.var(eos_m[:, d]) + np.var(neu_m[:, d]) + 1e-10
        fw[d] = 1.0 + (mu_diff / var_sum) * 2.0

    bw, pw, dw, mw = 0.42, 0.18, 0.07, 0.33
    k = 7

    sb = {c: s_bc[c].copy() for c in cids}
    sp = {c: s_ph[c].copy() for c in cids}
    sd = {c: s_dn[c].copy() for c in cids}
    smm = {c: s_morph[c].copy() for c in cids}

    for _ in range(2):
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
        preds_arr, margins_arr = np.array(preds), np.array(margins)
        for c in cids:
            cm = (preds_arr == c) & (margins_arr > 0.025)
            ci = np.where(cm)[0]
            if len(ci) == 0:
                continue
            ti = ci[np.argsort(margins_arr[ci])[::-1][:5]]
            sb[c] = np.concatenate([s_bc[c], bc_v[ti] * 0.5])
            sp[c] = np.concatenate([s_ph[c], ph_v[ti] * 0.5])
            sd[c] = np.concatenate([s_dn[c], dn_v[ti] * 0.5])
            smm[c] = np.concatenate([s_morph[c], mv[ti]])

    sm2 = np.concatenate([smm[c] for c in cids])
    gm2, gs2 = sm2.mean(0), sm2.std(0) + 1e-8
    snm = {c: (smm[c] - gm2) / gs2 for c in cids}
    snmw = {c: snm[c] * fw for c in cids}

    results = []
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

        if t1 in [3, 4] and mg < 0.012:
            for gc in [3, 4]:
                mdw = np.linalg.norm(qmw - snmw[gc], axis=1)
                msc = float(np.mean(1.0 / (1.0 + np.sort(mdw)[:5])))
                vbs = float(np.sort(sb[gc] @ bc_v[i])[::-1][:3].mean())
                vps = float(np.sort(sp[gc] @ ph_v[i])[::-1][:3].mean())
                scores[gc] = 0.25 * vbs + 0.20 * vps + 0.45 * msc
            t1 = 3 if scores[3] > scores[4] else 4

        results.append({"gt": int(lv[i]), "pred": t1, "margin": mg})
    return results


def build_val_cell_index():
    """Build index: list of (image_path, ann) for each cell in val set, matching cache order."""
    val_img_dir = DATA_ROOT / "images" / "val"
    val_lbl_dir = DATA_ROOT / "labels_polygon" / "val"
    cells = []
    for ip in sorted(val_img_dir.glob("*.png")):
        lp = val_lbl_dir / (ip.stem + ".txt")
        if not lp.exists():
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        anns = load_yolo_polygon(lp, h, w)
        for ann in anns:
            cells.append({"img_path": str(ip), "img_stem": ip.stem, "ann": ann, "h": h, "w": w})
    return cells


def visualize_classification_on_images(results, cell_index, target_images, save_prefix):
    """Overlay classification predictions on real images."""
    for img_stem in target_images:
        indices = [i for i, c in enumerate(cell_index) if c["img_stem"] == img_stem]
        if not indices:
            continue

        img_path = cell_index[indices[0]]["img_path"]
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        gt_img = img.copy()
        pred_img = img.copy()

        n_correct, n_total = 0, 0
        for idx in indices:
            if idx >= len(results):
                continue
            cell = cell_index[idx]
            res = results[idx]
            ann = cell["ann"]
            gt_c, pred_c = res["gt"], res["pred"]
            n_total += 1
            if gt_c == pred_c:
                n_correct += 1

            x1, y1, x2, y2, mx, my = ann_to_bbox(ann)

            gt_color = COLORS[gt_c]
            draw_polygon(gt_img, ann["xs"], ann["ys"], gt_color, 2)
            cx, cy = int(np.mean(ann["xs"])), int(np.mean(ann["ys"]))
            cv2.putText(gt_img, CLASS_NAMES[gt_c], (cx - 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2, cv2.LINE_AA)

            pred_color = COLORS[pred_c]
            is_correct = gt_c == pred_c
            thickness = 2 if is_correct else 3

            draw_polygon(pred_img, ann["xs"], ann["ys"], pred_color, thickness)
            label = CLASS_NAMES[pred_c]
            if not is_correct:
                label = f"{CLASS_NAMES[pred_c]}!"
                cv2.line(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(pred_img, (x2, y1), (x1, y2), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(pred_img, label, (cx - 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 2, cv2.LINE_AA)

        acc = n_correct / n_total if n_total else 0

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        axes[0].imshow(gt_img)
        axes[0].set_title(f'Ground Truth Labels ({n_total} cells)', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(pred_img)
        axes[1].set_title(f'10-Shot Predicted Labels\n'
                          f'Correct: {n_correct}/{n_total} ({acc:.1%})\n'
                          f'\"!\" marks = misclassified',
                          fontsize=13, fontweight='bold',
                          color='#27ae60' if acc > 0.8 else '#e74c3c')
        axes[1].axis('off')

        patches = [mpatches.Patch(color=COLORS_HEX[c], label=CLASS_FULL[c])
                   for c in sorted(CLASS_NAMES.keys())]
        axes[1].legend(handles=patches, loc='upper right', fontsize=10)

        plt.suptitle(f'10-Shot Classification on Real BALF Image (Seed={SEED})',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        out_path = OUT_DIR / f"{save_prefix}_{img_stem[-20:]}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path} (acc={acc:.1%})")


def visualize_crop_comparison(results, cell_index, save_path):
    """Grid of individual cell crops: GT label vs predicted label."""
    cids = sorted(CLASS_NAMES.keys())
    examples = {"correct": defaultdict(list), "wrong": defaultdict(list)}

    for i, res in enumerate(results):
        if i >= len(cell_index):
            break
        cell = cell_index[i]
        ann = cell["ann"]
        img = np.array(Image.open(cell["img_path"]).convert("RGB"))
        h, w = img.shape[:2]

        x1, y1, x2, y2, mx, my = ann_to_bbox(ann)
        crop = img[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (96, 96))

        entry = {"crop": crop, "gt": res["gt"], "pred": res["pred"]}
        if res["gt"] == res["pred"]:
            if len(examples["correct"][res["gt"]]) < 5:
                examples["correct"][res["gt"]].append(entry)
        else:
            if len(examples["wrong"][res["gt"]]) < 5:
                examples["wrong"][res["gt"]].append(entry)

    fig, axes = plt.subplots(4, 10, figsize=(22, 10))
    fig.suptitle('10-Shot Cell Classification: Correct (green) vs Misclassified (red)\n'
                 'Left: correctly classified | Right: misclassified (showing predicted label)',
                 fontsize=14, fontweight='bold')

    for row, c in enumerate(cids):
        for col in range(5):
            ax = axes[row][col]
            items = examples["correct"].get(c, [])
            if col < len(items):
                ax.imshow(items[col]["crop"])
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2ecc71')
                    spine.set_linewidth(4)
            else:
                ax.set_facecolor('#f0f0f0')
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(CLASS_FULL[c], fontsize=11, fontweight='bold',
                              color=COLORS_HEX[c], labelpad=10)

        for col in range(5):
            ax = axes[row][5 + col]
            items = examples["wrong"].get(c, [])
            if col < len(items):
                ax.imshow(items[col]["crop"])
                pred_name = CLASS_NAMES[items[col]["pred"]]
                ax.set_xlabel(f'→{pred_name}', fontsize=9, color='#e74c3c', fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#e74c3c')
                    spine.set_linewidth(4)
            else:
                ax.set_facecolor('#f0f0f0')
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0][0].set_title('Correct', fontsize=12, fontweight='bold', color='#27ae60', pad=10)
    axes[0][5].set_title('Misclassified', fontsize=12, fontweight='bold', color='#e74c3c', pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 80)
    print("REAL-IMAGE VISUALIZATION")
    print("=" * 80)

    # ---- PART 1: CellposeSAM Segmentation ----
    print("\n[1/3] CellposeSAM Segmentation on Real Images...")
    seg_images = [
        "2022-06-10-14-34-55-71733_2048-1536",
        "2022-06-10-14-38-15-82373_2048-1536",
    ]
    for img_name in seg_images:
        print(f"\n  Processing {img_name}...")
        save_path = OUT_DIR / f"seg_real_{img_name[-20:]}.png"
        visualize_segmentation(img_name, save_path)

    # ---- PART 2: Classification on Real Images ----
    print("\n[2/3] 10-Shot Classification on Real Images...")
    print("  Running classification pipeline...")
    results = run_classification_per_cell()
    gt_all = [r["gt"] for r in results]
    pred_all = [r["pred"] for r in results]
    acc = sum(g == p for g, p in zip(gt_all, pred_all)) / len(gt_all)
    print(f"  Overall accuracy: {acc:.4f} ({sum(g==p for g,p in zip(gt_all,pred_all))}/{len(gt_all)})")

    cell_index = build_val_cell_index()
    print(f"  Cell index: {len(cell_index)} cells, results: {len(results)} predictions")

    cls_images = [
        "2022-06-10-14-34-55-71733_2048-1536",
        "2022-06-10-14-38-15-82373_2048-1536",
        "2022-06-10-14-09-32-87353_2048-1536",
    ]
    visualize_classification_on_images(results, cell_index, cls_images, "cls_real")

    # ---- PART 3: Cell Crop Gallery ----
    print("\n[3/3] Cell crop gallery...")
    visualize_crop_comparison(results, cell_index, OUT_DIR / "cell_crop_gallery.png")

    print(f"\n{'=' * 80}")
    print(f"ALL REAL-IMAGE VISUALIZATIONS SAVED TO: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
