#!/usr/bin/env python3
"""
Visualize BEST pipeline results:
  Segmentation: GT vs Default vs Optimized vs PAMSR
  Classification: GT vs SADC+ATD (triple-backbone + diversity transduction + cascade)
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


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def run_cellpose(model, img, cellprob_threshold=0.0, diameter=None):
    masks, flows, _ = model.eval(img, diameter=diameter,
                                 cellprob_threshold=cellprob_threshold,
                                 channels=[0, 0])
    return masks, flows


def pamsr(model, img, primary_d=50, secondary_ds=[40, 65],
          cellprob=-3.0, rescue_prob_thr=1.0, min_area=80, overlap_thr=0.2):
    """Primary-Anchor Multi-Scale Rescue segmentation."""
    masks_p, flows_p, _ = model.eval(img, diameter=primary_d,
                                     cellprob_threshold=cellprob,
                                     channels=[0, 0])
    final = masks_p.copy()
    nid = final.max() + 1
    rescued = 0

    secondary_cells = []
    for sd in secondary_ds:
        masks_s, flows_s, _ = model.eval(img, diameter=sd,
                                         cellprob_threshold=cellprob,
                                         channels=[0, 0])
        prob_map_s = flows_s[2]
        for cid in range(1, masks_s.max() + 1):
            cm = masks_s == cid
            area = cm.sum()
            if area < min_area:
                continue
            # check overlap with primary
            overlap_p = np.logical_and(cm, masks_p > 0).sum() / area
            if overlap_p > overlap_thr:
                continue
            # mean probability
            mean_prob = prob_map_s[cm].mean()
            ys, xs = np.where(cm)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            secondary_cells.append({
                "mask": cm, "sd": sd, "prob": mean_prob,
                "bbox": bbox, "area": area
            })

    # consensus + rescue
    kept = []
    for c1 in secondary_cells:
        if c1["prob"] < rescue_prob_thr:
            continue
        cons = False
        for c2 in secondary_cells:
            if c1["sd"] == c2["sd"]:
                continue
            inter = np.logical_and(c1["mask"], c2["mask"]).sum()
            union = np.logical_or(c1["mask"], c2["mask"]).sum()
            iou = inter / union if union > 0 else 0
            if iou > 0.3:
                cons = True
                break
        if cons:
            kept.append(c1)

    # resolve overlaps among kept
    kept.sort(key=lambda x: x["prob"], reverse=True)
    for c1 in kept:
        conflict = False
        for cid in range(1, final.max() + 1):
            fm = final == cid
            inter = np.logical_and(c1["mask"], fm).sum()
            if inter > 0:
                conflict = True
                break
        if not conflict:
            final[c1["mask"]] = nid
            nid += 1
            rescued += 1

    return final, rescued


def masks_to_contours(masks):
    contours_list = []
    for cell_id in range(1, masks.max() + 1):
        cell_mask = (masks == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours_list.append(contours[0])
    return contours_list


def match_masks(pred_masks_full, gt_masks, iou_thr=0.5):
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


def draw_polygon(img, xs, ys, color, thickness=2):
    pts = np.array(list(zip(xs, ys)), dtype=np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


# ===================== CLASSIFICATION =====================

def load_cache(m, s):
    d = np.load(CACHE_DIR / f"{m}_{s}.npz")
    return d["feats"], d["morphs"], d["labels"]


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels):
        pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def run_best_classification():
    """SADC+ATD with diversity-aware transduction (matching paper best config)."""
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

    # Fisher weights for cascade
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
    sb_orig = {c: s_bc[c].copy() for c in cids}

    # Diversity-aware ATD (2 iterations)
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
            proto_c = sb_orig[c].mean(0)
            dists = np.array([np.linalg.norm(bc_v[idx] - proto_c) for idx in ci])
            div_scores = margins_arr[ci] * (1.0 + 0.3 * dists / (dists.mean() + 1e-8))
            ti = ci[np.argsort(div_scores)[::-1][:5]]
            sb[c] = np.concatenate([sb_orig[c], bc_v[ti] * 0.5])
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

        # cascade for Eos/Neu
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


def visualize_full_pipeline(img_name, results, cell_index, model):
    """4-panel: GT seg, Default seg, Optimized seg, PAMSR seg + classification overlay."""
    img_path = DATA_ROOT / "images" / "val" / f"{img_name}.png"
    lbl_path = DATA_ROOT / "labels_polygon" / "val" / f"{img_name}.txt"

    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    anns = load_yolo_polygon(lbl_path, h, w)

    # GT masks
    gt_masks = []
    for ann in anns:
        rr, cc = sk_polygon(ann["ys"], ann["xs"], shape=(h, w))
        m = np.zeros((h, w), dtype=bool)
        if len(rr) > 0:
            m[rr, cc] = True
        gt_masks.append(m)

    # run segmentations
    print(f"  [Seg] default...")
    masks_default, _ = run_cellpose(model, img, cellprob_threshold=0.0, diameter=None)
    print(f"  [Seg] optimized d=50 cp=-3...")
    masks_opt, _ = run_cellpose(model, img, cellprob_threshold=-3.0, diameter=50.0)
    print(f"  [Seg] PAMSR...")
    masks_pamsr, rescued = pamsr(model, img)

    tp_d, fp_d, fn_d = match_masks(masks_default, gt_masks)
    tp_o, fp_o, fn_o = match_masks(masks_opt, gt_masks)
    tp_p, fp_p, fn_p = match_masks(masks_pamsr, gt_masks)

    def metrics(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return prec, rec, f1

    pd, rd, fd = metrics(tp_d, fp_d, fn_d)
    po, ro, fo = metrics(tp_o, fp_o, fn_o)
    pp, rp, fp_ = metrics(tp_p, fp_p, fn_p)

    # classification overlay for this image
    indices = [i for i, c in enumerate(cell_index) if c["img_stem"] == img_name]
    n_correct, n_total = 0, 0
    pred_overlay = img.copy()
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
        pred_color = COLORS[pred_c]
        is_correct = gt_c == pred_c
        draw_polygon(pred_overlay, ann["xs"], ann["ys"], pred_color, 2 if is_correct else 3)
        cx, cy = int(np.mean(ann["xs"])), int(np.mean(ann["ys"]))
        label = CLASS_NAMES[pred_c]
        if not is_correct:
            label += "!"
            cv2.line(pred_overlay, (int(min(ann["xs"])), int(min(ann["ys"]))),
                     (int(max(ann["xs"])), int(max(ann["ys"]))), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(pred_overlay, (int(max(ann["xs"])), int(min(ann["ys"]))),
                     (int(min(ann["xs"])), int(max(ann["ys"]))), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(pred_overlay, label, (cx - 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 2, cv2.LINE_AA)

    cls_acc = n_correct / n_total if n_total else 0

    fig, axes = plt.subplots(2, 3, figsize=(30, 18))

    # GT
    gt_img = img.copy()
    for ann in anns:
        color = COLORS[ann["class_id"]]
        draw_polygon(gt_img, ann["xs"], ann["ys"], color, 2)
        cx, cy = int(np.mean(ann["xs"])), int(np.mean(ann["ys"]))
        cv2.putText(gt_img, CLASS_NAMES[ann["class_id"]], (cx - 10, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    axes[0, 0].imshow(gt_img)
    axes[0, 0].set_title(f'Ground Truth\n{len(anns)} cells', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    patches = [mpatches.Patch(color=COLORS_HEX[c], label=CLASS_FULL[c]) for c in sorted(CLASS_NAMES.keys())]
    axes[0, 0].legend(handles=patches, loc='upper right', fontsize=9)

    # Default
    def_img = img.copy()
    for cnt in masks_to_contours(masks_default):
        cv2.drawContours(def_img, [cnt], -1, (0, 200, 255), 2, cv2.LINE_AA)
    axes[0, 1].imshow(def_img)
    axes[0, 1].set_title(f'CellposeSAM Default\ncellprob=0, auto-diameter\n'
                         f'Detected: {masks_default.max()} | TP={tp_d} FP={fp_d} FN={fn_d}\n'
                         f'P={pd:.3f} R={rd:.3f} F1={fd:.3f}',
                         fontsize=11, fontweight='bold', color='#e67e22')
    axes[0, 1].axis('off')

    # Optimized
    opt_img = img.copy()
    for cnt in masks_to_contours(masks_opt):
        cv2.drawContours(opt_img, [cnt], -1, (50, 255, 50), 2, cv2.LINE_AA)
    axes[0, 2].imshow(opt_img)
    axes[0, 2].set_title(f'CellposeSAM Optimized\ncellprob=-3.0, diameter=50\n'
                         f'Detected: {masks_opt.max()} | TP={tp_o} FP={fp_o} FN={fn_o}\n'
                         f'P={po:.3f} R={ro:.3f} F1={fo:.3f}',
                         fontsize=11, fontweight='bold', color='#27ae60')
    axes[0, 2].axis('off')

    # PAMSR
    pamsr_img = img.copy()
    for cnt in masks_to_contours(masks_pamsr):
        cv2.drawContours(pamsr_img, [cnt], -1, (155, 89, 182), 2, cv2.LINE_AA)  # purple
    axes[1, 0].imshow(pamsr_img)
    axes[1, 0].set_title(f'PAMSR (Ours)\nprimary_d=50, aux=[40,65], cp=-3.0\n'
                         f'Detected: {masks_pamsr.max()} | TP={tp_p} FP={fp_p} FN={fn_p} (rescued={rescued})\n'
                         f'P={pp:.3f} R={rp:.3f} F1={fp_:.3f}',
                         fontsize=11, fontweight='bold', color='#8e44ad')
    axes[1, 0].axis('off')

    # Classification overlay on GT masks
    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title(f'10-Shot Classification (SADC+ATD)\n'
                         f'Correct: {n_correct}/{n_total} ({cls_acc:.1%}) | "!" = misclassified',
                         fontsize=12, fontweight='bold',
                         color='#27ae60' if cls_acc > 0.8 else '#e74c3c')
    axes[1, 1].axis('off')
    axes[1, 1].legend(handles=patches, loc='upper right', fontsize=9)

    # Side-by-side zoom comparison of a small region (center crop)
    cy, cx = h // 2, w // 2
    hh, ww = h // 4, w // 4
    zoom_gt = gt_img[cy - hh:cy + hh, cx - ww:cx + ww]
    zoom_pred = pred_overlay[cy - hh:cy + hh, cx - ww:cx + ww]
    zoom_combined = np.concatenate([zoom_gt, zoom_pred], axis=1)
    axes[1, 2].imshow(zoom_combined)
    axes[1, 2].set_title('Zoom: GT (left) vs Predicted (right)\nCenter crop', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle(f'BEST Pipeline Results on Real BALF Image: {img_name[-25:]}',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = OUT_DIR / f"best_pipeline_{img_name[-25:]}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    del masks_default, masks_opt, masks_pamsr
    gc.collect()
    print(f"  Saved: {out_path}")
    return cls_acc


def main():
    print("=" * 80)
    print("VISUALIZE BEST PIPELINE: PAMSR + SADC+ATD")
    print("=" * 80)

    from cellpose import models
    print("Loading Cellpose model...")
    cpmodel = models.CellposeModel(gpu=True)

    target_images = [
        "2022-06-10-14-34-55-71733_2048-1536",
        "2022-06-10-14-38-15-82373_2048-1536",
        "2022-06-10-14-09-32-87353_2048-1536",
    ]

    print("\nRunning best classification pipeline...")
    results = run_best_classification()
    gt_all = [r["gt"] for r in results]
    pred_all = [r["pred"] for r in results]
    acc = sum(g == p for g, p in zip(gt_all, pred_all)) / len(gt_all)
    print(f"  Overall val accuracy: {acc:.4f}")

    cell_index = build_val_cell_index()
    print(f"  Cell index: {len(cell_index)} cells")

    for img_name in target_images:
        print(f"\nProcessing {img_name}...")
        visualize_full_pipeline(img_name, results, cell_index, cpmodel)

    print(f"\nAll done. Check {OUT_DIR}")


if __name__ == "__main__":
    main()
