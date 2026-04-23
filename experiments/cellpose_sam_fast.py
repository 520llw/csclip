#!/usr/bin/env python3
"""
Cellpose-SAM fast evaluation — reduced configs, progress tracking.
Focus on key comparisons only.
"""
import sys, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.stdout.reconfigure(line_buffering=True)

DATA2_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
MC_ROOT = Path("/home/xut/csclip/cell_datasets/MultiCenter_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}


def load_gt(lbl_path, h, w):
    anns = []
    if not lbl_path.exists():
        return anns
    for line in open(lbl_path):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        if cid not in CLASS_NAMES:
            continue
        pts = [float(x) for x in parts[1:]]
        xs = [pts[i] * w for i in range(0, len(pts), 2)]
        ys = [pts[i] * h for i in range(1, len(pts), 2)]
        rr, cc = sk_polygon(ys, xs, shape=(h, w))
        if len(rr) == 0:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = True
        anns.append(mask)
    return anns


def match(gt_masks, pred_masks, iou_thr=0.5):
    tp, matched = 0, set()
    n_pred = pred_masks.max() if pred_masks.max() > 0 else 0
    for pid in range(1, n_pred + 1):
        pm = pred_masks == pid
        if pm.sum() < 30:
            continue
        best, bi = 0, -1
        for gi, gm in enumerate(gt_masks):
            if gi in matched:
                continue
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou = inter / union if union > 0 else 0
            if iou > best:
                best, bi = iou, gi
        if best >= iou_thr and bi >= 0:
            tp += 1
            matched.add(bi)
    return tp, n_pred - tp, len(gt_masks) - tp


def pamsr(model, img, primary_d=50, secondary_ds=[40, 65],
          cellprob=-3.0, rescue_prob_thr=1.0, min_area=80, overlap_thr=0.2):
    masks_p, flows_p, _ = model.eval(img, diameter=primary_d,
                                     cellprob_threshold=cellprob)
    final = masks_p.copy()
    nid = final.max() + 1

    secondary_cells = []
    for sd in secondary_ds:
        masks_s, flows_s, _ = model.eval(img, diameter=sd,
                                         cellprob_threshold=cellprob)
        prob_map_s = flows_s[2]
        for cid in range(1, masks_s.max() + 1):
            cm = masks_s == cid
            area = cm.sum()
            if area < min_area:
                continue
            if np.sum(final[cm] > 0) > overlap_thr * area:
                continue
            mp = float(prob_map_s[cm].mean())
            secondary_cells.append({"mask": cm, "mp": mp, "d": sd, "area": area})
        del masks_s, flows_s

    for i, c1 in enumerate(secondary_cells):
        c1["has_consensus"] = False
        for j, c2 in enumerate(secondary_cells):
            if i == j or c1["d"] == c2["d"]:
                continue
            inter = np.logical_and(c1["mask"], c2["mask"]).sum()
            union = np.logical_or(c1["mask"], c2["mask"]).sum()
            if union > 0 and inter / union > 0.3:
                c1["has_consensus"] = True
                break

    rescued = 0
    for c in sorted(secondary_cells, key=lambda x: -x["mp"]):
        if c["mp"] <= rescue_prob_thr or not c["has_consensus"]:
            continue
        if np.sum(final[c["mask"]] > 0) > overlap_thr * c["area"]:
            continue
        final[c["mask"] & (final == 0)] = nid
        nid += 1
        rescued += 1
    return final, rescued


def qc_filter(masks, prob_map, min_area=80, max_area=50000,
              min_circularity=0.3, min_prob=0.0):
    """Post-segmentation quality control: remove low-quality masks."""
    import cv2
    filtered = np.zeros_like(masks)
    nid = 1
    n_removed = 0

    for cid in range(1, masks.max() + 1):
        cm = masks == cid
        area = cm.sum()
        if area < min_area or area > max_area:
            n_removed += 1
            continue

        ys, xs = np.where(cm)
        contour = np.column_stack([xs, ys]).astype(np.int32)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) if len(hull) >= 3 else area
        convexity = area / (hull_area + 1e-8)

        perimeter = cv2.arcLength(hull, True) if len(hull) >= 3 else 0
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8) if perimeter > 0 else 0

        mean_prob = float(prob_map[cm].mean()) if prob_map is not None else 1.0

        if circularity < min_circularity or mean_prob < min_prob:
            n_removed += 1
            continue

        filtered[cm] = nid
        nid += 1

    return filtered, n_removed


def eval_dataset(model, dataset_name, data_root, img_ext="png"):
    img_dir = data_root / "images" / "val"
    lbl_dir = data_root / "labels_polygon" / "val"

    images = sorted(list(img_dir.glob(f"*.{img_ext}")) +
                    list(img_dir.glob("*.jpg")) +
                    list(img_dir.glob("*.jpeg")))
    images = list(dict.fromkeys(images))

    if not images:
        print(f"  WARNING: No images found in {img_dir}", flush=True)
        return {}

    configs = [
        ("cpsam_auto",      "auto"),
        ("cpsam_d50_cp-3",  "single"),
        ("cpsam_d50_cp-3+qc", "single_qc"),
        ("pamsr_cpsam",     "pamsr"),
        ("pamsr_cpsam+qc",  "pamsr_qc"),
    ]

    agg = defaultdict(lambda: [0, 0, 0])
    extra = defaultdict(lambda: {"rescued": 0, "qc_removed": 0})

    for idx, ip in enumerate(images):
        lp = lbl_dir / (ip.stem + ".txt")
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        gt = load_gt(lp, h, w)
        if not gt:
            continue

        masks_auto, _, _ = model.eval(img)
        masks_d50, flows_d50, _ = model.eval(img, diameter=50, cellprob_threshold=-3.0)
        prob_map = flows_d50[2]

        # Auto
        tp, fp, fn = match(gt, masks_auto)
        agg["cpsam_auto"][0] += tp; agg["cpsam_auto"][1] += fp; agg["cpsam_auto"][2] += fn

        # d50 cp-3
        tp, fp, fn = match(gt, masks_d50)
        agg["cpsam_d50_cp-3"][0] += tp; agg["cpsam_d50_cp-3"][1] += fp; agg["cpsam_d50_cp-3"][2] += fn

        # d50 cp-3 + QC
        masks_qc, n_rem = qc_filter(masks_d50, prob_map)
        tp, fp, fn = match(gt, masks_qc)
        agg["cpsam_d50_cp-3+qc"][0] += tp; agg["cpsam_d50_cp-3+qc"][1] += fp; agg["cpsam_d50_cp-3+qc"][2] += fn
        extra["cpsam_d50_cp-3+qc"]["qc_removed"] += n_rem

        # PAMSR
        masks_pamsr, rescued = pamsr(model, img)
        tp, fp, fn = match(gt, masks_pamsr)
        agg["pamsr_cpsam"][0] += tp; agg["pamsr_cpsam"][1] += fp; agg["pamsr_cpsam"][2] += fn
        extra["pamsr_cpsam"]["rescued"] += rescued

        # PAMSR + QC
        masks_pamsr_qc, n_rem2 = qc_filter(masks_pamsr, prob_map)
        tp, fp, fn = match(gt, masks_pamsr_qc)
        agg["pamsr_cpsam+qc"][0] += tp; agg["pamsr_cpsam+qc"][1] += fp; agg["pamsr_cpsam+qc"][2] += fn
        extra["pamsr_cpsam+qc"]["qc_removed"] += n_rem2

        del masks_auto, masks_d50, masks_qc, masks_pamsr, masks_pamsr_qc
        gc.collect()

        print(f"  [{idx+1}/{len(images)}] {ip.stem}", flush=True)

    print(f"\n{'='*100}")
    print(f"CELLPOSE-SAM RESULTS — {dataset_name} ({len(images)} val images)")
    print(f"{'='*100}")
    print(f"{'Method':<25} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}  Extra")
    print("-" * 100)

    rows = []
    for name, (tp, fp, fn) in agg.items():
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        rows.append((name, tp, fp, fn, p, r, f1))

    for n, tp, fp, fn, p, r, f1 in sorted(rows, key=lambda x: -x[-1]):
        e = extra.get(n, {})
        e_str = " ".join(f"{k}={v}" for k, v in e.items()) if e else ""
        print(f"{n:<25} {tp:>5} {fp:>5} {fn:>5} {p:>7.4f} {r:>7.4f} {f1:>7.4f}  {e_str}")

    best = max(rows, key=lambda x: x[-1])
    print(f"\n*** BEST: {best[0]} -> F1={best[-1]:.4f} ***")
    return {n: f1 for n, _, _, _, _, _, f1 in rows}


def main():
    from cellpose import models
    print("="*80)
    print("Cellpose-SAM Fast Evaluation (5 core configs)")
    print("="*80)

    model = models.CellposeModel(gpu=True)
    print(f"Model: {getattr(model, 'pretrained_model', 'unknown')}", flush=True)

    print("\n--- data2_organized ---", flush=True)
    d2 = eval_dataset(model, "data2_organized", DATA2_ROOT, "png")

    print("\n--- MultiCenter_organized ---", flush=True)
    mc = eval_dataset(model, "MultiCenter_organized", MC_ROOT, "jpg")

    print("\n\n" + "="*80)
    print("COMPARISON WITH PREVIOUS BASELINES")
    print("="*80)
    print(f"{'Dataset':<15} {'Metric':<15} {'Old CellposeModel':>20} {'New CellposeSAM':>20}")
    print("-" * 75)
    if d2:
        best_d2 = max(d2.values())
        print(f"{'data2':<15} {'Best F1':<15} {'0.7276':>20} {best_d2:>20.4f}")
    if mc:
        best_mc = max(mc.values())
        print(f"{'MC':<15} {'Best F1':<15} {'0.5517':>20} {best_mc:>20.4f}")


if __name__ == "__main__":
    main()
