#!/usr/bin/env python3
"""
CellposeSAM actual usability test on BALF images.
Tests segmentation quality against ground truth annotations.

Metrics:
- Detection precision/recall/F1 (IoU-based matching)
- Mean IoU of matched cells
- Per-image breakdown

Configurations tested:
1. Default CellposeSAM (cellprob_threshold=0.0)
2. Optimized (cellprob_threshold=-2.0)
3. With/without adaptive preprocessing
4. With auto diameter estimation
5. Different cellprob_threshold values
"""
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from skimage import io as skio
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}


def load_gt_annotations(label_path, h, w):
    """Load ground truth YOLO polygon annotations."""
    anns = []
    if not label_path.exists():
        return anns
    for line in open(label_path):
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
        x1, y1, x2, y2 = int(np.min(cc)), int(np.min(rr)), int(np.max(cc))+1, int(np.max(rr))+1
        anns.append({
            "class_id": cid,
            "mask": mask,
            "bbox": (x1, y1, x2, y2),
            "area": int(np.sum(mask)),
        })
    return anns


def pred_points_to_mask(points, h, w):
    """Convert normalized polygon points to binary mask."""
    xs = [points[i] * w for i in range(0, len(points), 2)]
    ys = [points[i] * h for i in range(1, len(points), 2)]
    if len(xs) < 3:
        return None
    rr, cc = sk_polygon(ys, xs, shape=(h, w))
    if len(rr) == 0:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return mask


def compute_iou(mask1, mask2):
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / union if union > 0 else 0.0


def match_predictions(gt_anns, pred_masks, iou_thr=0.3):
    """Match predictions to ground truth using Hungarian-like greedy matching."""
    n_gt = len(gt_anns)
    n_pred = len(pred_masks)

    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0, 0, []
    if n_gt == 0:
        return 0, n_pred, 0, 0, []
    if n_pred == 0:
        return n_gt, 0, 0, 0, []

    iou_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            iou_matrix[i, j] = compute_iou(gt_anns[i]["mask"], pred_masks[j])

    matched_gt = set()
    matched_pred = set()
    matched_ious = []

    while True:
        if len(matched_gt) == n_gt or len(matched_pred) == n_pred:
            break
        remaining = iou_matrix.copy()
        for i in matched_gt:
            remaining[i, :] = -1
        for j in matched_pred:
            remaining[:, j] = -1
        best = np.unravel_index(np.argmax(remaining), remaining.shape)
        if remaining[best[0], best[1]] < iou_thr:
            break
        matched_gt.add(best[0])
        matched_pred.add(best[1])
        matched_ious.append(remaining[best[0], best[1]])

    tp = len(matched_gt)
    fn = n_gt - tp
    fp = n_pred - len(matched_pred)

    return tp, fp, fn, matched_ious


def run_cellpose_config(images, config_name, **kwargs):
    """Run CellposeSAM with specific configuration on a list of images."""
    from labeling_tool.cellpose_utils import (
        _get_model, assess_image_quality, adaptive_preprocess,
        estimate_cell_diameters, postprocess_segmentation,
        _label_map_to_polygons
    )

    gpu = kwargs.get("gpu", True)
    cellprob_threshold = kwargs.get("cellprob_threshold", 0.0)
    diameter = kwargs.get("diameter", 30.0)
    preprocess = kwargs.get("preprocess", False)
    auto_diam = kwargs.get("auto_diameter", False)
    min_area = kwargs.get("min_area", 100)
    flow_threshold = kwargs.get("flow_threshold", 0.4)

    model = _get_model(gpu)
    results = {}

    for img_path in images:
        img = skio.imread(str(img_path))
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
        h, w = img.shape[:2]

        proc_img = img.copy()
        if preprocess:
            quality = assess_image_quality(img)
            proc_img = adaptive_preprocess(img, quality)

        if auto_diam:
            diams = estimate_cell_diameters(img)
            use_diam = float(diams[1]) if len(diams) > 1 else float(diams[0])
        else:
            use_diam = diameter

        t0 = time.time()
        result = model.eval([proc_img], diameter=use_diam, channels=[0, 0],
                            cellprob_threshold=cellprob_threshold,
                            flow_threshold=flow_threshold)
        elapsed = time.time() - t0

        label_map = result[0][0]
        label_map = postprocess_segmentation(label_map, min_area=min_area)
        polys = _label_map_to_polygons(label_map, 0, min_area)

        pred_masks = []
        for p in polys:
            m = pred_points_to_mask(p["points"], h, w)
            if m is not None:
                pred_masks.append(m)

        results[img_path.name] = {
            "pred_masks": pred_masks,
            "n_pred": len(pred_masks),
            "time": elapsed,
        }

    return results


def evaluate_config(images_dir, labels_dir, image_list, results, iou_thr=0.3):
    """Evaluate CellposeSAM results against ground truth."""
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []
    per_image = []

    for img_path in image_list:
        img = skio.imread(str(img_path))
        h, w = img.shape[:2]

        label_path = labels_dir / (img_path.stem + ".txt")
        gt_anns = load_gt_annotations(label_path, h, w)

        pred_masks = results[img_path.name]["pred_masks"]
        tp, fp, fn, matched_ious = match_predictions(gt_anns, pred_masks, iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(matched_ious)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall > 0 else 0

        per_image.append({
            "name": img_path.name,
            "gt": len(gt_anns),
            "pred": len(pred_masks),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0,
            "time": results[img_path.name]["time"],
        })

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall > 0 else 0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": mean_iou,
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
        "n_images": len(image_list),
        "per_image": per_image,
    }


def main():
    images_dir = DATA_ROOT / "images" / "val"
    labels_dir = DATA_ROOT / "labels_polygon" / "val"
    image_list = sorted(images_dir.glob("*.png"))

    print(f"CellposeSAM Evaluation on {len(image_list)} validation images")
    print(f"{'='*80}")

    configs = [
        ("default_cp0", {"cellprob_threshold": 0.0, "diameter": 30.0}),
        ("optimized_cpm2", {"cellprob_threshold": -2.0, "diameter": 30.0}),
        ("cpm3", {"cellprob_threshold": -3.0, "diameter": 30.0}),
        ("cpm1", {"cellprob_threshold": -1.0, "diameter": 30.0}),
        ("cpm2_d40", {"cellprob_threshold": -2.0, "diameter": 40.0}),
        ("cpm2_d20", {"cellprob_threshold": -2.0, "diameter": 20.0}),
        ("cpm2_d50", {"cellprob_threshold": -2.0, "diameter": 50.0}),
        ("cpm2_preproc", {"cellprob_threshold": -2.0, "diameter": 30.0, "preprocess": True}),
        ("cpm2_auto_d", {"cellprob_threshold": -2.0, "auto_diameter": True}),
        ("cpm2_flow03", {"cellprob_threshold": -2.0, "diameter": 30.0, "flow_threshold": 0.3}),
        ("cpm2_flow05", {"cellprob_threshold": -2.0, "diameter": 30.0, "flow_threshold": 0.5}),
    ]

    all_metrics = {}
    for name, kwargs in configs:
        print(f"\nRunning: {name}...")
        t0 = time.time()
        results = run_cellpose_config(image_list, name, **kwargs)
        eval_result = evaluate_config(images_dir, labels_dir, image_list, results)
        elapsed = time.time() - t0
        all_metrics[name] = eval_result

        print(f"  Precision={eval_result['precision']:.4f}  "
              f"Recall={eval_result['recall']:.4f}  "
              f"F1={eval_result['f1']:.4f}  "
              f"mIoU={eval_result['mean_iou']:.4f}  "
              f"TP={eval_result['total_tp']} FP={eval_result['total_fp']} FN={eval_result['total_fn']}  "
              f"[{elapsed:.1f}s]")

    print(f"\n{'='*120}")
    print("COMPREHENSIVE CELLPOSESAM EVALUATION")
    print(f"{'='*120}")
    header = f"{'Config':<25} {'Prec':>7} {'Recall':>7} {'F1':>7} {'mIoU':>7}  {'TP':>5} {'FP':>5} {'FN':>5}"
    print(header)
    print("-" * 120)
    for name in sorted(all_metrics.keys(), key=lambda x: -all_metrics[x]["f1"]):
        m = all_metrics[name]
        print(f"{name:<25} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['mean_iou']:>7.4f}  "
              f"{m['total_tp']:>5} {m['total_fp']:>5} {m['total_fn']:>5}")

    best_name = max(all_metrics.keys(), key=lambda x: all_metrics[x]["f1"])
    best = all_metrics[best_name]
    print(f"\nBEST CONFIG: {best_name}")
    print(f"  F1={best['f1']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, mIoU={best['mean_iou']:.4f}")

    print(f"\n--- Per-image breakdown (best config: {best_name}) ---")
    for pi in best["per_image"]:
        flag = " ⚠️" if pi["f1"] < 0.5 else ""
        print(f"  {pi['name']:<50} GT={pi['gt']:>3} Pred={pi['pred']:>3} "
              f"P={pi['precision']:.2f} R={pi['recall']:.2f} F1={pi['f1']:.2f} "
              f"IoU={pi['mean_iou']:.2f}{flag}")

    worst = sorted(best["per_image"], key=lambda x: x["f1"])[:5]
    print(f"\n--- Worst 5 images (potential noisy/difficult cases) ---")
    for pi in worst:
        print(f"  {pi['name']}: F1={pi['f1']:.3f} (GT={pi['gt']}, Pred={pi['pred']}, "
              f"TP={pi['tp']}, FP={pi['fp']}, FN={pi['fn']})")


if __name__ == "__main__":
    main()
