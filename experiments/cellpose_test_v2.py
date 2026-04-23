#!/usr/bin/env python3
"""
CellposeSAM evaluation v2 - memory-efficient, focused configs.
Processes one image at a time, releases masks immediately.
"""
import os
import sys
import time
import gc
from pathlib import Path

import cv2
import numpy as np
from skimage import io as skio
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}


def load_gt_masks(label_path, h, w):
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
        anns.append({"class_id": cid, "mask": mask})
    return anns


def poly_to_mask(points, h, w):
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


def match_and_score(gt_anns, pred_masks, iou_thr=0.3):
    n_gt, n_pred = len(gt_anns), len(pred_masks)
    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0, []
    if n_gt == 0:
        return 0, n_pred, 0, []
    if n_pred == 0:
        return 0, 0, n_gt, []

    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i in range(n_gt):
        for j in range(n_pred):
            inter = np.sum(gt_anns[i]["mask"] & pred_masks[j])
            union = np.sum(gt_anns[i]["mask"] | pred_masks[j])
            iou_matrix[i, j] = inter / union if union > 0 else 0

    matched_gt, matched_pred = set(), set()
    matched_ious = []
    while len(matched_gt) < n_gt and len(matched_pred) < n_pred:
        rem = iou_matrix.copy()
        for i in matched_gt: rem[i, :] = -1
        for j in matched_pred: rem[:, j] = -1
        bi, bj = np.unravel_index(np.argmax(rem), rem.shape)
        if rem[bi, bj] < iou_thr:
            break
        matched_gt.add(bi)
        matched_pred.add(bj)
        matched_ious.append(rem[bi, bj])

    tp = len(matched_gt)
    return tp, n_pred - len(matched_pred), n_gt - tp, matched_ious


def run_eval(image_list, labels_dir, config_name, **kwargs):
    from labeling_tool.cellpose_utils import (
        _get_model, assess_image_quality, adaptive_preprocess,
        estimate_cell_diameters, postprocess_segmentation,
        _label_map_to_polygons
    )

    cellprob_threshold = kwargs.get("cellprob_threshold", 0.0)
    diameter = kwargs.get("diameter", 30.0)
    preprocess = kwargs.get("preprocess", False)
    auto_diam = kwargs.get("auto_diameter", False)
    min_area = kwargs.get("min_area", 100)
    flow_threshold = kwargs.get("flow_threshold", 0.4)
    iou_thr = kwargs.get("iou_thr", 0.3)

    model = _get_model(True)
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []
    per_image = []

    for idx, img_path in enumerate(image_list):
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
            m = poly_to_mask(p["points"], h, w)
            if m is not None:
                pred_masks.append(m)

        label_path = labels_dir / (img_path.stem + ".txt")
        gt_anns = load_gt_masks(label_path, h, w)

        tp, fp, fn, matched_ious = match_and_score(gt_anns, pred_masks, iou_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(matched_ious)

        prec = tp/(tp+fp) if tp+fp > 0 else 0
        rec = tp/(tp+fn) if tp+fn > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0

        per_image.append({
            "name": img_path.name, "gt": len(gt_anns), "pred": len(pred_masks),
            "tp": tp, "fp": fp, "fn": fn, "f1": f1, "time": elapsed,
            "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0,
        })

        del pred_masks, gt_anns, label_map, proc_img
        if idx % 5 == 0:
            print(f"  [{idx+1}/{len(image_list)}] {img_path.name}: "
                  f"GT={per_image[-1]['gt']} Pred={per_image[-1]['pred']} "
                  f"F1={f1:.3f} [{elapsed:.1f}s]")

    prec = total_tp/(total_tp+total_fp) if total_tp+total_fp > 0 else 0
    rec = total_tp/(total_tp+total_fn) if total_tp+total_fn > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
    miou = float(np.mean(all_ious)) if all_ious else 0

    return {
        "precision": prec, "recall": rec, "f1": f1, "mean_iou": miou,
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
        "per_image": per_image,
    }


def main():
    images_dir = DATA_ROOT / "images" / "val"
    labels_dir = DATA_ROOT / "labels_polygon" / "val"
    image_list = sorted(images_dir.glob("*.png"))

    print(f"CellposeSAM Evaluation v2 on {len(image_list)} validation images")
    print("=" * 100)

    configs = [
        ("default_cp0", {"cellprob_threshold": 0.0, "diameter": 30.0}),
        ("optimized_cpm2", {"cellprob_threshold": -2.0, "diameter": 30.0}),
        ("cpm3", {"cellprob_threshold": -3.0, "diameter": 30.0}),
        ("cpm1", {"cellprob_threshold": -1.0, "diameter": 30.0}),
        ("cpm2_preproc", {"cellprob_threshold": -2.0, "diameter": 30.0, "preprocess": True}),
        ("cpm2_auto_d", {"cellprob_threshold": -2.0, "auto_diameter": True}),
    ]

    all_metrics = {}
    for name, kwargs in configs:
        print(f"\n>>> Config: {name}")
        result = run_eval(image_list, labels_dir, name, **kwargs)
        all_metrics[name] = result
        print(f"  TOTAL: Prec={result['precision']:.4f} Rec={result['recall']:.4f} "
              f"F1={result['f1']:.4f} mIoU={result['mean_iou']:.4f} "
              f"TP={result['total_tp']} FP={result['total_fp']} FN={result['total_fn']}")
        gc.collect()

    print(f"\n{'='*120}")
    print("CELLPOSESAM EVALUATION SUMMARY")
    print(f"{'='*120}")
    print(f"{'Config':<25} {'Prec':>7} {'Recall':>7} {'F1':>7} {'mIoU':>7}  {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 100)
    for name in sorted(all_metrics.keys(), key=lambda x: -all_metrics[x]["f1"]):
        m = all_metrics[name]
        print(f"{name:<25} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['mean_iou']:>7.4f}  "
              f"{m['total_tp']:>5} {m['total_fp']:>5} {m['total_fn']:>5}")

    best_name = max(all_metrics.keys(), key=lambda x: all_metrics[x]["f1"])
    best = all_metrics[best_name]
    print(f"\nBEST: {best_name} → F1={best['f1']:.4f}")

    worst = sorted(best["per_image"], key=lambda x: x["f1"])[:5]
    print(f"\n--- Worst 5 images ({best_name}) ---")
    for pi in worst:
        print(f"  {pi['name']}: F1={pi['f1']:.3f} (GT={pi['gt']}, Pred={pi['pred']}, "
              f"TP={pi['tp']}, FP={pi['fp']}, FN={pi['fn']}, IoU={pi['mean_iou']:.3f})")


if __name__ == "__main__":
    main()
