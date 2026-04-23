#!/usr/bin/env python3
"""
CellposeSAM segmentation benchmark.
Compares original vs enhanced preprocessing on data2 images.
Measures: detection count, IoU with GT annotations, precision/recall of detection.
"""
import sys
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_gt_masks(label_path, img_h, img_w):
    """Load GT annotations as binary masks."""
    masks = []
    if not label_path.exists():
        return masks
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        if cid not in CLASS_NAMES:
            continue
        pts = [float(x) for x in parts[1:]]
        xs = [pts[i] * img_w for i in range(0, len(pts), 2)]
        ys = [pts[i] * img_h for i in range(1, len(pts), 2)]
        rr, cc = sk_polygon(ys, xs, shape=(img_h, img_w))
        if len(rr) == 0:
            continue
        mask = np.zeros((img_h, img_w), dtype=bool)
        mask[rr, cc] = True
        masks.append({"class_id": cid, "mask": mask, "area": int(np.sum(mask))})
    return masks


def masks_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0.0


def label_map_to_masks(label_map, min_area=50):
    masks = []
    ids = np.unique(label_map)
    ids = ids[ids != 0]
    for uid in ids:
        m = (label_map == uid)
        area = int(np.sum(m))
        if area >= min_area:
            masks.append({"mask": m, "area": area})
    return masks


def match_detections(gt_masks, det_masks, iou_threshold=0.3):
    """Match detections to GT using Hungarian-like greedy matching."""
    matched_gt = set()
    matched_det = set()
    matches = []

    iou_matrix = np.zeros((len(gt_masks), len(det_masks)))
    for i, gm in enumerate(gt_masks):
        for j, dm in enumerate(det_masks):
            iou_matrix[i, j] = masks_iou(gm["mask"], dm["mask"])

    while True:
        if iou_matrix.size == 0:
            break
        best_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        best_iou = iou_matrix[best_idx]
        if best_iou < iou_threshold:
            break
        i, j = best_idx
        matches.append({"gt_idx": i, "det_idx": j, "iou": float(best_iou)})
        matched_gt.add(i)
        matched_det.add(j)
        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    tp = len(matches)
    fp = len(det_masks) - len(matched_det)
    fn = len(gt_masks) - len(matched_gt)
    return {"tp": tp, "fp": fp, "fn": fn, "matches": matches}


def run_cellpose_original(img_path, diameters):
    """Run cellpose with ORIGINAL code (no preprocessing)."""
    from skimage import io as skio
    from labeling_tool.cellpose_utils import _get_model

    img = skio.imread(str(img_path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]

    model = _get_model(gpu=True)

    if len(diameters) == 1:
        result = model.eval([img], diameter=float(diameters[0]), channels=[0, 0])
        return result[0][0], img
    else:
        combined = np.zeros(img.shape[:2], dtype=np.int32)
        next_id = 1
        for d in diameters:
            result = model.eval([img], diameter=float(d), channels=[0, 0])
            lm = result[0][0]
            ids = np.unique(lm)
            ids = ids[ids != 0]
            for uid in ids:
                mask = lm == uid
                overlap = combined[mask]
                if overlap.any() and np.sum(overlap > 0) > 0.5 * np.sum(mask):
                    continue
                combined[mask & (combined == 0)] = next_id
                next_id += 1
        return combined, img


def run_cellpose_enhanced(img_path, diameters):
    """Run cellpose with ENHANCED preprocessing."""
    from skimage import io as skio
    from labeling_tool.cellpose_utils import (
        _get_model, assess_image_quality, adaptive_preprocess,
        postprocess_segmentation
    )

    img = skio.imread(str(img_path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]

    quality = assess_image_quality(img)
    img_enhanced = adaptive_preprocess(img, quality)

    model = _get_model(gpu=True)

    if len(diameters) == 1:
        result = model.eval([img_enhanced], diameter=float(diameters[0]), channels=[0, 0])
        label_map = result[0][0]
    else:
        combined = np.zeros(img.shape[:2], dtype=np.int32)
        next_id = 1
        for d in diameters:
            result = model.eval([img_enhanced], diameter=float(d), channels=[0, 0])
            lm = result[0][0]
            ids = np.unique(lm)
            ids = ids[ids != 0]
            for uid in ids:
                mask = lm == uid
                overlap = combined[mask]
                if overlap.any() and np.sum(overlap > 0) > 0.5 * np.sum(mask):
                    continue
                combined[mask & (combined == 0)] = next_id
                next_id += 1
        label_map = combined

    label_map = postprocess_segmentation(label_map, min_area=50)
    return label_map, img, quality


def main():
    print("=" * 70)
    print("CELLPOSESAM SEGMENTATION BENCHMARK")
    print("=" * 70)

    val_items = []
    img_dir = DATA_ROOT / "images" / "val"
    lbl_dir = DATA_ROOT / "labels_polygon" / "val"
    for ip in sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / (ip.stem + ".txt")
        if lbl.exists():
            val_items.append({"image_path": str(ip), "label_path": str(lbl), "name": ip.name})

    print(f"Testing on {len(val_items)} val images")

    diameters_list = [[30.0], [20.0, 30.0, 45.0]]

    for diams in diameters_list:
        diam_str = "+".join(str(int(d)) for d in diams)
        print(f"\n{'='*70}")
        print(f"DIAMETER CONFIG: {diam_str}")
        print(f"{'='*70}")

        orig_stats = {"tp": 0, "fp": 0, "fn": 0, "ious": [], "det_counts": [], "gt_counts": [],
                      "time": 0}
        enh_stats = {"tp": 0, "fp": 0, "fn": 0, "ious": [], "det_counts": [], "gt_counts": [],
                     "time": 0, "clahe_count": 0, "denoise_count": 0}

        for idx, item in enumerate(val_items):
            ip = item["image_path"]
            lp = Path(item["label_path"])
            from skimage import io as skio
            tmp_img = skio.imread(ip)
            if tmp_img.ndim == 2:
                h, w = tmp_img.shape
            else:
                h, w = tmp_img.shape[:2]

            gt_masks = load_gt_masks(lp, h, w)
            if not gt_masks:
                continue

            # Original
            t0 = time.time()
            lm_orig, _ = run_cellpose_original(ip, diams)
            orig_stats["time"] += time.time() - t0
            det_orig = label_map_to_masks(lm_orig)
            match_orig = match_detections(gt_masks, det_orig, iou_threshold=0.3)
            orig_stats["tp"] += match_orig["tp"]
            orig_stats["fp"] += match_orig["fp"]
            orig_stats["fn"] += match_orig["fn"]
            orig_stats["det_counts"].append(len(det_orig))
            orig_stats["gt_counts"].append(len(gt_masks))
            for m in match_orig["matches"]:
                orig_stats["ious"].append(m["iou"])

            # Enhanced
            t0 = time.time()
            lm_enh, _, quality = run_cellpose_enhanced(ip, diams)
            enh_stats["time"] += time.time() - t0
            det_enh = label_map_to_masks(lm_enh)
            match_enh = match_detections(gt_masks, det_enh, iou_threshold=0.3)
            enh_stats["tp"] += match_enh["tp"]
            enh_stats["fp"] += match_enh["fp"]
            enh_stats["fn"] += match_enh["fn"]
            enh_stats["det_counts"].append(len(det_enh))
            enh_stats["gt_counts"].append(len(gt_masks))
            for m in match_enh["matches"]:
                enh_stats["ious"].append(m["iou"])
            if quality["needs_clahe"]:
                enh_stats["clahe_count"] += 1
            if quality["needs_denoise"]:
                enh_stats["denoise_count"] += 1

            if (idx + 1) % 6 == 0:
                print(f"  Processed {idx+1}/{len(val_items)} images...")

        # Print results
        for label, stats in [("ORIGINAL", orig_stats), ("ENHANCED", enh_stats)]:
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            mean_iou = float(np.mean(stats["ious"])) if stats["ious"] else 0.0
            mean_det = float(np.mean(stats["det_counts"]))
            mean_gt = float(np.mean(stats["gt_counts"]))

            print(f"\n  [{label}] (d={diam_str})")
            print(f"    TP={tp} FP={fp} FN={fn}")
            print(f"    Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
            print(f"    Mean IoU (matched)={mean_iou:.4f}")
            print(f"    Avg detections/image={mean_det:.1f}  Avg GT/image={mean_gt:.1f}")
            print(f"    Total time={stats['time']:.1f}s  ({stats['time']/len(val_items):.2f}s/img)")
            if label == "ENHANCED":
                print(f"    CLAHE applied: {stats['clahe_count']}/{len(val_items)} images")
                print(f"    Denoise applied: {stats['denoise_count']}/{len(val_items)} images")

    # Save results
    print(f"\nDone.")


if __name__ == "__main__":
    main()
