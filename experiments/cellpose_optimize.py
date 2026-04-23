#!/usr/bin/env python3
"""
CellposeSAM optimization: focus on reducing false positives.

Analysis of baseline:
- d=30: TP=996 FP=1163 FN=320 -> 54% of detections are FP
- Most FP are likely: debris, staining artifacts, overlapping fragments, background blobs
- Need better post-processing to filter non-cell detections

Strategies:
1. Stricter area filtering (too-small and too-large masks)
2. Circularity filtering (cells are roughly round, artifacts are irregular) 
3. Intensity-based filtering (cells have different intensity profile than background)
4. Mean intensity check (cells should be within certain RGB range)
5. Background contrast check (cell should be darker/different from surrounding)
6. Multi-scale consensus (only keep detections found at multiple diameters)
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
import math
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}


def load_gt_masks(label_path, img_h, img_w):
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


def match_detections(gt_masks, det_masks, iou_threshold=0.3):
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


def enhanced_postprocess(label_map, img_rgb, 
                          min_area=150, max_area=8000, 
                          min_circularity=0.3,
                          min_solidity=0.6,
                          contrast_threshold=10.0):
    """Enhanced post-processing to filter false positives."""
    cleaned = np.zeros_like(label_map)
    next_id = 1
    ids = np.unique(label_map)
    ids = ids[ids != 0]
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb
    
    for uid in ids:
        mask = (label_map == uid).astype(np.uint8)
        area = int(np.sum(mask))
        
        if area < min_area or area > max_area:
            continue
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < min_area:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * math.pi * cnt_area / (perimeter ** 2)
            if circularity < min_circularity:
                continue
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = cnt_area / hull_area
            if solidity < min_solidity:
                continue
        
        # Intensity contrast check: cell interior should differ from border region
        mask_bool = mask.astype(bool)
        cell_pixels = gray[mask_bool]
        if len(cell_pixels) == 0:
            continue
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(mask, kernel, iterations=2)
        border_mask = (dilated > 0) & (~mask_bool)
        border_pixels = gray[border_mask]
        if len(border_pixels) > 10:
            contrast = abs(float(np.mean(cell_pixels)) - float(np.mean(border_pixels)))
            if contrast < contrast_threshold:
                continue
        
        cleaned[mask_bool] = next_id
        next_id += 1
    
    return cleaned


def run_cellpose(img, diameter):
    from labeling_tool.cellpose_utils import _get_model
    model = _get_model(gpu=True)
    result = model.eval([img], diameter=float(diameter), channels=[0, 0])
    return result[0][0]


def run_with_preprocess(img, diameter):
    """Run with CLAHE preprocessing for low-contrast images."""
    from labeling_tool.cellpose_utils import assess_image_quality, adaptive_preprocess
    quality = assess_image_quality(img)
    img_enhanced = adaptive_preprocess(img, quality)
    from labeling_tool.cellpose_utils import _get_model
    model = _get_model(gpu=True)
    result = model.eval([img_enhanced], diameter=float(diameter), channels=[0, 0])
    return result[0][0], quality


def main():
    print("=" * 70)
    print("CELLPOSESAM OPTIMIZATION: Reducing False Positives")
    print("=" * 70)

    val_items = []
    img_dir = DATA_ROOT / "images" / "val"
    lbl_dir = DATA_ROOT / "labels_polygon" / "val"
    for ip in sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / (ip.stem + ".txt")
        if lbl.exists():
            val_items.append({"image_path": str(ip), "label_path": str(lbl), "name": ip.name})
    print(f"Testing on {len(val_items)} images")

    # First analyze GT cell properties to calibrate thresholds
    print("\n--- GT Cell Statistics ---")
    gt_areas = []
    for item in val_items:
        from skimage import io as skio
        img = skio.imread(item["image_path"])
        if img.ndim == 2:
            h, w = img.shape
        else:
            h, w = img.shape[:2]
        gt_masks = load_gt_masks(Path(item["label_path"]), h, w)
        for gm in gt_masks:
            gt_areas.append(gm["area"])
    gt_areas = np.array(gt_areas)
    print(f"  GT areas: min={gt_areas.min()}, p5={np.percentile(gt_areas, 5):.0f}, "
          f"median={np.median(gt_areas):.0f}, p95={np.percentile(gt_areas, 95):.0f}, max={gt_areas.max()}")

    configs = [
        {"name": "baseline (no filter)", "min_area": 50, "max_area": 100000, "min_circ": 0.0, "min_solid": 0.0, "contrast": 0.0},
        {"name": "area_only [150,8000]", "min_area": 150, "max_area": 8000, "min_circ": 0.0, "min_solid": 0.0, "contrast": 0.0},
        {"name": "area+circ 0.3", "min_area": 150, "max_area": 8000, "min_circ": 0.3, "min_solid": 0.0, "contrast": 0.0},
        {"name": "area+circ+solid", "min_area": 150, "max_area": 8000, "min_circ": 0.3, "min_solid": 0.6, "contrast": 0.0},
        {"name": "full filter c=5", "min_area": 150, "max_area": 8000, "min_circ": 0.3, "min_solid": 0.6, "contrast": 5.0},
        {"name": "full filter c=10", "min_area": 150, "max_area": 8000, "min_circ": 0.3, "min_solid": 0.6, "contrast": 10.0},
        {"name": "full filter c=15", "min_area": 150, "max_area": 8000, "min_circ": 0.3, "min_solid": 0.6, "contrast": 15.0},
        {"name": "strict area [200,5000]", "min_area": 200, "max_area": 5000, "min_circ": 0.35, "min_solid": 0.65, "contrast": 8.0},
        {"name": "loose area [100,10000]", "min_area": 100, "max_area": 10000, "min_circ": 0.25, "min_solid": 0.55, "contrast": 5.0},
        {"name": "circ 0.4 + c=8", "min_area": 150, "max_area": 8000, "min_circ": 0.4, "min_solid": 0.6, "contrast": 8.0},
        {"name": "circ 0.2 + c=8", "min_area": 150, "max_area": 8000, "min_circ": 0.2, "min_solid": 0.6, "contrast": 8.0},
        {"name": "balanced", "min_area": 150, "max_area": 6000, "min_circ": 0.3, "min_solid": 0.6, "contrast": 8.0},
        {"name": "preprocess+balanced", "min_area": 150, "max_area": 6000, "min_circ": 0.3, "min_solid": 0.6, "contrast": 8.0, "preprocess": True},
    ]

    diameter = 30.0

    results = {}
    for cfg in configs:
        name = cfg["name"]
        use_preprocess = cfg.get("preprocess", False)
        stats = {"tp": 0, "fp": 0, "fn": 0, "ious": [], "det_counts": []}
        
        for idx, item in enumerate(val_items):
            from skimage import io as skio
            img = skio.imread(item["image_path"])
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[-1] == 4:
                img = img[:, :, :3]
            h, w = img.shape[:2]
            
            gt_masks = load_gt_masks(Path(item["label_path"]), h, w)
            if not gt_masks:
                continue
            
            if use_preprocess:
                lm, _ = run_with_preprocess(img, diameter)
            else:
                lm = run_cellpose(img, diameter)
            
            # Apply enhanced post-processing
            if cfg["min_circ"] > 0 or cfg["min_solid"] > 0 or cfg["contrast"] > 0:
                lm = enhanced_postprocess(lm, img,
                    min_area=cfg["min_area"], max_area=cfg["max_area"],
                    min_circularity=cfg["min_circ"], min_solidity=cfg["min_solid"],
                    contrast_threshold=cfg["contrast"])
            
            det_masks = label_map_to_masks(lm, min_area=cfg["min_area"])
            match = match_detections(gt_masks, det_masks, iou_threshold=0.3)
            stats["tp"] += match["tp"]
            stats["fp"] += match["fp"]
            stats["fn"] += match["fn"]
            stats["det_counts"].append(len(det_masks))
            for m in match["matches"]:
                stats["ious"].append(m["iou"])
        
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        mean_iou = float(np.mean(stats["ious"])) if stats["ious"] else 0.0
        avg_det = float(np.mean(stats["det_counts"]))
        
        results[name] = {"prec": prec, "rec": rec, "f1": f1, "iou": mean_iou, 
                          "tp": tp, "fp": fp, "fn": fn, "avg_det": avg_det}
        print(f"\n  [{name}]")
        print(f"    TP={tp} FP={fp} FN={fn}")
        print(f"    Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} IoU={mean_iou:.4f}")
        print(f"    Avg det/img={avg_det:.1f}")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY (d=30)")
    print("=" * 100)
    header = f"{'Config':<30} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'Det/img':>7}"
    print(header)
    print("-" * 95)
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["f1"])
    for name, v in sorted_results:
        print(f"{name:<30} {v['prec']:>7.4f} {v['rec']:>7.4f} {v['f1']:>7.4f} {v['iou']:>7.4f} "
              f"{v['tp']:>6} {v['fp']:>6} {v['fn']:>6} {v['avg_det']:>7.1f}")


if __name__ == "__main__":
    main()
