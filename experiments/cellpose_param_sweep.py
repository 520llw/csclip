#!/usr/bin/env python3
"""
CellposeSAM parameter sweep: cellprob_threshold and flow_threshold.
These control detection sensitivity at the model level, much better than post-hoc filtering.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
import math
from pathlib import Path

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
    matched_gt, matched_det, matches = set(), set(), []
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
    return {"tp": len(matches), "fp": len(det_masks)-len(matched_det), 
            "fn": len(gt_masks)-len(matched_gt), "matches": matches}


def label_map_to_masks(label_map, min_area=80):
    masks = []
    ids = np.unique(label_map)
    for uid in ids[ids != 0]:
        m = (label_map == uid)
        area = int(np.sum(m))
        if area >= min_area:
            masks.append({"mask": m, "area": area})
    return masks


def main():
    print("=" * 70)
    print("CELLPOSESAM PARAMETER SWEEP")
    print("=" * 70)

    val_items = []
    img_dir = DATA_ROOT / "images" / "val"
    lbl_dir = DATA_ROOT / "labels_polygon" / "val"
    for ip in sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / (ip.stem + ".txt")
        if lbl.exists():
            val_items.append({"image_path": str(ip), "label_path": str(lbl)})
    print(f"Images: {len(val_items)}")

    from labeling_tool.cellpose_utils import _get_model, assess_image_quality, adaptive_preprocess
    model = _get_model(gpu=True)

    # Preload images
    from skimage import io as skio
    images = []
    gt_all = []
    for item in val_items:
        img = skio.imread(item["image_path"])
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
        h, w = img.shape[:2]
        gt = load_gt_masks(Path(item["label_path"]), h, w)
        if gt:
            images.append(img)
            gt_all.append(gt)
    print(f"Valid images: {len(images)}")

    diameter = 30.0
    
    configs = []
    # Sweep cellprob_threshold (higher = fewer but more confident detections)
    for cp in [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        configs.append({"cp": cp, "ft": 0.4, "preprocess": False, "label": f"cp={cp:.1f}"})
    
    # Sweep flow_threshold at best cellprob
    for ft in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        configs.append({"cp": 2.0, "ft": ft, "preprocess": False, "label": f"cp=2.0 ft={ft:.1f}"})
    
    # With preprocessing at promising configs
    for cp in [1.0, 1.5, 2.0, 2.5]:
        configs.append({"cp": cp, "ft": 0.4, "preprocess": True, "label": f"PRE cp={cp:.1f}"})
    
    # With tile_norm for uneven illumination
    for cp in [1.0, 1.5, 2.0]:
        configs.append({"cp": cp, "ft": 0.4, "tile_norm": 100, "label": f"TILE cp={cp:.1f}"})

    results = []
    for cfg in configs:
        stats = {"tp": 0, "fp": 0, "fn": 0, "ious": []}
        for img, gt in zip(images, gt_all):
            if cfg.get("preprocess", False):
                quality = assess_image_quality(img)
                run_img = adaptive_preprocess(img, quality)
            else:
                run_img = img
            
            normalize_cfg = True
            if cfg.get("tile_norm"):
                normalize_cfg = {"tile_norm_blocksize": cfg["tile_norm"]}
            
            result = model.eval([run_img], diameter=diameter, channels=[0, 0],
                                cellprob_threshold=cfg["cp"], flow_threshold=cfg["ft"],
                                normalize=normalize_cfg)
            lm = result[0][0]
            det = label_map_to_masks(lm, min_area=80)
            match = match_detections(gt, det)
            stats["tp"] += match["tp"]
            stats["fp"] += match["fp"]
            stats["fn"] += match["fn"]
            for m in match["matches"]:
                stats["ious"].append(m["iou"])
        
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec = tp/(tp+fn) if (tp+fn) else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        iou = float(np.mean(stats["ious"])) if stats["ious"] else 0.0
        results.append({"name": cfg["label"], "prec": prec, "rec": rec, "f1": f1, 
                         "iou": iou, "tp": tp, "fp": fp, "fn": fn})
        print(f"  {cfg['label']:<25} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} IoU={iou:.4f} TP={tp} FP={fp} FN={fn}")

    print("\n" + "=" * 100)
    print("SUMMARY sorted by F1")
    print("=" * 100)
    header = f"{'Config':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7} {'TP':>6} {'FP':>6} {'FN':>6}"
    print(header)
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x["f1"]):
        print(f"{r['name']:<25} {r['prec']:>7.4f} {r['rec']:>7.4f} {r['f1']:>7.4f} {r['iou']:>7.4f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")


if __name__ == "__main__":
    main()
