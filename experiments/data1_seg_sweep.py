#!/usr/bin/env python3
"""
CellposeSAM parameter sweep on data1_organized (clinical BALF dataset).
Grid search over diameter and cellprob_threshold on val set.
Images: 853x640 JPG, 7 classes, no EXIF rotation.
"""
import os, sys, time, gc
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data1_organized")
CLASS_NAMES = {0: "CCEC", 1: "RBC", 2: "SEC", 3: "Eosinophil",
               4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
IOU_THR = 0.5

def load_gt_masks(label_path, img_h, img_w):
    masks = []
    if not label_path.exists():
        return masks
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
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


def match_detections(gt_masks, det_masks, iou_threshold=IOU_THR):
    if not gt_masks or not det_masks:
        return {"tp": 0, "fp": len(det_masks), "fn": len(gt_masks), "matches": []}
    iou_matrix = np.zeros((len(gt_masks), len(det_masks)))
    for i, gm in enumerate(gt_masks):
        for j, dm in enumerate(det_masks):
            iou_matrix[i, j] = masks_iou(gm["mask"], dm["mask"])
    matched_gt, matched_det, matches = set(), set(), []
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
    return {"tp": len(matches), "fp": len(det_masks) - len(matched_det),
            "fn": len(gt_masks) - len(matched_gt), "matches": matches}


def label_map_to_masks(label_map, min_area=50):
    masks = []
    for uid in np.unique(label_map):
        if uid == 0:
            continue
        m = (label_map == uid)
        area = int(np.sum(m))
        if area >= min_area:
            masks.append({"mask": m, "area": area})
    return masks


def main():
    print("=" * 90)
    print("CellposeSAM Parameter Sweep on data1_organized (val set)")
    print("=" * 90)

    img_dir = DATA_ROOT / "images" / "val"
    lbl_dir = DATA_ROOT / "labels_polygon" / "val"
    extensions = (".jpg", ".jpeg", ".png")
    val_items = []
    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() in extensions:
            lbl = lbl_dir / (ip.stem + ".txt")
            if lbl.exists():
                val_items.append({"image_path": str(ip), "label_path": str(lbl)})
    print(f"Val images with labels: {len(val_items)}")

    images, gt_all = [], []
    for item in val_items:
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        gt = load_gt_masks(Path(item["label_path"]), h, w)
        if gt:
            images.append(img)
            gt_all.append(gt)
    print(f"Valid images (with GT): {len(images)}")

    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    # Grid search: diameter × cellprob_threshold
    diameters = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0]
    cellprob_thrs = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    results = []
    total_configs = len(diameters) * len(cellprob_thrs)
    config_idx = 0

    for diam in diameters:
        for cp in cellprob_thrs:
            config_idx += 1
            t0 = time.time()
            stats = {"tp": 0, "fp": 0, "fn": 0, "ious": []}
            for img, gt in zip(images, gt_all):
                result = model.eval([img], diameter=diam, cellprob_threshold=cp)
                lm = result[0][0]
                det = label_map_to_masks(lm, min_area=50)
                match = match_detections(gt, det)
                stats["tp"] += match["tp"]
                stats["fp"] += match["fp"]
                stats["fn"] += match["fn"]
                for m in match["matches"]:
                    stats["ious"].append(m["iou"])

            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            iou = float(np.mean(stats["ious"])) if stats["ious"] else 0.0
            elapsed = time.time() - t0
            results.append({"diam": diam, "cp": cp, "prec": prec, "rec": rec,
                            "f1": f1, "iou": iou, "tp": tp, "fp": fp, "fn": fn})
            print(f"  [{config_idx}/{total_configs}] d={diam:.0f} cp={cp:.1f} "
                  f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} IoU={iou:.4f} "
                  f"TP={tp} FP={fp} FN={fn} ({elapsed:.1f}s)")

    # Also sweep flow_threshold at top-3 configs
    results.sort(key=lambda x: -x["f1"])
    top3 = results[:3]
    print("\nTop-3 configs for flow_threshold sweep:")
    for r in top3:
        print(f"  d={r['diam']:.0f} cp={r['cp']:.1f} F1={r['f1']:.4f}")

    ft_results = []
    for r in top3:
        for ft in [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]:
            t0 = time.time()
            stats = {"tp": 0, "fp": 0, "fn": 0, "ious": []}
            for img, gt in zip(images, gt_all):
                result = model.eval([img], diameter=r["diam"],
                                    cellprob_threshold=r["cp"],
                                    flow_threshold=ft)
                lm = result[0][0]
                det = label_map_to_masks(lm, min_area=50)
                match = match_detections(gt, det)
                stats["tp"] += match["tp"]
                stats["fp"] += match["fp"]
                stats["fn"] += match["fn"]
                for m in match["matches"]:
                    stats["ious"].append(m["iou"])
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            iou = float(np.mean(stats["ious"])) if stats["ious"] else 0.0
            elapsed = time.time() - t0
            ft_results.append({"diam": r["diam"], "cp": r["cp"], "ft": ft,
                                "prec": prec, "rec": rec, "f1": f1, "iou": iou,
                                "tp": tp, "fp": fp, "fn": fn})
            print(f"  d={r['diam']:.0f} cp={r['cp']:.1f} ft={ft:.1f} "
                  f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} IoU={iou:.4f} ({elapsed:.1f}s)")

    # Final summary
    all_configs = results + ft_results
    all_configs.sort(key=lambda x: -x["f1"])

    print(f"\n{'='*100}")
    print("FINAL RANKING (Top-15 by F1)")
    print(f"{'='*100}")
    header = f"{'Config':<35} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7} {'TP':>6} {'FP':>6} {'FN':>6}"
    print(header)
    print("-" * 95)
    for r in all_configs[:15]:
        ft_str = f" ft={r.get('ft', 0.4):.1f}" if 'ft' in r else ""
        name = f"d={r['diam']:.0f} cp={r['cp']:.1f}{ft_str}"
        print(f"{name:<35} {r['prec']:>7.4f} {r['rec']:>7.4f} {r['f1']:>7.4f} "
              f"{r['iou']:>7.4f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")

    best = all_configs[0]
    print(f"\nBEST CONFIG: d={best['diam']:.0f}, cp={best['cp']:.1f}"
          f"{', ft=' + str(best.get('ft','default')) if 'ft' in best else ''}")
    print(f"  F1={best['f1']:.4f}, Prec={best['prec']:.4f}, Rec={best['rec']:.4f}, IoU={best['iou']:.4f}")


if __name__ == "__main__":
    main()
