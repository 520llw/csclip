#!/usr/bin/env python3
"""
Final segmentation evaluation on data1_organized with optimized and default params.
Reports: overall + per-class + per-image stats, compares default vs optimized.
"""
import os, sys, time, gc
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
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
        return {"tp": 0, "fp": len(det_masks), "fn": len(gt_masks),
                "matches": [], "tp_gt_classes": []}
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
    tp_gt_classes = [gt_masks[m["gt_idx"]]["class_id"] for m in matches]
    fn_gt_classes = [gt_masks[i]["class_id"] for i in range(len(gt_masks)) if i not in matched_gt]
    return {"tp": len(matches), "fp": len(det_masks) - len(matched_det),
            "fn": len(gt_masks) - len(matched_gt), "matches": matches,
            "tp_gt_classes": tp_gt_classes, "fn_gt_classes": fn_gt_classes}


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


def run_config(name, images, gt_all, filenames, model, diameter, cellprob_thr, flow_thr=0.4):
    print(f"\n{'='*80}")
    print(f"CONFIG: {name} (d={diameter}, cp={cellprob_thr}, ft={flow_thr})")
    print(f"{'='*80}")

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []
    per_class_tp = defaultdict(int)
    per_class_gt = defaultdict(int)
    per_class_fn = defaultdict(int)
    per_img_f1 = []
    t0 = time.time()

    for idx, (img, gt, fn) in enumerate(zip(images, gt_all, filenames)):
        result = model.eval([img], diameter=diameter,
                            cellprob_threshold=cellprob_thr,
                            flow_threshold=flow_thr)
        lm = result[0][0]
        det = label_map_to_masks(lm, min_area=50)
        match = match_detections(gt, det)

        total_tp += match["tp"]
        total_fp += match["fp"]
        total_fn += match["fn"]

        for m in match["matches"]:
            all_ious.append(m["iou"])
        for cid in match["tp_gt_classes"]:
            per_class_tp[cid] += 1
        for cid in match.get("fn_gt_classes", []):
            per_class_fn[cid] += 1
        for gm in gt:
            per_class_gt[gm["class_id"]] += 1

        p_img = match["tp"] / max(match["tp"] + match["fp"], 1)
        r_img = match["tp"] / max(match["tp"] + match["fn"], 1)
        f1_img = 2 * p_img * r_img / max(p_img + r_img, 1e-9)
        per_img_f1.append(f1_img)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(images):
            spd = (idx + 1) / (time.time() - t0)
            print(f"  [{idx+1}/{len(images)}] {spd:.2f} img/s", flush=True)

    elapsed = time.time() - t0
    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print(f"\n{'─'*70}")
    print(f"RESULTS — {name}")
    print(f"{'─'*70}")
    print(f"  Total pred:     {total_tp + total_fp}")
    print(f"  Total GT:       {total_tp + total_fn}")
    print(f"  TP (IoU>={IOU_THR}): {total_tp}")
    print(f"  FP:             {total_fp}")
    print(f"  FN:             {total_fn}")
    print(f"  Precision:      {prec:.4f}")
    print(f"  Recall:         {rec:.4f}")
    print(f"  F1 (overall):   {f1:.4f}")
    if all_ious:
        print(f"  mIoU (matched): {np.mean(all_ious):.4f} ± {np.std(all_ious):.4f}")
    print(f"  F1 (per-img):   {np.mean(per_img_f1):.4f} ± {np.std(per_img_f1):.4f}")
    print(f"  Time:           {elapsed:.1f}s ({len(images)/max(elapsed,1):.2f} img/s)")

    print(f"\n  Per-class detection rate (GT matched / GT total):")
    for cid in sorted(CLASS_NAMES.keys()):
        gt_n = per_class_gt.get(cid, 0)
        tp_n = per_class_tp.get(cid, 0)
        rate = tp_n / gt_n if gt_n > 0 else 0.0
        print(f"    {CLASS_NAMES[cid]:<15} {tp_n:>5}/{gt_n:<5} = {rate:.4f}")

    return {"prec": prec, "rec": rec, "f1": f1,
            "miou": float(np.mean(all_ious)) if all_ious else 0,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "per_class_tp": dict(per_class_tp), "per_class_gt": dict(per_class_gt)}


def main():
    print("=" * 90)
    print("CellposeSAM Segmentation Evaluation on data1_organized")
    print("=" * 90)

    img_dir = DATA_ROOT / "images" / "val"
    lbl_dir = DATA_ROOT / "labels_polygon" / "val"
    extensions = (".jpg", ".jpeg", ".png")

    images, gt_all, filenames = [], [], []
    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() not in extensions:
            continue
        lbl = lbl_dir / (ip.stem + ".txt")
        if not lbl.exists():
            continue
        img = np.array(Image.open(str(ip)).convert("RGB"))
        h, w = img.shape[:2]
        gt = load_gt_masks(lbl, h, w)
        if gt:
            images.append(img)
            gt_all.append(gt)
            filenames.append(ip.name)

    print(f"Val images: {len(images)}")
    total_gt = sum(len(g) for g in gt_all)
    print(f"Total GT cells: {total_gt}")

    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    # Default CellposeSAM parameters
    res_default = run_config("Default CellposeSAM", images, gt_all, filenames,
                             model, diameter=30.0, cellprob_thr=0.0, flow_thr=0.4)

    # Optimized parameters from sweep
    res_opt = run_config("Optimized (d=60, cp=-2, ft=0.3)", images, gt_all, filenames,
                         model, diameter=60.0, cellprob_thr=-2.0, flow_thr=0.3)

    # Also test d=50, cp=-3, ft=0.3 (2nd best)
    res_opt2 = run_config("Optimized-2 (d=50, cp=-3, ft=0.3)", images, gt_all, filenames,
                          model, diameter=50.0, cellprob_thr=-3.0, flow_thr=0.3)

    # Comparison
    print(f"\n\n{'='*90}")
    print("COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<40} {'Prec':>7} {'Rec':>7} {'F1':>7} {'mIoU':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 90)
    for name, r in [("Default (d=30, cp=0)", res_default),
                    ("Optimized (d=60, cp=-2, ft=0.3)", res_opt),
                    ("Optimized-2 (d=50, cp=-3, ft=0.3)", res_opt2)]:
        print(f"{name:<40} {r['prec']:>7.4f} {r['rec']:>7.4f} {r['f1']:>7.4f} "
              f"{r['miou']:>7.4f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")

    imp = res_opt['f1'] - res_default['f1']
    print(f"\nF1 improvement: {imp:+.4f} ({imp/max(res_default['f1'],1e-9)*100:+.1f}%)")


if __name__ == "__main__":
    main()
