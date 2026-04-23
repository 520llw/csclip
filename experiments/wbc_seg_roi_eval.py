#!/usr/bin/env python3
"""
WBC-Seg ROI-constrained evaluation.

Problem: the YOLO-seg GT only covers roughly the upper-right half of each
3120x4160 image (coordinate system mismatch). Computing metrics on the full
image penalizes CellposeSAM for correct predictions in unannotated regions.

Solution: derive an evaluation ROI from GT (union of all GT polygons' bbox,
slightly dilated). Only keep pred instances whose centroid falls inside ROI,
and compute IoU only inside ROI.

Caches Cellpose label maps to /tmp/wbc_cpsam_cache/ so the expensive inference
step runs only once.
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

sys.path.insert(0, "/home/xut/csclip")
sys.path.insert(0, "/home/xut/csclip/experiments")

from wbc_seg_benchmark import (parse_yolo_seg_label, _patch_cellpose_numpy2,
                                instance_tpfpfn, semantic_iou_dice)
_patch_cellpose_numpy2()

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset")
IMG_DIR = DATA_ROOT / "images" / "val"
LBL_DIR = DATA_ROOT / "labels" / "val"
CACHE = Path("/tmp/wbc_cpsam_cache")
CACHE.mkdir(exist_ok=True)


def cellpose_label_map(img, model, cache_path: Path):
    if cache_path.exists():
        return np.load(cache_path)["label_map"].astype(np.int32)
    result = model.eval([img], diameter=30.0, channels=[0, 0],
                        cellprob_threshold=-2.0)
    lm = np.array(result[0][0], dtype=np.int32, copy=True)
    np.savez_compressed(cache_path, label_map=lm.astype(np.int32))
    return lm


def instances_from_label_map(lm, min_area_px=100):
    ids = np.unique(lm); ids = ids[ids != 0]
    out = []
    for uid in ids:
        m = np.array(lm == uid, dtype=bool, copy=True)
        if int(m.sum()) >= min_area_px:
            out.append(m)
    return out


def otsu_instances(img, min_area_px=100):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        out.append(np.array(labels == i, dtype=bool, copy=True))
    return out


def compute_roi(gt_inst, shape, dilate_px=20):
    """ROI = bbox of GT union, slightly dilated."""
    h, w = shape
    if not gt_inst:
        return None
    union = np.zeros((h, w), dtype=bool)
    for m in gt_inst:
        union |= m
    ys, xs = np.where(union)
    if len(xs) == 0:
        return None
    x1 = max(0, int(xs.min()) - dilate_px)
    y1 = max(0, int(ys.min()) - dilate_px)
    x2 = min(w - 1, int(xs.max()) + dilate_px)
    y2 = min(h - 1, int(ys.max()) + dilate_px)
    return (x1, y1, x2, y2)


def mask_in_roi(mask, roi):
    x1, y1, x2, y2 = roi
    return mask[y1:y2+1, x1:x2+1]


def filter_instances_in_roi(instances, roi, center_rule=True):
    """Keep instances whose centroid (if center_rule) or bbox-overlap>50%
    falls inside ROI."""
    if not instances:
        return []
    x1, y1, x2, y2 = roi
    out = []
    for m in instances:
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        if center_rule:
            cx, cy = xs.mean(), ys.mean()
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                out.append(m)
        else:
            mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()
            ix1 = max(mx1, x1); iy1 = max(my1, y1)
            ix2 = min(mx2, x2); iy2 = min(my2, y2)
            if ix2 < ix1 or iy2 < iy1:
                continue
            inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
            bbox_area = (mx2 - mx1 + 1) * (my2 - my1 + 1)
            if inter / bbox_area >= 0.5:
                out.append(m)
    return out


def semantic_iou_in_roi(pred_inst, gt_inst, shape, roi):
    """Compute semantic IoU only inside ROI."""
    h, w = shape
    x1, y1, x2, y2 = roi
    Hs, Ws = y2 - y1 + 1, x2 - x1 + 1
    gt_u = np.zeros((Hs, Ws), dtype=bool)
    for m in gt_inst:
        gt_u |= m[y1:y2+1, x1:x2+1]
    pr_u = np.zeros((Hs, Ws), dtype=bool)
    for m in pred_inst:
        pr_u |= m[y1:y2+1, x1:x2+1]
    inter = np.logical_and(pr_u, gt_u).sum()
    union = np.logical_or(pr_u, gt_u).sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (pr_u.sum() + gt_u.sum()) if (pr_u.sum() + gt_u.sum()) > 0 else 0.0
    return float(iou), float(dice)


def main():
    print("=" * 100)
    print("WBC-Seg ROI-constrained evaluation")
    print("=" * 100, flush=True)

    imgs = sorted(IMG_DIR.iterdir())
    imgs = [p for p in imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"  {len(imgs)} images found")

    # Load Cellpose only if cache incomplete
    need_cellpose = any(not (CACHE / f"{p.stem}.npz").exists() for p in imgs)
    cp_model = None
    if need_cellpose:
        print("[Loading CellposeSAM ...]", flush=True)
        from cellpose import models as cp_models
        cp_model = cp_models.CellposeModel(gpu=True, pretrained_model="cpsam")

    sem_full = {"otsu": [], "cpsam": []}
    sem_roi = {"otsu": [], "cpsam": []}
    inst_full = {k: [0, 0, 0, 0.0, 0] for k in ["otsu", "cpsam"]}   # TP,FP,FN,iou_sum,n_match
    inst_roi = {k: [0, 0, 0, 0.0, 0] for k in ["otsu", "cpsam"]}
    roi_coverage = []   # ROI area / image area

    t0 = time.time()
    for i, p in enumerate(imgs):
        try:
            img = np.array(Image.open(p).convert("RGB"))
            h, w = img.shape[:2]
            gt = parse_yolo_seg_label(LBL_DIR / (p.stem + ".txt"), w, h)
            if not gt:
                continue
            roi = compute_roi(gt, (h, w))
            if roi is None:
                continue
            rx1, ry1, rx2, ry2 = roi
            roi_area = (rx2 - rx1 + 1) * (ry2 - ry1 + 1)
            roi_coverage.append(roi_area / (h * w))

            # Predictions
            lm = cellpose_label_map(img, cp_model, CACHE / f"{p.stem}.npz")
            ins_c = instances_from_label_map(lm)
            ins_o = otsu_instances(img)

            # --- Full-image semantic ---
            io, do = semantic_iou_dice(ins_o, gt, (h, w))
            ic, dc = semantic_iou_dice(ins_c, gt, (h, w))
            sem_full["otsu"].append((io, do))
            sem_full["cpsam"].append((ic, dc))

            # --- ROI-constrained semantic ---
            io_r, do_r = semantic_iou_in_roi(ins_o, gt, (h, w), roi)
            ic_r, dc_r = semantic_iou_in_roi(ins_c, gt, (h, w), roi)
            sem_roi["otsu"].append((io_r, do_r))
            sem_roi["cpsam"].append((ic_r, dc_r))

            # --- Instance P/R full-image ---
            for tag, preds in [("otsu", ins_o), ("cpsam", ins_c)]:
                tp, fp, fn, ious, nm = instance_tpfpfn(preds, gt, iou_thr=0.5)
                inst_full[tag][0] += tp; inst_full[tag][1] += fp
                inst_full[tag][2] += fn; inst_full[tag][3] += ious
                inst_full[tag][4] += nm

            # --- Instance P/R ROI-filtered ---
            ins_o_r = filter_instances_in_roi(ins_o, roi)
            ins_c_r = filter_instances_in_roi(ins_c, roi)
            for tag, preds in [("otsu", ins_o_r), ("cpsam", ins_c_r)]:
                tp, fp, fn, ious, nm = instance_tpfpfn(preds, gt, iou_thr=0.5)
                inst_roi[tag][0] += tp; inst_roi[tag][1] += fp
                inst_roi[tag][2] += fn; inst_roi[tag][3] += ious
                inst_roi[tag][4] += nm

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(imgs)}] {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(imgs)}] ERROR {p.name}: {e}", flush=True)

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Mean ROI coverage of full image: {np.mean(roi_coverage)*100:.1f}%\n")

    # Report
    def fmt_sem(rows):
        arr = np.array(rows)
        return f"IoU={arr[:,0].mean():.4f}±{arr[:,0].std():.4f}  Dice={arr[:,1].mean():.4f}"

    def fmt_inst(s):
        tp, fp, fn, ious, nm = s
        P = tp / (tp + fp) if tp + fp > 0 else 0
        R = tp / (tp + fn) if tp + fn > 0 else 0
        F = 2 * P * R / (P + R) if P + R > 0 else 0
        miou = ious / nm if nm > 0 else 0
        return f"P={P:.3f}  R={R:.3f}  F1={F:.3f}  MatchedIoU={miou:.3f}  TP={tp} FP={fp} FN={fn}"

    print("=" * 100)
    print("Semantic IoU / Dice")
    print("=" * 100)
    print(f"{'Method':<20} {'Full image':<45} {'ROI only':<45}")
    print("-" * 110)
    for tag, name in [("otsu", "Otsu"), ("cpsam", "CellposeSAM")]:
        print(f"{name:<20} {fmt_sem(sem_full[tag]):<45} {fmt_sem(sem_roi[tag]):<45}")

    print("\n" + "=" * 100)
    print("Instance segmentation @ IoU=0.5")
    print("=" * 100)
    print("Full image:")
    for tag, name in [("otsu", "Otsu"), ("cpsam", "CellposeSAM")]:
        print(f"  {name:<14} {fmt_inst(inst_full[tag])}")
    print("\nROI-filtered (only predictions with centroid inside GT bbox hull):")
    for tag, name in [("otsu", "Otsu"), ("cpsam", "CellposeSAM")]:
        print(f"  {name:<14} {fmt_inst(inst_roi[tag])}")


if __name__ == "__main__":
    main()
