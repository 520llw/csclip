#!/usr/bin/env python3
"""
WBC-Seg benchmark: evaluate our segmentation pipeline against gold-standard
YOLO-seg polygons on /home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset.

Metrics:
  - Semantic IoU / Dice (pred vs GT, all cells merged into foreground)
  - Instance-level Precision/Recall/F1 @ IoU=0.5

Methods compared:
  1. Otsu        - classical threshold + morphology (CPU, sub-ms/image)
  2. CellposeSAM - cpsam pretrained (Cellpose 4.x, SAM-based, our pipeline's
                   actual segmentation model)
Note: CellposeSAM is a single model (not Cellpose + SAM3 cascade).
"""
from __future__ import annotations
import os, sys, time, random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

sys.path.insert(0, "/home/xut/csclip")
sys.path.insert(0, "/home/xut/csclip/sam3")
sys.path.insert(0, "/home/xut/csclip/experiments")

# --- Patch cellpose.utils.fill_holes_and_remove_small_masks for NumPy 2.0 ---
# fastremap.unique() returns a C-subclassed ndarray that trips np.nonzero on NumPy 2.0
def _patch_cellpose_numpy2():
    from cellpose import utils as cp_utils
    from scipy.ndimage import find_objects
    try:
        import fill_voids
    except ImportError:
        fill_voids = None

    def fill_holes_and_remove_small_masks_patched(masks, min_size=15):
        # Minimal replacement using pure numpy/scipy (no fastremap internals).
        masks = np.asarray(masks)
        if min_size > 0:
            uniq, counts = np.unique(masks, return_counts=True)
            nonzero = uniq != 0
            uniq_nz = uniq[nonzero]
            counts_nz = counts[nonzero]
            to_remove = uniq_nz[counts_nz < min_size]
            if len(to_remove) > 0:
                masks = np.where(np.isin(masks, to_remove), 0, masks)
        # Fill holes per instance using scipy or fill_voids
        slices = find_objects(masks)
        new_masks = np.zeros_like(masks)
        j = 0
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            m = (masks[slc] == (i + 1))
            if fill_voids is not None:
                try:
                    m = fill_voids.fill(m)
                except Exception:
                    pass
            j += 1
            new_masks[slc][m] = j
        return new_masks

    cp_utils.fill_holes_and_remove_small_masks = fill_holes_and_remove_small_masks_patched
    # Also patch in dynamics module (where it may be imported)
    try:
        from cellpose import dynamics as cp_dyn
        cp_dyn.fill_holes_and_remove_small_masks = fill_holes_and_remove_small_masks_patched
    except Exception:
        pass

_patch_cellpose_numpy2()

from raabin_segment_eval import otsu_segment, cellpose_segment

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset")
SAM3_CKPT = Path("/home/xut/csclip/labeling_tool/weights/sam3.pt")

SPLIT = "val"  # evaluate on val split
IMG_DIR = DATA_ROOT / "images" / SPLIT
LBL_DIR = DATA_ROOT / "labels" / SPLIT


# =============== Ground-truth parsing ===============

def parse_yolo_seg_label(lbl_path: Path, w: int, h: int):
    """Parse YOLO-seg label into list of instance masks of shape (h, w).
    Each line: class_id x1 y1 x2 y2 ... (normalized)
    """
    instances = []
    if not lbl_path.exists():
        return instances
    with lbl_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + at least 3 (x,y) points
                continue
            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            pts = []
            for i in range(0, len(coords), 2):
                x = int(np.clip(coords[i] * w, 0, w - 1))
                y = int(np.clip(coords[i + 1] * h, 0, h - 1))
                pts.append([x, y])
            if len(pts) < 3:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
            if mask.sum() >= 10:
                instances.append(mask.astype(bool))
    return instances


# =============== Metrics ===============

def semantic_iou_dice(pred_inst, gt_inst, shape):
    h, w = shape
    if gt_inst:
        gt_union = np.zeros((h, w), dtype=bool)
        for m in gt_inst:
            gt_union |= m
    else:
        gt_union = np.zeros((h, w), dtype=bool)
    if pred_inst:
        pred_union = np.zeros((h, w), dtype=bool)
        for m in pred_inst:
            pred_union |= m
    else:
        pred_union = np.zeros((h, w), dtype=bool)
    inter = np.logical_and(pred_union, gt_union).sum()
    union = np.logical_or(pred_union, gt_union).sum()
    iou = inter / union if union > 0 else (1.0 if not gt_inst and not pred_inst else 0.0)
    dice = (2 * inter) / (pred_union.sum() + gt_union.sum()) if (pred_union.sum() + gt_union.sum()) > 0 else \
           (1.0 if not gt_inst and not pred_inst else 0.0)
    return float(iou), float(dice)


def _bbox_of(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (xs.min(), ys.min(), xs.max(), ys.max())


def _bboxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def instance_tpfpfn(pred_inst, gt_inst, iou_thr=0.5):
    """Greedy matching by IoU with bbox pre-filtering.
    Returns (TP, FP, FN, matched_iou_sum, n_matched)."""
    if not pred_inst and not gt_inst:
        return 0, 0, 0, 0.0, 0
    if not pred_inst:
        return 0, 0, len(gt_inst), 0.0, 0
    if not gt_inst:
        return 0, len(pred_inst), 0, 0.0, 0
    M, N = len(pred_inst), len(gt_inst)

    # Precompute bboxes and areas
    pred_bboxes = [_bbox_of(m) for m in pred_inst]
    gt_bboxes = [_bbox_of(m) for m in gt_inst]
    pred_areas = np.array([m.sum() for m in pred_inst], dtype=np.int64)
    gt_areas = np.array([m.sum() for m in gt_inst], dtype=np.int64)

    # Only compute IoU for pairs whose bboxes overlap
    candidates = []  # (iou, i, j)
    for i in range(M):
        bi = pred_bboxes[i]
        if bi is None:
            continue
        pi = pred_inst[i]
        # Crop to bbox
        x1, y1, x2, y2 = bi
        pi_crop = pi[y1:y2+1, x1:x2+1]
        for j in range(N):
            bj = gt_bboxes[j]
            if bj is None or not _bboxes_overlap(bi, bj):
                continue
            # Compute intersection in overlapping bbox region
            ix1 = max(bi[0], bj[0]); iy1 = max(bi[1], bj[1])
            ix2 = min(bi[2], bj[2]); iy2 = min(bi[3], bj[3])
            pi_reg = pi[iy1:iy2+1, ix1:ix2+1]
            gj_reg = gt_inst[j][iy1:iy2+1, ix1:ix2+1]
            inter = int(np.logical_and(pi_reg, gj_reg).sum())
            if inter == 0:
                continue
            union = int(pred_areas[i] + gt_areas[j] - inter)
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_thr * 0.5:  # keep candidates with at least half threshold
                candidates.append((iou, i, j))

    # Greedy matching: repeatedly pick highest IoU
    matched_p = set()
    matched_g = set()
    iou_sum = 0.0
    n_match = 0
    candidates.sort(reverse=True)
    for iou_val, i, j in candidates:
        if iou_val < iou_thr:
            break
        if i in matched_p or j in matched_g:
            continue
        matched_p.add(i); matched_g.add(j)
        iou_sum += iou_val
        n_match += 1
    tp = n_match
    fp = M - tp
    fn = N - tp
    return tp, fp, fn, iou_sum, n_match


def aggregate_semantic(rows):
    arr = np.array(rows)
    return {"mean_iou": float(arr[:, 0].mean()),
            "std_iou": float(arr[:, 0].std()),
            "mean_dice": float(arr[:, 1].mean())}


def aggregate_instance(tps, fps, fns, iou_sums, matches):
    tp = sum(tps); fp = sum(fps); fn = sum(fns)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    mean_iou_matched = sum(iou_sums) / max(sum(matches), 1)
    return {"P50": prec, "R50": rec, "F1_50": f1,
            "mean_matched_iou": float(mean_iou_matched),
            "TP": tp, "FP": fp, "FN": fn}


# =============== Main ===============

def main():
    print("=" * 100)
    print(f"WBC-Seg benchmark ({SPLIT} split)")
    print(f"Ground truth: YOLO-seg polygons with ~{22684 if SPLIT=='val' else 110930} instances")
    print("=" * 100, flush=True)

    imgs = sorted(IMG_DIR.iterdir())
    imgs = [p for p in imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"  {len(imgs)} images found\n")

    # Load models once
    print("[Loading Cellpose cpsam ...]", flush=True)
    from cellpose import models as cp_models
    cp_model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model="cpsam")

    print(f"  CellposeSAM on device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n", flush=True)

    # Storage
    sem = {"otsu": [], "cpsam": []}
    inst_tp = {k: [] for k in ["otsu", "cpsam"]}
    inst_fp = {k: [] for k in ["otsu", "cpsam"]}
    inst_fn = {k: [] for k in ["otsu", "cpsam"]}
    inst_iousum = {k: [] for k in ["otsu", "cpsam"]}
    inst_nm = {k: [] for k in ["otsu", "cpsam"]}

    # Checkpoint file: skip already-processed images on rerun
    import json
    ckpt_path = Path("/tmp/wbc_seg_ckpt.jsonl")
    processed = set()
    if ckpt_path.exists():
        with ckpt_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    processed.add(d["name"])
                    # restore
                    sem["otsu"].append(tuple(d["sem_otsu"]))
                    sem["cpsam"].append(tuple(d["sem_cpsam"]))
                    for tag in ("otsu", "cpsam"):
                        inst_tp[tag].append(d[f"inst_{tag}"][0])
                        inst_fp[tag].append(d[f"inst_{tag}"][1])
                        inst_fn[tag].append(d[f"inst_{tag}"][2])
                        inst_iousum[tag].append(d[f"inst_{tag}"][3])
                        inst_nm[tag].append(d[f"inst_{tag}"][4])
                except Exception:
                    pass
        print(f"  Restored {len(processed)} images from checkpoint\n", flush=True)

    t0 = time.time()
    for i, img_path in enumerate(imgs):
        if img_path.name in processed:
            continue
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            h, w = img.shape[:2]

            gt = parse_yolo_seg_label(LBL_DIR / (img_path.stem + ".txt"), w, h)
            if not gt:
                continue

            ins_o = otsu_segment(img)
            ins_c = cellpose_segment(img, cp_model)

            so = semantic_iou_dice(ins_o, gt, (h, w))
            sc = semantic_iou_dice(ins_c, gt, (h, w))
            sem["otsu"].append(so); sem["cpsam"].append(sc)

            rec = {"name": img_path.name,
                   "sem_otsu": list(so), "sem_cpsam": list(sc)}
            for tag, preds in [("otsu", ins_o), ("cpsam", ins_c)]:
                tp, fp, fn, iou_sum, nm = instance_tpfpfn(preds, gt, iou_thr=0.5)
                inst_tp[tag].append(tp); inst_fp[tag].append(fp); inst_fn[tag].append(fn)
                inst_iousum[tag].append(iou_sum); inst_nm[tag].append(nm)
                rec[f"inst_{tag}"] = [int(tp), int(fp), int(fn), float(iou_sum), int(nm)]
            with ckpt_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(imgs)}] {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(imgs)}] ERROR on {img_path.name}: {e}", flush=True)
            continue

    print(f"\nTotal new-image time: {time.time()-t0:.1f}s\n")

    # Report
    print("=" * 100)
    print("Semantic Segmentation (pixel-level, all cells merged)")
    print("=" * 100)
    print(f"{'Method':<20} {'Mean IoU':>10} {'Std IoU':>10} {'Mean Dice':>12}")
    print("-" * 100)
    for tag, name in [("otsu", "Otsu"),
                       ("cpsam", "CellposeSAM")]:
        a = aggregate_semantic(sem[tag])
        print(f"{name:<20} {a['mean_iou']:>10.4f} {a['std_iou']:>10.4f} {a['mean_dice']:>12.4f}")

    print("\n" + "=" * 100)
    print("Instance Segmentation @ IoU=0.5")
    print("=" * 100)
    print(f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MeanIoU*':>10} "
          f"{'TP':>7} {'FP':>7} {'FN':>7}")
    print("-" * 100)
    for tag, name in [("otsu", "Otsu"),
                       ("cpsam", "CellposeSAM")]:
        a = aggregate_instance(inst_tp[tag], inst_fp[tag], inst_fn[tag],
                                inst_iousum[tag], inst_nm[tag])
        print(f"{name:<20} {a['P50']:>10.4f} {a['R50']:>10.4f} {a['F1_50']:>10.4f} "
              f"{a['mean_matched_iou']:>10.4f} {a['TP']:>7} {a['FP']:>7} {a['FN']:>7}")
    print("\n* MeanIoU = mean IoU of matched pred-GT pairs (only successful matches)")


if __name__ == "__main__":
    main()
