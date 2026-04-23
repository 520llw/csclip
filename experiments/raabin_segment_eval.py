#!/usr/bin/env python3
"""
Raabin-WBC-top3 segmentation evaluation.

Each Raabin image is already a cropped single WBC. GT label is a bbox covering
the full image (normalized 0.5 0.5 1.0 1.0) — i.e., "the whole image is one cell".

We evaluate our segmentation pipeline (Cellpose / Cellpose+SAM3 / Otsu baseline)
on a stratified subsample (100/class = 300 images). Metrics:
  - Detection rate: % images with >= 1 detected cell
  - Correct-count rate: % images with EXACTLY 1 detected cell (ideal)
  - Mean foreground coverage: mask_area / image_area (should be ~0.3-0.8 for a
    real cell, close to 1.0 means the model just segments the whole image)
  - Mean IoU vs. GT full-image bbox: intersection(mask, full_img) / union
    (since GT is full image, this equals mask_area / image_area)
  - Mean mask-bbox IoU with GT bbox: bbox of predicted mask vs full image bbox
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

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/Raabin-WBC-top3")
IMG_DIR = DATA_ROOT / "images"
LBL_DIR = DATA_ROOT / "labels"
SAM3_CKPT = Path("/home/xut/csclip/labeling_tool/weights/sam3.pt")
OUT_LOG = Path("/tmp/raabin_segment_eval.log")

CLASS_NAMES = {0: "Eosinophil", 1: "Lymphocyte", 2: "Neutrophil"}
N_PER_CLASS = 100
SEED = 42


def read_label(label_path: Path):
    with label_path.open() as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    return int(parts[0]), [float(x) for x in parts[1:]]


def build_sampled_index():
    """Stratified sample N_PER_CLASS per class."""
    pools = {c: [] for c in CLASS_NAMES}
    for img in sorted(IMG_DIR.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = LBL_DIR / (img.stem + ".txt")
        if not lbl.exists():
            continue
        info = read_label(lbl)
        if info is None:
            continue
        cid, _ = info
        if cid in pools:
            pools[cid].append(img)
    random.seed(SEED)
    selected = []
    for cid, pool in pools.items():
        n = min(N_PER_CLASS, len(pool))
        if len(pool) > n:
            sub = random.sample(pool, n)
        else:
            sub = pool
        selected.extend([(p, cid) for p in sub])
    return selected


# ============== Otsu baseline (from raabin_extract.py) ==============

def otsu_segment(img_rgb, min_area_px=100):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    instances = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        m = (labels == i)
        instances.append(m)
    return instances


# ============== Cellpose ==============

def cellpose_segment(img_rgb, model, diameter=30.0, min_area_px=100):
    """Run CellposeSAM (cpsam) and return list of instance boolean masks.
    Filter instances smaller than min_area_px absolute pixels (previously used
    a 0.5% relative threshold which breaks on large multi-cell images).
    """
    result = model.eval([img_rgb], diameter=float(diameter), channels=[0, 0],
                        cellprob_threshold=-2.0)
    # Force standard ndarray to avoid fastremap/Cellpose C-subclass ndarrays
    label_map = np.array(result[0][0], dtype=np.int32, copy=True)
    ids = np.unique(label_map)
    ids = ids[ids != 0]
    instances = []
    for uid in ids:
        m = np.array(label_map == uid, dtype=bool, copy=True)
        if int(m.sum()) < min_area_px:
            continue
        instances.append(m)
    return instances


# ============== SAM3 refinement ==============

def sam3_refine(img_rgb, instances, sam3_proc, image_path):
    """Refine each Cellpose mask by re-prompting SAM3 with the instance bbox."""
    if not instances:
        return []
    from PIL import Image as PImage
    image = PImage.fromarray(img_rgb)
    w, h = image.size
    state = sam3_proc.set_image(image)
    refined = []
    try:
        for m in instances:
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            try:
                sam3_proc.reset_all_prompts(state)
            except Exception:
                pass
            # Use predict_inst with bbox prompt
            import numpy as _np
            try:
                masks, scores, _ = sam3_proc.model.predict_inst(
                    state,
                    box=_np.array([x1, y1, x2, y2], dtype=_np.float32),
                    multimask_output=False,
                    normalize_coords=True,
                )
                if masks is None or len(masks) == 0:
                    refined.append(m)
                    continue
                mm = masks[0].squeeze()
                if isinstance(mm, torch.Tensor):
                    mm = mm.detach().cpu().numpy()
                bin_mask = (mm > 0.5) if mm.dtype != bool else mm
                if bin_mask.shape != m.shape:
                    bin_mask = cv2.resize(bin_mask.astype(np.uint8), (w, h),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                # Fuse: if SAM3 mask is similar enough (IoU>0.5) use it, else keep Cellpose
                inter = np.logical_and(bin_mask, m).sum()
                union = np.logical_or(bin_mask, m).sum()
                iou = inter / max(union, 1)
                if iou > 0.5:
                    refined.append(bin_mask)
                else:
                    refined.append(m)
            except Exception as e:
                print(f"    SAM3 error: {e}", flush=True)
                refined.append(m)
    finally:
        try:
            sam3_proc.reset_all_prompts(state)
        except Exception:
            pass
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return refined


# ============== Metrics ==============

def mask_metrics(instances, img_shape):
    """Return: n_inst, largest_coverage, bbox_iou_vs_full."""
    h, w = img_shape[:2]
    img_area = h * w
    if not instances:
        return 0, 0.0, 0.0
    areas = [m.sum() for m in instances]
    largest = int(np.argmax(areas))
    m = instances[largest]
    coverage = float(m.sum()) / img_area
    # Bbox of mask
    ys, xs = np.where(m)
    if len(xs) == 0:
        return len(instances), 0.0, 0.0
    mx1, mx2 = xs.min(), xs.max()
    my1, my2 = ys.min(), ys.max()
    pred_bbox = (mx1, my1, mx2, my2)
    full_bbox = (0, 0, w - 1, h - 1)
    # IoU of bboxes
    ix1 = max(pred_bbox[0], full_bbox[0])
    iy1 = max(pred_bbox[1], full_bbox[1])
    ix2 = min(pred_bbox[2], full_bbox[2])
    iy2 = min(pred_bbox[3], full_bbox[3])
    if ix2 < ix1 or iy2 < iy1:
        bbox_iou = 0.0
    else:
        inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
        a1 = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)
        a2 = (full_bbox[2] - full_bbox[0] + 1) * (full_bbox[3] - full_bbox[1] + 1)
        bbox_iou = inter / (a1 + a2 - inter)
    return len(instances), coverage, float(bbox_iou)


def aggregate(results):
    n = len(results)
    n_inst = np.array([r[0] for r in results])
    cov = np.array([r[1] for r in results])
    bbiou = np.array([r[2] for r in results])
    return {
        "detection_rate": float((n_inst >= 1).mean()),
        "single_rate": float((n_inst == 1).mean()),
        "mean_inst": float(n_inst.mean()),
        "mean_coverage": float(cov.mean()),
        "std_coverage": float(cov.std()),
        "mean_bbox_iou": float(bbiou.mean()),
    }


def main():
    print("=" * 100)
    print("Raabin-WBC-top3 Segmentation Pipeline Evaluation")
    print(f"Sampling {N_PER_CLASS} per class = {N_PER_CLASS * 3} total")
    print("=" * 100, flush=True)

    samples = build_sampled_index()
    print(f"Selected {len(samples)} images")
    from collections import Counter
    print("  Distribution:", Counter(c for _, c in samples))

    # Load models
    print("\n[Loading Cellpose cpsam ...]", flush=True)
    from cellpose import models as cp_models
    cp_model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model="cpsam")
    print("  Cellpose loaded")

    print("\n[Loading SAM3 ...]", flush=True)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3_pkg = Path(sys.modules["sam3"].__file__).parent
    bpe_path = sam3_pkg / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        bpe_path = sam3_pkg.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=str(SAM3_CKPT),
        device=device,
        eval_mode=True,
        load_from_HF=False,
        enable_inst_interactivity=True,
    )
    sam3_proc = Sam3Processor(sam3_model, confidence_threshold=0.3, device=device)
    print("  SAM3 loaded")

    # Run
    otsu_res, cp_res, sam_res = [], [], []
    per_class_res = {c: {"otsu": [], "cp": [], "cpsam": []} for c in CLASS_NAMES}

    t0 = time.time()
    for i, (path, cid) in enumerate(samples):
        img = np.array(Image.open(path).convert("RGB"))

        r1 = mask_metrics(otsu_segment(img), img.shape)
        otsu_res.append(r1); per_class_res[cid]["otsu"].append(r1)

        cp_instances = cellpose_segment(img, cp_model)
        r2 = mask_metrics(cp_instances, img.shape)
        cp_res.append(r2); per_class_res[cid]["cp"].append(r2)

        cpsam_instances = sam3_refine(img, cp_instances, sam3_proc, path)
        r3 = mask_metrics(cpsam_instances, img.shape)
        sam_res.append(r3); per_class_res[cid]["cpsam"].append(r3)

        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(samples)}] {time.time()-t0:.1f}s", flush=True)

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    print("\n" + "=" * 100)
    print("Overall Results")
    print("=" * 100)
    print(f"{'Method':<20} {'DetRate':>8} {'1Cell%':>8} {'MeanInst':>9} {'Cover':>8} {'±std':>8} {'BBoxIoU':>8}")
    print("-" * 100)
    for name, res in [("Otsu (baseline)", otsu_res),
                       ("Cellpose (cpsam)", cp_res),
                       ("Cellpose+SAM3", sam_res)]:
        a = aggregate(res)
        print(f"{name:<20} {a['detection_rate']:>8.3f} {a['single_rate']:>8.3f} "
              f"{a['mean_inst']:>9.2f} {a['mean_coverage']:>8.3f} "
              f"±{a['std_coverage']:>6.3f} {a['mean_bbox_iou']:>8.3f}")

    print("\nPer-class breakdown (Cellpose+SAM3):")
    for cid, name in CLASS_NAMES.items():
        a = aggregate(per_class_res[cid]["cpsam"])
        print(f"  {name:<12} det={a['detection_rate']:.3f}  single={a['single_rate']:.3f}  "
              f"cover={a['mean_coverage']:.3f}  bbox_iou={a['mean_bbox_iou']:.3f}")


if __name__ == "__main__":
    main()
