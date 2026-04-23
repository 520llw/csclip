#!/usr/bin/env python3
"""
Multi-Scale Consensus with Probability-guided Filtering (MSCPF)
===============================================================
CellposeSAM segmentation innovations:
  1. Multi-Scale Consensus: run at 3 diameters, keep cells detected at 2+ scales
  2. Cell Probability Confidence: use CellposeSAM's probability map to score cells
  3. Flow Radial Consistency: use flow field to assess segmentation quality
"""
import gc, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
MC_ROOT = Path("/home/xut/csclip/cell_datasets/MultiCenter_organized")
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


def compute_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0


def match_and_score(gt_anns, pred_masks, iou_thr=0.5):
    tp, matched_gt = 0, set()
    n_pred = pred_masks.max() if pred_masks.max() > 0 else 0
    matched_ious = []

    for pid in range(1, n_pred + 1):
        pm = pred_masks == pid
        if pm.sum() == 0:
            continue
        best_iou, best_gi = 0, -1
        for gi, ga in enumerate(gt_anns):
            if gi in matched_gt:
                continue
            iou = compute_iou(pm, ga["mask"])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thr and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)
            matched_ious.append(best_iou)

    fp = n_pred - tp
    fn = len(gt_anns) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    miou = float(np.mean(matched_ious)) if matched_ious else 0
    return {"tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f1": f1, "miou": miou}


# ==================== Innovation 1: Multi-Scale Consensus ====================

def run_cellpose_single(img, diameter, cellprob_threshold=-3.0):
    from cellpose import models
    model = models.CellposeModel(gpu=True)
    masks, flows, styles = model.eval(
        img, diameter=diameter, cellprob_threshold=cellprob_threshold, channels=[0, 0])
    del model
    gc.collect()
    return masks, flows


def multi_scale_consensus(img, diameters=[40, 50, 65], cellprob=-3.0,
                          consensus_iou=0.3, prob_thr=-1.0, flow_thr=0.5):
    """Run CellposeSAM at multiple scales, keep cells with multi-scale consensus.

    Returns: final_masks, stats_dict
    """
    from cellpose import models
    model = models.CellposeModel(gpu=True)

    scale_results = []
    for d in diameters:
        masks, flows, _ = model.eval(
            img, diameter=d, cellprob_threshold=cellprob, channels=[0, 0])
        prob_map = flows[2]
        flow_field = flows[1]
        scale_results.append({"masks": masks, "prob_map": prob_map, "flow": flow_field})

    del model
    gc.collect()

    h, w = img.shape[:2]
    all_cells = []

    for si, sr in enumerate(scale_results):
        masks = sr["masks"]
        for cid in range(1, masks.max() + 1):
            cell_mask = masks == cid
            if cell_mask.sum() < 50:
                continue
            ys, xs = np.where(cell_mask)
            cy, cx = ys.mean(), xs.mean()
            area = cell_mask.sum()

            mean_prob = sr["prob_map"][cell_mask].mean()

            flow = sr["flow"]
            fy, fx = flow[0][cell_mask], flow[1][cell_mask]
            dy_to_center = cy - ys.astype(float)
            dx_to_center = cx - xs.astype(float)
            norms = np.sqrt(dy_to_center ** 2 + dx_to_center ** 2) + 1e-8
            dy_to_center /= norms
            dx_to_center /= norms
            flow_norms = np.sqrt(fy ** 2 + fx ** 2) + 1e-8
            fy_n = fy / flow_norms
            fx_n = fx / flow_norms
            radial_cos = fy_n * dy_to_center + fx_n * dx_to_center
            radial_consistency = float(np.mean(radial_cos))

            all_cells.append({
                "scale_idx": si, "mask": cell_mask, "cx": cx, "cy": cy,
                "area": area, "mean_prob": float(mean_prob),
                "radial_consistency": radial_consistency,
            })

    n_scales = len(diameters)
    for i, cell in enumerate(all_cells):
        cross_scale_count = 1
        for j, other in enumerate(all_cells):
            if i == j or other["scale_idx"] == cell["scale_idx"]:
                continue
            iou = compute_iou(cell["mask"], other["mask"])
            if iou > consensus_iou:
                cross_scale_count += 1
                break
        cell["consensus_count"] = min(cross_scale_count, n_scales)

    final_masks = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    accepted = {"consensus": 0, "prob_pass": 0, "rejected": 0}

    sorted_cells = sorted(all_cells, key=lambda c: (-c["consensus_count"], -c["mean_prob"]))

    for cell in sorted_cells:
        overlap = final_masks[cell["mask"]]
        if overlap.any() and np.sum(overlap > 0) > 0.3 * cell["area"]:
            continue

        if cell["consensus_count"] >= 2:
            final_masks[cell["mask"] & (final_masks == 0)] = next_id
            next_id += 1
            accepted["consensus"] += 1
        elif cell["mean_prob"] > prob_thr and cell["radial_consistency"] > flow_thr:
            final_masks[cell["mask"] & (final_masks == 0)] = next_id
            next_id += 1
            accepted["prob_pass"] += 1
        else:
            accepted["rejected"] += 1

    stats = {
        "total_candidates": len(all_cells),
        "accepted_consensus": accepted["consensus"],
        "accepted_prob": accepted["prob_pass"],
        "rejected": accepted["rejected"],
        "final_cells": next_id - 1,
    }
    return final_masks, stats


def run_single_scale(img, diameter=50, cellprob=-3.0):
    """Baseline: single-scale CellposeSAM."""
    from cellpose import models
    model = models.CellposeModel(gpu=True)
    masks, flows, styles = model.eval(
        img, diameter=diameter, cellprob_threshold=cellprob, channels=[0, 0])
    del model
    gc.collect()
    return masks


def run_default(img):
    """Default CellposeSAM (cellprob=0, auto-diameter)."""
    from cellpose import models
    model = models.CellposeModel(gpu=True)
    masks, flows, styles = model.eval(img, channels=[0, 0])
    del model
    gc.collect()
    return masks


def evaluate_on_dataset(data_root, split="val", max_images=None, all_classes=False):
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels_polygon" / split

    if not img_dir.exists():
        print(f"  Image dir not found: {img_dir}")
        return {}

    image_files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if max_images:
        image_files = image_files[:max_images]

    configs = {
        "default":         lambda img: run_default(img),
        "single_d50_cp3":  lambda img: run_single_scale(img, 50, -3.0),
        "single_d40_cp3":  lambda img: run_single_scale(img, 40, -3.0),
    }

    mscpf_configs = [
        ("mscpf_40_50_65",        [40, 50, 65], -3.0, -1.0, 0.5),
        ("mscpf_35_50_70",        [35, 50, 70], -3.0, -1.0, 0.5),
        ("mscpf_40_50_65_pt-2",   [40, 50, 65], -3.0, -2.0, 0.5),
        ("mscpf_40_50_65_ft0.3",  [40, 50, 65], -3.0, -1.0, 0.3),
        ("mscpf_40_50_65_ft0.7",  [40, 50, 65], -3.0, -1.0, 0.7),
        ("mscpf_30_50_80",        [30, 50, 80], -3.0, -1.0, 0.5),
    ]

    results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "ious": []})

    class_filter = None if all_classes else CLASS_NAMES

    for idx, img_path in enumerate(image_files):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        gt = load_gt_masks(lbl_path, h, w)
        if not gt:
            continue

        print(f"  [{idx + 1}/{len(image_files)}] {img_path.name} ({len(gt)} GT cells)")

        for name, run_fn in configs.items():
            masks = run_fn(img)
            m = match_and_score(gt, masks)
            results[name]["tp"] += m["tp"]
            results[name]["fp"] += m["fp"]
            results[name]["fn"] += m["fn"]
            if m["miou"] > 0:
                results[name]["ious"].append(m["miou"])
            gc.collect()

        for name, diameters, cellprob, prob_thr, flow_thr in mscpf_configs:
            masks, stats = multi_scale_consensus(
                img, diameters, cellprob, prob_thr=prob_thr, flow_thr=flow_thr)
            m = match_and_score(gt, masks)
            results[name]["tp"] += m["tp"]
            results[name]["fp"] += m["fp"]
            results[name]["fn"] += m["fn"]
            if m["miou"] > 0:
                results[name]["ious"].append(m["miou"])
            del masks
            gc.collect()

        gc.collect()

    final = {}
    for name, r in results.items():
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        miou = float(np.mean(r["ious"])) if r["ious"] else 0
        final[name] = {"tp": tp, "fp": fp, "fn": fn,
                        "prec": prec, "rec": rec, "f1": f1, "miou": miou}
    return final


def main():
    print("=" * 100)
    print("MSCPF SEGMENTATION EVALUATION — data2_organized")
    print("=" * 100)

    results = evaluate_on_dataset(DATA_ROOT, "val")

    print(f"\n{'=' * 120}")
    print(f"{'Method':<35} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} {'mIoU':>7}")
    print("-" * 120)
    sr = sorted(results.items(), key=lambda x: -x[1]["f1"])
    for name, r in sr:
        print(f"{name:<35} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
              f"{r['prec']:>7.4f} {r['rec']:>7.4f} {r['f1']:>7.4f} {r['miou']:>7.4f}")

    best = sr[0]
    print(f"\n*** BEST: {best[0]} -> F1={best[1]['f1']:.4f}, "
          f"P={best[1]['prec']:.4f}, R={best[1]['rec']:.4f} ***")


if __name__ == "__main__":
    main()
