#!/usr/bin/env python3
"""
Evaluate Cellpose-SAM (v4) segmentation on data2 and MultiCenter.

Cellpose 4.1.1 includes Cellpose-SAM which uses SAM's ViT-L encoder.
By default, CellposeModel loads the 'cpsam' model.

Tests:
  1. Default CellposeSAM (auto diameter)
  2. CellposeSAM with d=50 (our optimized diameter)
  3. CellposeSAM + PAMSR (multi-scale rescue)
  4. Various diameter and cellprob configurations
"""
import sys, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.stdout.reconfigure(line_buffering=True)

DATA2_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
MC_ROOT = Path("/home/xut/csclip/cell_datasets/MultiCenter_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}


def load_gt(lbl_path, h, w):
    anns = []
    if not lbl_path.exists():
        return anns
    for line in open(lbl_path):
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
        anns.append(mask)
    return anns


def match(gt_masks, pred_masks, iou_thr=0.5):
    tp, matched = 0, set()
    n_pred = pred_masks.max() if pred_masks.max() > 0 else 0
    for pid in range(1, n_pred + 1):
        pm = pred_masks == pid
        if pm.sum() < 30:
            continue
        best, bi = 0, -1
        for gi, gm in enumerate(gt_masks):
            if gi in matched:
                continue
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou = inter / union if union > 0 else 0
            if iou > best:
                best, bi = iou, gi
        if best >= iou_thr and bi >= 0:
            tp += 1
            matched.add(bi)
    return tp, n_pred - tp, len(gt_masks) - tp


def pamsr(model, img, primary_d=50, secondary_ds=[40, 65],
          cellprob=-3.0, rescue_prob_thr=0.0, rescue_need_consensus=True,
          min_area=80, overlap_thr=0.2):
    h, w = img.shape[:2]
    masks_p, flows_p, _ = model.eval(img, diameter=primary_d,
                                     cellprob_threshold=cellprob, channels=[0, 0])
    prob_map_p = flows_p[2]
    final = masks_p.copy()
    nid = final.max() + 1

    secondary_cells = []
    for sd in secondary_ds:
        masks_s, flows_s, _ = model.eval(img, diameter=sd,
                                         cellprob_threshold=cellprob, channels=[0, 0])
        prob_map_s = flows_s[2]
        for cid in range(1, masks_s.max() + 1):
            cm = masks_s == cid
            area = cm.sum()
            if area < min_area:
                continue
            overlap_with_primary = np.sum(final[cm] > 0)
            if overlap_with_primary > overlap_thr * area:
                continue
            mp = float(prob_map_s[cm].mean())
            secondary_cells.append({"mask": cm, "mp": mp, "d": sd, "area": area})
        del masks_s, flows_s
        gc.collect()

    if rescue_need_consensus and len(secondary_ds) > 1:
        for i, c1 in enumerate(secondary_cells):
            c1["has_consensus"] = False
            for j, c2 in enumerate(secondary_cells):
                if i == j or c1["d"] == c2["d"]:
                    continue
                inter = np.logical_and(c1["mask"], c2["mask"]).sum()
                union = np.logical_or(c1["mask"], c2["mask"]).sum()
                if union > 0 and inter / union > 0.3:
                    c1["has_consensus"] = True
                    break
    else:
        for c in secondary_cells:
            c["has_consensus"] = True

    stats = {"primary": nid - 1, "rescued": 0, "rej": 0}
    for c in sorted(secondary_cells, key=lambda x: -x["mp"]):
        passes = c["mp"] > rescue_prob_thr
        if rescue_need_consensus:
            passes = passes and c["has_consensus"]
        if not passes:
            stats["rej"] += 1
            continue
        overlap_with_final = np.sum(final[c["mask"]] > 0)
        if overlap_with_final > overlap_thr * c["area"]:
            stats["rej"] += 1
            continue
        final[c["mask"] & (final == 0)] = nid
        nid += 1
        stats["rescued"] += 1

    return final, stats


def eval_dataset(model, dataset_name, data_root, img_ext="png"):
    img_dir = data_root / "images" / "val"
    lbl_dir = data_root / "labels_polygon" / "val"

    images = sorted(list(img_dir.glob(f"*.{img_ext}")) +
                    list(img_dir.glob("*.jpg")) +
                    list(img_dir.glob("*.jpeg")))
    images = list(dict.fromkeys(images))  # deduplicate

    if not images:
        print(f"  WARNING: No images found in {img_dir}", flush=True)
        return

    configs = [
        ("cpsam_auto",        {"channels": [0, 0]}),
        ("cpsam_d50_cp0",     {"diameter": 50, "cellprob_threshold": 0.0, "channels": [0, 0]}),
        ("cpsam_d50_cp-1",    {"diameter": 50, "cellprob_threshold": -1.0, "channels": [0, 0]}),
        ("cpsam_d50_cp-3",    {"diameter": 50, "cellprob_threshold": -3.0, "channels": [0, 0]}),
        ("cpsam_d40_cp-3",    {"diameter": 40, "cellprob_threshold": -3.0, "channels": [0, 0]}),
        ("cpsam_d60_cp-3",    {"diameter": 60, "cellprob_threshold": -3.0, "channels": [0, 0]}),
        ("cpsam_d30_cp-3",    {"diameter": 30, "cellprob_threshold": -3.0, "channels": [0, 0]}),
        ("pamsr_cpsam",       "pamsr"),
        ("pamsr_cpsam_p1",    "pamsr_p1"),
    ]

    agg = defaultdict(lambda: [0, 0, 0])
    pamsr_stats = defaultdict(lambda: {"primary": 0, "rescued": 0, "rej": 0})

    for idx, ip in enumerate(images):
        lp = lbl_dir / (ip.stem + ".txt")
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        gt = load_gt(lp, h, w)
        if not gt:
            continue

        for cname, params in configs:
            if cname.startswith("pamsr"):
                p_thr = 1.0 if "p1" in cname else 0.0
                masks, stats = pamsr(model, img, primary_d=50, secondary_ds=[40, 65],
                                     cellprob=-3.0, rescue_prob_thr=p_thr,
                                     rescue_need_consensus=True)
                tp, fp, fn = match(gt, masks)
                for k, v in stats.items():
                    pamsr_stats[cname][k] += v
                del masks
            else:
                masks, _, _ = model.eval(img, **params)
                tp, fp, fn = match(gt, masks)
                del masks

            agg[cname][0] += tp
            agg[cname][1] += fp
            agg[cname][2] += fn
            gc.collect()

        if (idx + 1) % 5 == 0 or idx == 0 or idx == len(images) - 1:
            print(f"  [{idx+1}/{len(images)}] {ip.stem}", flush=True)
        gc.collect()

    print(f"\n{'='*110}")
    print(f"CELLPOSE-SAM RESULTS — {dataset_name} val ({len(images)} images)")
    print(f"{'='*110}")
    print(f"{'Method':<25} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}  PAMSR-stats")
    print("-" * 110)

    rows = []
    for name, (tp, fp, fn) in agg.items():
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        rows.append((name, tp, fp, fn, p, r, f1))

    for n, tp, fp, fn, p, r, f1 in sorted(rows, key=lambda x: -x[-1]):
        ps = pamsr_stats.get(n, {})
        ps_str = f"pri={ps.get('primary','-')} res={ps.get('rescued','-')} rej={ps.get('rej','-')}" if ps else ""
        print(f"{n:<25} {tp:>5} {fp:>5} {fn:>5} {p:>7.4f} {r:>7.4f} {f1:>7.4f}  {ps_str}", flush=True)

    best = max(rows, key=lambda x: x[-1])
    print(f"\n*** BEST: {best[0]} -> F1={best[-1]:.4f} ***")
    return {name: {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}
            for name, tp, fp, fn, p, r, f1 in rows}


def main():
    from cellpose import models

    print("="*80)
    print("Cellpose-SAM (v4) Segmentation Evaluation")
    print("="*80)

    model = models.CellposeModel(gpu=True)
    print(f"Model: {getattr(model, 'pretrained_model', 'unknown')}", flush=True)

    print("\n--- data2_organized ---", flush=True)
    d2_results = eval_dataset(model, "data2_organized", DATA2_ROOT, "png")

    print("\n--- MultiCenter_organized ---", flush=True)
    mc_results = eval_dataset(model, "MultiCenter_organized", MC_ROOT, "jpg")

    print("\n\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Dataset':<20} {'Best Config':<25} {'F1':>7}")
    print("-" * 55)
    if d2_results:
        best_d2 = max(d2_results.items(), key=lambda x: x[1]["f1"])
        print(f"{'data2':<20} {best_d2[0]:<25} {best_d2[1]['f1']:>7.4f}")
    if mc_results:
        best_mc = max(mc_results.items(), key=lambda x: x[1]["f1"])
        print(f"{'MultiCenter':<20} {best_mc[0]:<25} {best_mc[1]['f1']:>7.4f}")

    print("\nPrevious baselines:")
    print(f"  data2 CellposeModel d50 cp-3: F1=0.7261")
    print(f"  data2 PAMSR:                   F1=0.7276")
    print(f"  MC optimized d50:              F1=0.5517")


if __name__ == "__main__":
    main()
