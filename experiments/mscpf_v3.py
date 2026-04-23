#!/usr/bin/env python3
"""
MSCPF v3: Primary-Anchor + Multi-Scale Rescue (PAMSR)

Strategy: Use the best single-scale (d=50) as the anchor, then selectively
rescue missed cells from secondary scales. Only adds cells that:
1. Were NOT detected by the primary scale (non-overlapping)
2. Have high cellpose probability (confidence gating)
3. Are confirmed by ≥2 secondary scales (cross-scale consensus)

This preserves the strong precision of the primary scale while improving recall.
"""
import sys, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.stdout.reconfigure(line_buffering=True)

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
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
    """
    Primary-Anchor Multi-Scale Rescue.
    1. Run primary scale → keep all
    2. Run secondary scales → extract cell candidates
    3. Rescue: add secondary cells that don't overlap primary AND pass quality gate
    """
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


def main():
    from cellpose import models
    model = models.CellposeModel(gpu=True)

    img_dir = DATA_ROOT / "images" / "val"
    lbl_dir = DATA_ROOT / "labels_polygon" / "val"
    images = sorted(img_dir.glob("*.png"))

    configs = [
        ("default", None),
        ("single_d50_cp3", None),
        # PAMSR variants
        ("pamsr_40_65_cons",      {"secondary_ds": [40, 65], "rescue_prob_thr": 0.0, "rescue_need_consensus": True}),
        ("pamsr_40_65_nocons",    {"secondary_ds": [40, 65], "rescue_prob_thr": 1.0, "rescue_need_consensus": False}),
        ("pamsr_40_65_prob2",     {"secondary_ds": [40, 65], "rescue_prob_thr": 2.0, "rescue_need_consensus": False}),
        ("pamsr_40_65_cons_p1",   {"secondary_ds": [40, 65], "rescue_prob_thr": 1.0, "rescue_need_consensus": True}),
        ("pamsr_35_65_cons",      {"secondary_ds": [35, 65], "rescue_prob_thr": 0.0, "rescue_need_consensus": True}),
        ("pamsr_40_55_70_cons",   {"secondary_ds": [40, 55, 70], "rescue_prob_thr": 0.0, "rescue_need_consensus": True}),
        ("pamsr_40_65_cons_ov01", {"secondary_ds": [40, 65], "rescue_prob_thr": 0.0, "rescue_need_consensus": True, "overlap_thr": 0.1}),
        ("pamsr_40_65_cons_p-1",  {"secondary_ds": [40, 65], "rescue_prob_thr": -1.0, "rescue_need_consensus": True}),
    ]

    agg = defaultdict(lambda: [0, 0, 0])
    total_stats = defaultdict(lambda: {"primary": 0, "rescued": 0, "rej": 0})

    for idx, ip in enumerate(images):
        lp = lbl_dir / (ip.stem + ".txt")
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        gt = load_gt(lp, h, w)
        if not gt:
            continue

        for cname, params in configs:
            if cname == "default":
                masks, _, _ = model.eval(img, channels=[0, 0])
                tp, fp, fn = match(gt, masks)
                del masks
            elif cname == "single_d50_cp3":
                masks, _, _ = model.eval(img, diameter=50, cellprob_threshold=-3.0, channels=[0, 0])
                tp, fp, fn = match(gt, masks)
                del masks
            else:
                masks, stats = pamsr(model, img, **params)
                tp, fp, fn = match(gt, masks)
                for k, v in stats.items():
                    total_stats[cname][k] += v
                del masks

            agg[cname][0] += tp
            agg[cname][1] += fp
            agg[cname][2] += fn
            gc.collect()

        print(f"[{idx+1}/{len(images)}] {ip.stem[-25:]}", flush=True)
        gc.collect()

    print(f"\n{'='*100}")
    print(f"PAMSR RESULTS — data2_organized val ({len(images)} images)")
    print(f"{'='*100}")
    print(f"{'Method':<28} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}  Rescue-stats")
    print("-" * 100)
    rows = []
    for name, (tp, fp, fn) in agg.items():
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        rows.append((name, tp, fp, fn, p, r, f1))

    for n, tp, fp, fn, p, r, f1 in sorted(rows, key=lambda x: -x[-1]):
        rs = total_stats.get(n, {})
        rs_str = f"pri={rs.get('primary','-')} res={rs.get('rescued','-')} rej={rs.get('rej','-')}" if rs else ""
        marker = " ***" if n not in ("default", "single_d50_cp3") and f1 > 0.7261 else ""
        print(f"{n:<28} {tp:>5} {fp:>5} {fn:>5} {p:>7.4f} {r:>7.4f} {f1:>7.4f}  {rs_str}{marker}")

    best_v3 = max([r for r in rows if r[0] not in ("default", "single_d50_cp3")], key=lambda x: x[-1])
    single = next(r for r in rows if r[0] == "single_d50_cp3")
    print(f"\n*** BEST PAMSR: {best_v3[0]} -> F1={best_v3[-1]:.4f} ***")
    print(f"*** Single-scale baseline: F1={single[-1]:.4f} ***")
    print(f"*** Improvement: {best_v3[-1] - single[-1]:+.4f} ***")


if __name__ == "__main__":
    main()
