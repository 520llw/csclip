#!/usr/bin/env python3
"""
MSCPF v2: Multi-Scale Consensus with Quality-Aware Filtering
Key improvements over v1:
- Composite Quality Score (CQS) for ALL detections including consensus
- Best-mask selection when merging across scales
- Circularity-based shape regularization to reject artifact FPs
- Adaptive score thresholding per image
"""
import sys, gc, itertools
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


def compute_circularity(mask):
    """4*pi*area / perimeter^2. Perfect circle = 1.0."""
    from skimage.measure import find_contours
    area = mask.sum()
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return 0.0
    c = max(contours, key=len)
    perimeter = np.sum(np.sqrt(np.sum(np.diff(c, axis=0)**2, axis=1)))
    if perimeter < 1e-6:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def mscpf_v2(model, img, diameters=[40, 50, 65], cellprob=-3.0,
             score_thr=0.4, circ_thr=0.3, min_area=80, max_area_ratio=0.01,
             w_cons=0.35, w_prob=0.30, w_flow=0.20, w_circ=0.15):
    """
    MSCPF v2: All detections scored by Composite Quality Score.
    CQS = w_cons * consensus_norm + w_prob * prob_norm + w_flow * flow_norm + w_circ * circ_norm
    """
    h, w = img.shape[:2]
    max_area = h * w * max_area_ratio
    scale_cells = []

    for d in diameters:
        masks, flows, _ = model.eval(img, diameter=d, cellprob_threshold=cellprob, channels=[0, 0])
        prob_map = flows[2]
        flow_field = flows[1]

        for cid in range(1, masks.max() + 1):
            cm = masks == cid
            area = cm.sum()
            if area < min_area or area > max_area:
                continue

            ys, xs = np.where(cm)
            cy, cx = ys.mean(), xs.mean()

            mp = float(prob_map[cm].mean())

            fy, fx = flow_field[0][cm], flow_field[1][cm]
            dy = cy - ys.astype(float)
            dx = cx - xs.astype(float)
            norms = np.sqrt(dy**2 + dx**2) + 1e-8
            fn = np.sqrt(fy**2 + fx**2) + 1e-8
            rc = float(np.mean((fy/fn) * (dy/norms) + (fx/fn) * (dx/norms)))

            circ = compute_circularity(cm)

            scale_cells.append({
                "mask": cm, "mp": mp, "rc": rc, "circ": circ,
                "area": area, "d": d, "cx": cx, "cy": cy
            })

        del masks, flows
        gc.collect()

    groups = []
    assigned = [False] * len(scale_cells)

    for i, c1 in enumerate(scale_cells):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j, c2 in enumerate(scale_cells):
            if assigned[j] or c1["d"] == c2["d"]:
                continue
            inter = np.logical_and(c1["mask"], c2["mask"]).sum()
            union = np.logical_or(c1["mask"], c2["mask"]).sum()
            if union > 0 and inter / union > 0.3:
                group.append(j)
                assigned[j] = True
        groups.append(group)

    candidates = []
    for grp in groups:
        n_scales = len(set(scale_cells[i]["d"] for i in grp))
        best_idx = max(grp, key=lambda i: scale_cells[i]["mp"])
        bc = scale_cells[best_idx]
        bc["n_scales"] = n_scales
        candidates.append(bc)

    if not candidates:
        return np.zeros((h, w), dtype=np.int32), {"kept": 0, "rej": 0}

    cons_vals = np.array([c["n_scales"] for c in candidates], dtype=float)
    prob_vals = np.array([c["mp"] for c in candidates])
    flow_vals = np.array([max(0.0, c["rc"]) for c in candidates])
    circ_vals = np.array([c["circ"] for c in candidates])

    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-10)

    cn = norm01(cons_vals)
    pn = norm01(prob_vals)
    fn = norm01(flow_vals)
    crn = norm01(circ_vals)

    cqs = w_cons * cn + w_prob * pn + w_flow * fn + w_circ * crn

    final = np.zeros((h, w), dtype=np.int32)
    nid = 1
    stats = {"kept": 0, "rej": 0}

    order = np.argsort(-cqs)
    for idx in order:
        c = candidates[idx]
        if c["circ"] < circ_thr:
            stats["rej"] += 1
            continue
        if cqs[idx] < score_thr:
            stats["rej"] += 1
            continue
        overlap = final[c["mask"]]
        if np.sum(overlap > 0) > 0.3 * c["area"]:
            continue
        final[c["mask"] & (final == 0)] = nid
        nid += 1
        stats["kept"] += 1

    return final, stats


def main():
    from cellpose import models
    model = models.CellposeModel(gpu=True)

    img_dir = DATA_ROOT / "images" / "val"
    lbl_dir = DATA_ROOT / "labels_polygon" / "val"
    images = sorted(img_dir.glob("*.png"))

    configs = [
        ("default",           {}),
        ("single_d50_cp3",    {}),
        # MSCPF v2 variants
        ("v2_base",           {"diameters": [40, 50, 65], "score_thr": 0.35, "circ_thr": 0.25}),
        ("v2_strict",         {"diameters": [40, 50, 65], "score_thr": 0.45, "circ_thr": 0.35}),
        ("v2_prob_heavy",     {"diameters": [40, 50, 65], "score_thr": 0.35, "circ_thr": 0.25,
                               "w_cons": 0.25, "w_prob": 0.40, "w_flow": 0.20, "w_circ": 0.15}),
        ("v2_wide_scale",     {"diameters": [35, 50, 70], "score_thr": 0.35, "circ_thr": 0.25}),
        ("v2_4scale",         {"diameters": [35, 45, 55, 70], "score_thr": 0.35, "circ_thr": 0.25}),
        ("v2_strict_circ",    {"diameters": [40, 50, 65], "score_thr": 0.35, "circ_thr": 0.40}),
    ]

    agg = defaultdict(lambda: [0, 0, 0])

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
                masks, stats = mscpf_v2(model, img, **params)
                tp, fp, fn = match(gt, masks)
                del masks

            agg[cname][0] += tp
            agg[cname][1] += fp
            agg[cname][2] += fn
            gc.collect()

        print(f"[{idx+1}/{len(images)}] {ip.stem[-25:]}", flush=True)
        gc.collect()

    print(f"\n{'='*100}")
    print(f"MSCPF v2 RESULTS — data2_organized val ({len(images)} images)")
    print(f"{'='*100}")
    print(f"{'Method':<25} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 75)
    rows = []
    for name, (tp, fp, fn) in agg.items():
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        rows.append((name, tp, fp, fn, p, r, f1))
    for n, tp, fp, fn, p, r, f1 in sorted(rows, key=lambda x: -x[-1]):
        marker = " ***" if n not in ("default", "single_d50_cp3") and f1 > 0.7261 else ""
        print(f"{n:<25} {tp:>5} {fp:>5} {fn:>5} {p:>7.4f} {r:>7.4f} {f1:>7.4f}{marker}")

    best_v2 = max([r for r in rows if r[0] not in ("default", "single_d50_cp3")], key=lambda x: x[-1])
    print(f"\n*** BEST v2: {best_v2[0]} -> F1={best_v2[-1]:.4f} ***")
    single = next(r for r in rows if r[0] == "single_d50_cp3")
    print(f"*** Single-scale baseline: F1={single[-1]:.4f} ***")
    print(f"*** Improvement: {best_v2[-1] - single[-1]:+.4f} ***")


if __name__ == "__main__":
    main()
