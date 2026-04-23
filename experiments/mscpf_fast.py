#!/usr/bin/env python3
"""
MSCPF Fast Evaluation: Optimized for speed.
- Reuse model across scales (don't recreate each time)
- Fewer configs, focused on promising ones
- Process images sequentially with immediate output
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


def mscpf(model, img, diameters=[40, 50, 65], cellprob=-3.0, prob_thr=-1.0, flow_thr=0.5):
    """Multi-Scale Consensus + Probability + Flow filtering."""
    h, w = img.shape[:2]
    scale_cells = []

    for d in diameters:
        masks, flows, _ = model.eval(img, diameter=d, cellprob_threshold=cellprob, channels=[0, 0])
        prob_map = flows[2]
        flow_field = flows[1]

        for cid in range(1, masks.max() + 1):
            cm = masks == cid
            if cm.sum() < 50:
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

            scale_cells.append({"mask": cm, "mp": mp, "rc": rc, "d": d})

        del masks, flows
        gc.collect()

    for i, c1 in enumerate(scale_cells):
        cnt = 1
        for j, c2 in enumerate(scale_cells):
            if i == j or c1["d"] == c2["d"]:
                continue
            inter = np.logical_and(c1["mask"], c2["mask"]).sum()
            union = np.logical_or(c1["mask"], c2["mask"]).sum()
            if union > 0 and inter / union > 0.3:
                cnt += 1
                break
        c1["consensus"] = cnt

    final = np.zeros((h, w), dtype=np.int32)
    nid = 1
    stats = {"cons": 0, "prob": 0, "rej": 0}

    for c in sorted(scale_cells, key=lambda x: (-x["consensus"], -x["mp"])):
        overlap = final[c["mask"]]
        if overlap.any() and np.sum(overlap > 0) > 0.3 * c["mask"].sum():
            continue
        if c["consensus"] >= 2:
            final[c["mask"] & (final == 0)] = nid
            nid += 1
            stats["cons"] += 1
        elif c["mp"] > prob_thr and c["rc"] > flow_thr:
            final[c["mask"] & (final == 0)] = nid
            nid += 1
            stats["prob"] += 1
        else:
            stats["rej"] += 1

    return final, stats


def main():
    from cellpose import models
    model = models.CellposeModel(gpu=True)

    img_dir = DATA_ROOT / "images" / "val"
    lbl_dir = DATA_ROOT / "labels_polygon" / "val"
    images = sorted(img_dir.glob("*.png"))

    agg = defaultdict(lambda: [0, 0, 0])

    for idx, ip in enumerate(images):
        lp = lbl_dir / (ip.stem + ".txt")
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        gt = load_gt(lp, h, w)
        if not gt:
            continue

        # Default
        masks, _, _ = model.eval(img, channels=[0, 0])
        tp, fp, fn = match(gt, masks)
        agg["default"][0] += tp
        agg["default"][1] += fp
        agg["default"][2] += fn

        # Single-scale optimized
        masks, _, _ = model.eval(img, diameter=50, cellprob_threshold=-3.0, channels=[0, 0])
        tp, fp, fn = match(gt, masks)
        agg["single_d50_cp3"][0] += tp
        agg["single_d50_cp3"][1] += fp
        agg["single_d50_cp3"][2] += fn

        # MSCPF variants
        for name, diams, pt, ft in [
            ("mscpf_40_50_65",       [40, 50, 65], -1.0, 0.5),
            ("mscpf_40_50_65_pt-2",  [40, 50, 65], -2.0, 0.5),
            ("mscpf_40_50_65_ft0.3", [40, 50, 65], -1.0, 0.3),
            ("mscpf_35_50_70",       [35, 50, 70], -1.0, 0.5),
        ]:
            masks, stats = mscpf(model, img, diams, prob_thr=pt, flow_thr=ft)
            tp, fp, fn = match(gt, masks)
            agg[name][0] += tp
            agg[name][1] += fp
            agg[name][2] += fn
            del masks
            gc.collect()

        print(f"[{idx+1}/{len(images)}] {ip.stem[-25:]}", flush=True)
        gc.collect()

    print(f"\n{'='*100}")
    print(f"MSCPF RESULTS — data2_organized val ({len(images)} images)")
    print(f"{'='*100}")
    print(f"{'Method':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-"*80)
    rows = []
    for name, (tp, fp, fn) in agg.items():
        p = tp/(tp+fp) if tp+fp else 0
        r = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        rows.append((name, tp, fp, fn, p, r, f1))
    for n, tp, fp, fn, p, r, f1 in sorted(rows, key=lambda x: -x[-1]):
        print(f"{n:<30} {tp:>5} {fp:>5} {fn:>5} {p:>7.4f} {r:>7.4f} {f1:>7.4f}")

    best = max(rows, key=lambda x: x[-1])
    print(f"\n*** BEST: {best[0]} -> F1={best[-1]:.4f} ***")


if __name__ == "__main__":
    main()
