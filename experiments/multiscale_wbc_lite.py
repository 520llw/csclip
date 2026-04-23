#!/usr/bin/env python3
"""
WBC-Seg: PAMSR vs Single-Scale (lightweight version).
- Resize to max 1536px to reduce memory
- Only 4 configs: 2 single + 2 PAMSR
- Aggressive memory cleanup per image
"""
import sys, gc, time, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageOps

sys.stdout.reconfigure(line_buffering=True)

WBC_ROOT = Path("/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset")
MAX_IMG_EDGE = 1536


def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMG_EDGE:
        scale = MAX_IMG_EDGE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def load_gt_masks(label_path, img_h, img_w):
    from skimage.draw import polygon as sk_polygon
    masks = []
    if not label_path.exists():
        return masks
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        pts = [float(x) for x in parts[1:]]
        xs = [pts[i] * img_w for i in range(0, len(pts), 2)]
        ys = [pts[i] * img_h for i in range(1, len(pts), 2)]
        rr, cc = sk_polygon(ys, xs, shape=(img_h, img_w))
        if len(rr) == 0:
            continue
        mask = np.zeros((img_h, img_w), dtype=bool)
        mask[rr, cc] = True
        masks.append(mask)
    return masks


def match_fast(gt_masks, pred_label_map, iou_thr=0.5, min_area=30):
    """Faster matching: only check pred cells overlapping GT bboxes."""
    tp, matched_gt = 0, set()
    ious = []
    n_pred = int(pred_label_map.max()) if pred_label_map.max() > 0 else 0
    n_valid_pred = 0

    for pid in range(1, n_pred + 1):
        pm = pred_label_map == pid
        if pm.sum() < min_area:
            continue
        n_valid_pred += 1
        best, bi = 0.0, -1
        for gi, gm in enumerate(gt_masks):
            if gi in matched_gt:
                continue
            inter = np.logical_and(pm, gm).sum()
            if inter == 0:
                continue
            union = np.logical_or(pm, gm).sum()
            iou = inter / union if union > 0 else 0
            if iou > best:
                best, bi = iou, gi
        if best >= iou_thr and bi >= 0:
            tp += 1
            matched_gt.add(bi)
            ious.append(best)

    fp = n_valid_pred - tp
    fn = len(gt_masks) - tp
    return tp, fp, fn, ious


def run_single(model, img, diameter, cellprob):
    res = model.eval([img], diameter=diameter, cellprob_threshold=cellprob)
    masks = res[0][0]
    del res
    return masks


def run_pamsr(model, img, primary_d, secondary_ds, cellprob=-2.0,
              rescue_prob_thr=0.0, consensus=True, min_area=80, overlap_thr=0.2):
    res_p = model.eval([img], diameter=primary_d, cellprob_threshold=cellprob)
    masks_p = res_p[0][0]
    prob_p = res_p[1][0][2]
    final = masks_p.copy()
    nid = int(final.max()) + 1
    del res_p, masks_p
    gc.collect()

    secondary_cells = []
    for sd in secondary_ds:
        res_s = model.eval([img], diameter=sd, cellprob_threshold=cellprob)
        masks_s = res_s[0][0]
        prob_s = res_s[1][0][2]
        for cid in range(1, int(masks_s.max()) + 1):
            cm = masks_s == cid
            area = int(cm.sum())
            if area < min_area:
                continue
            overlap = int(np.sum(final[cm] > 0))
            if overlap > overlap_thr * area:
                continue
            mp = float(prob_s[cm].mean())
            secondary_cells.append({"mask": cm.copy(), "mp": mp, "d": sd, "area": area})
        del masks_s, prob_s, res_s
        gc.collect()

    if consensus and len(secondary_ds) > 1:
        for i, c1 in enumerate(secondary_cells):
            c1["ok"] = False
            for j, c2 in enumerate(secondary_cells):
                if i == j or c1["d"] == c2["d"]:
                    continue
                inter = np.logical_and(c1["mask"], c2["mask"]).sum()
                union = np.logical_or(c1["mask"], c2["mask"]).sum()
                if union > 0 and inter / union > 0.3:
                    c1["ok"] = True
                    break
    else:
        for c in secondary_cells:
            c["ok"] = True

    rescued = 0
    for c in sorted(secondary_cells, key=lambda x: -x["mp"]):
        if c["mp"] <= rescue_prob_thr:
            continue
        if consensus and not c["ok"]:
            continue
        overlap = int(np.sum(final[c["mask"]] > 0))
        if overlap > overlap_thr * c["area"]:
            continue
        final[c["mask"] & (final == 0)] = nid
        nid += 1
        rescued += 1

    del secondary_cells
    gc.collect()
    return final, {"primary": nid - 1 - rescued, "rescued": rescued}


def main():
    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    print("=" * 100)
    print("WBC-Seg: PAMSR vs Single-Scale (lite)")
    print(f"Max image edge: {MAX_IMG_EDGE}px")
    print("=" * 100, flush=True)

    img_dir = WBC_ROOT / "images" / "val"
    lbl_dir = WBC_ROOT / "labels" / "val"
    all_imgs = sorted([p for p in img_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"Images: {len(all_imgs)}")

    agg = {
        "Single d=30 cp=-2": {"tp": 0, "fp": 0, "fn": 0, "ious": []},
        "Single d=30 cp=-3": {"tp": 0, "fp": 0, "fn": 0, "ious": []},
        "PAMSR p30 s[20,45] cons": {"tp": 0, "fp": 0, "fn": 0, "ious": [], "pri": 0, "res": 0},
        "PAMSR p30 s[20,45] nocons": {"tp": 0, "fp": 0, "fn": 0, "ious": [], "pri": 0, "res": 0},
    }

    t0 = time.time()
    n_valid = 0

    for idx, ip in enumerate(all_imgs):
        lp = lbl_dir / (ip.stem + ".txt")
        try:
            img = load_image(str(ip))
        except Exception as e:
            print(f"  SKIP {ip.name}: {e}", flush=True)
            continue
        h, w = img.shape[:2]
        gt = load_gt_masks(lp, h, w)
        if not gt:
            del img; gc.collect()
            continue
        n_valid += 1

        # Single d=30 cp=-2
        masks = run_single(model, img, 30.0, -2.0)
        tp, fp, fn, ious = match_fast(gt, masks)
        agg["Single d=30 cp=-2"]["tp"] += tp
        agg["Single d=30 cp=-2"]["fp"] += fp
        agg["Single d=30 cp=-2"]["fn"] += fn
        agg["Single d=30 cp=-2"]["ious"].extend(ious)
        del masks; gc.collect()

        # Single d=30 cp=-3
        masks = run_single(model, img, 30.0, -3.0)
        tp, fp, fn, ious = match_fast(gt, masks)
        agg["Single d=30 cp=-3"]["tp"] += tp
        agg["Single d=30 cp=-3"]["fp"] += fp
        agg["Single d=30 cp=-3"]["fn"] += fn
        agg["Single d=30 cp=-3"]["ious"].extend(ious)
        del masks; gc.collect()

        # PAMSR cons
        masks, stats = run_pamsr(model, img, 30, [20, 45], cellprob=-2.0, consensus=True)
        tp, fp, fn, ious = match_fast(gt, masks)
        agg["PAMSR p30 s[20,45] cons"]["tp"] += tp
        agg["PAMSR p30 s[20,45] cons"]["fp"] += fp
        agg["PAMSR p30 s[20,45] cons"]["fn"] += fn
        agg["PAMSR p30 s[20,45] cons"]["ious"].extend(ious)
        agg["PAMSR p30 s[20,45] cons"]["pri"] += stats["primary"]
        agg["PAMSR p30 s[20,45] cons"]["res"] += stats["rescued"]
        del masks; gc.collect()

        # PAMSR nocons
        masks, stats = run_pamsr(model, img, 30, [20, 45], cellprob=-2.0, consensus=False)
        tp, fp, fn, ious = match_fast(gt, masks)
        agg["PAMSR p30 s[20,45] nocons"]["tp"] += tp
        agg["PAMSR p30 s[20,45] nocons"]["fp"] += fp
        agg["PAMSR p30 s[20,45] nocons"]["fn"] += fn
        agg["PAMSR p30 s[20,45] nocons"]["ious"].extend(ious)
        agg["PAMSR p30 s[20,45] nocons"]["pri"] += stats["primary"]
        agg["PAMSR p30 s[20,45] nocons"]["res"] += stats["rescued"]
        del masks; gc.collect()

        del img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0 or (idx + 1) == len(all_imgs):
            elapsed = time.time() - t0
            spd = (idx + 1) / elapsed
            eta = (len(all_imgs) - idx - 1) / spd if spd > 0 else 0
            print(f"  [{idx+1}/{len(all_imgs)}] {spd:.3f} img/s, ETA {eta/60:.0f}min", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'─'*100}")
    print(f"RESULTS — WBC-Seg ({n_valid} valid images, {elapsed:.0f}s)")
    print(f"{'─'*100}")
    print(f"{'Method':<35} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'mIoU':>7}  Rescue")
    print("-" * 100)

    rows = []
    for name in ["Single d=30 cp=-2", "Single d=30 cp=-3",
                  "PAMSR p30 s[20,45] cons", "PAMSR p30 s[20,45] nocons"]:
        d = agg[name]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        miou = float(np.mean(d["ious"])) if d["ious"] else 0
        rows.append((name, tp, fp, fn, p, r, f1, miou))
        ps = f"pri={d.get('pri','')} res={d.get('res','')}" if "PAMSR" in name else ""
        print(f"{name:<35} {tp:>6} {fp:>6} {fn:>6} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {miou:>7.4f}  {ps}")

    single_best = max([r for r in rows if "PAMSR" not in r[0]], key=lambda x: x[6])
    ms_best = max([r for r in rows if "PAMSR" in r[0]], key=lambda x: x[6])
    imp = ms_best[6] - single_best[6]
    print(f"\n  Best single: {single_best[0]} → F1={single_best[6]:.4f}")
    print(f"  Best PAMSR:  {ms_best[0]} → F1={ms_best[6]:.4f}")
    print(f"  Δ F1 = {imp:+.4f} ({imp/max(single_best[6],1e-9)*100:+.1f}%)")
    print(f"  Precision: {single_best[4]:.4f} → {ms_best[4]:.4f} ({ms_best[4]-single_best[4]:+.4f})")
    print(f"  Recall:    {single_best[5]:.4f} → {ms_best[5]:.4f} ({ms_best[5]-single_best[5]:+.4f})")


if __name__ == "__main__":
    main()
