#!/usr/bin/env python3
"""
Multi-scale (PAMSR) vs Single-scale on WBC-Seg only.
Reduced configs for speed: 3 single + 4 PAMSR = 7 configs.
Images resized to max 2048px to avoid OOM.
"""
import sys, gc, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

sys.stdout.reconfigure(line_buffering=True)

WBC_ROOT = Path("/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset")
MAX_IMG_EDGE = 2048


def load_image(path, max_edge=MAX_IMG_EDGE):
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max_edge and max(w, h) > max_edge:
        scale = max_edge / max(w, h)
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


def match(gt_masks, pred_label_map, iou_thr=0.5, min_area=30):
    tp, matched_gt = 0, set()
    n_pred = int(pred_label_map.max()) if pred_label_map.max() > 0 else 0
    ious_matched = []
    for pid in range(1, n_pred + 1):
        pm = pred_label_map == pid
        if pm.sum() < min_area:
            continue
        best, bi = 0.0, -1
        for gi, gm in enumerate(gt_masks):
            if gi in matched_gt:
                continue
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou = inter / union if union > 0 else 0
            if iou > best:
                best, bi = iou, gi
        if best >= iou_thr and bi >= 0:
            tp += 1
            matched_gt.add(bi)
            ious_matched.append(best)
    fp = sum(1 for pid in range(1, n_pred + 1) if (pred_label_map == pid).sum() >= min_area) - tp
    fn = len(gt_masks) - tp
    return tp, fp, fn, ious_matched


def pamsr(model, img, primary_d, secondary_ds, cellprob=-2.0,
          rescue_prob_thr=0.0, rescue_need_consensus=True,
          min_area=80, overlap_thr=0.2):
    res_p = model.eval([img], diameter=primary_d, cellprob_threshold=cellprob)
    masks_p = res_p[0][0]
    flows_p = res_p[1][0]
    prob_map_p = flows_p[2]
    final = masks_p.copy()
    nid = int(final.max()) + 1

    secondary_cells = []
    for sd in secondary_ds:
        res_s = model.eval([img], diameter=sd, cellprob_threshold=cellprob)
        masks_s = res_s[0][0]
        flows_s = res_s[1][0]
        prob_map_s = flows_s[2]
        for cid in range(1, int(masks_s.max()) + 1):
            cm = masks_s == cid
            area = int(cm.sum())
            if area < min_area:
                continue
            overlap = int(np.sum(final[cm] > 0))
            if overlap > overlap_thr * area:
                continue
            mp = float(prob_map_s[cm].mean())
            secondary_cells.append({"mask": cm, "mp": mp, "d": sd, "area": area})
        del masks_s, flows_s, res_s
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
        overlap = int(np.sum(final[c["mask"]] > 0))
        if overlap > overlap_thr * c["area"]:
            stats["rej"] += 1
            continue
        final[c["mask"] & (final == 0)] = nid
        nid += 1
        stats["rescued"] += 1
    return final, stats


def main():
    import torch
    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    print("=" * 110)
    print("WBC-Seg: PAMSR Multi-Scale vs Single-Scale")
    print("=" * 110, flush=True)

    img_dir = WBC_ROOT / "images" / "val"
    lbl_dir = WBC_ROOT / "labels" / "val"
    exts = (".jpg", ".jpeg", ".png")
    all_imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    print(f"Images: {len(all_imgs)}")

    configs = [
        ("Single d=30 cp=-2 (reported best)", "single", {"diameter": 30.0, "cellprob_threshold": -2.0}),
        ("Single d=30 cp=-3", "single", {"diameter": 30.0, "cellprob_threshold": -3.0}),
        ("Single d=25 cp=-2", "single", {"diameter": 25.0, "cellprob_threshold": -2.0}),
        ("PAMSR p30 s[20,45] cons", "pamsr",
         {"primary_d": 30, "secondary_ds": [20, 45], "cellprob": -2.0,
          "rescue_prob_thr": 0.0, "rescue_need_consensus": True}),
        ("PAMSR p30 s[20,45] nocons", "pamsr",
         {"primary_d": 30, "secondary_ds": [20, 45], "cellprob": -2.0,
          "rescue_prob_thr": 0.0, "rescue_need_consensus": False}),
        ("PAMSR p30 s[20,45] p1 cons", "pamsr",
         {"primary_d": 30, "secondary_ds": [20, 45], "cellprob": -2.0,
          "rescue_prob_thr": 1.0, "rescue_need_consensus": True}),
        ("PAMSR p30 s[15,40,55] cons", "pamsr",
         {"primary_d": 30, "secondary_ds": [15, 40, 55], "cellprob": -2.0,
          "rescue_prob_thr": 0.0, "rescue_need_consensus": True}),
    ]

    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "ious": []})
    pamsr_stats = defaultdict(lambda: {"primary": 0, "rescued": 0, "rej": 0})

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

        for cname, ctype, params in configs:
            try:
                if ctype == "pamsr":
                    masks, stats = pamsr(model, img, **params)
                    tp, fp, fn, ious = match(gt, masks)
                    for k, v in stats.items():
                        pamsr_stats[cname][k] += v
                    del masks
                else:
                    res = model.eval([img], **params)
                    masks = res[0][0]
                    tp, fp, fn, ious = match(gt, masks)
                    del masks, res
                agg[cname]["tp"] += tp
                agg[cname]["fp"] += fp
                agg[cname]["fn"] += fn
                agg[cname]["ious"].extend(ious)
            except Exception as e:
                print(f"  ERR {cname} {ip.name}: {e}", flush=True)
            gc.collect()

        del img; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0 or (idx + 1) == len(all_imgs):
            spd = (idx + 1) / (time.time() - t0)
            print(f"  [{idx+1}/{len(all_imgs)}] {spd:.2f} img/s", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'─'*110}")
    print(f"RESULTS — WBC-Seg ({n_valid} valid images, {elapsed:.0f}s)")
    print(f"{'─'*110}")
    print(f"{'Method':<42} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'mIoU':>7}  Rescue")
    print("-" * 110)

    rows = []
    for cname, _, _ in configs:
        d = agg[cname]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        miou = float(np.mean(d["ious"])) if d["ious"] else 0
        rows.append((cname, tp, fp, fn, p, r, f1, miou))

    for n, tp, fp, fn, p, r, f1, miou in sorted(rows, key=lambda x: -x[6]):
        ps = pamsr_stats.get(n, {})
        ps_str = (f"pri={ps['primary']} res={ps['rescued']} rej={ps['rej']}"
                  if ps.get('primary', 0) > 0 else "")
        print(f"{n:<42} {tp:>6} {fp:>6} {fn:>6} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {miou:>7.4f}  {ps_str}")

    single_best = max([r for r in rows if "pamsr" not in r[0].lower()], key=lambda x: x[6])
    ms_best = max([r for r in rows if "pamsr" in r[0].lower()], key=lambda x: x[6])
    imp = ms_best[6] - single_best[6]
    print(f"\n  Best single-scale: {single_best[0]} → F1={single_best[6]:.4f}")
    print(f"  Best multi-scale:  {ms_best[0]} → F1={ms_best[6]:.4f}")
    print(f"  PAMSR improvement: ΔF1 = {imp:+.4f} ({imp/max(single_best[6],1e-9)*100:+.1f}%)")
    print(f"  Precision: {single_best[4]:.4f} → {ms_best[4]:.4f} ({ms_best[4]-single_best[4]:+.4f})")
    print(f"  Recall:    {single_best[5]:.4f} → {ms_best[5]:.4f} ({ms_best[5]-single_best[5]:+.4f})")


if __name__ == "__main__":
    main()
