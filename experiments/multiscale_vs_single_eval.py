#!/usr/bin/env python3
"""
Multi-scale (PAMSR) vs Single-scale segmentation evaluation.
Datasets: data1_organized (clinical, 853x640) + WBC-Seg (external, 3120x4160)
For each dataset:
  1. Single-scale with best tuned parameters
  2. PAMSR with various secondary scale configs
"""
import sys, gc, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

sys.stdout.reconfigure(line_buffering=True)

# ==================== Config ====================

DATA1_ROOT = Path("/home/xut/csclip/cell_datasets/data1_organized")
WBC_ROOT = Path("/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset")

DATA1_CLASSES = {0: "CCEC", 1: "RBC", 2: "SEC", 3: "Eosinophil",
                 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}

MAX_IMG_EDGE = 2048


# ==================== I/O ====================

def load_image(path, max_edge=None):
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max_edge and max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def load_gt_masks(label_path, img_h, img_w, class_filter=None):
    from skimage.draw import polygon as sk_polygon
    masks = []
    if not label_path.exists():
        return masks
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        if class_filter is not None and cid not in class_filter:
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
    """Match predicted label map against GT masks. Returns (tp, fp, fn, ious)."""
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

    fp = 0
    for pid in range(1, n_pred + 1):
        pm = pred_label_map == pid
        if pm.sum() >= min_area:
            fp += 1
    fp = fp - tp
    fn = len(gt_masks) - tp
    return tp, fp, fn, ious_matched


# ==================== PAMSR ====================

def pamsr(model, img, primary_d, secondary_ds, cellprob=-3.0,
          rescue_prob_thr=0.0, rescue_need_consensus=True,
          min_area=80, overlap_thr=0.2, flow_thr=0.4):
    """Primary-Anchor Multi-Scale Rescue."""
    res_p = model.eval([img], diameter=primary_d,
                       cellprob_threshold=cellprob, flow_threshold=flow_thr)
    masks_p = res_p[0][0]
    flows_p = res_p[1][0]
    prob_map_p = flows_p[2]

    final = masks_p.copy()
    nid = int(final.max()) + 1

    secondary_cells = []
    for sd in secondary_ds:
        res_s = model.eval([img], diameter=sd,
                           cellprob_threshold=cellprob, flow_threshold=flow_thr)
        masks_s = res_s[0][0]
        flows_s = res_s[1][0]
        prob_map_s = flows_s[2]

        for cid in range(1, int(masks_s.max()) + 1):
            cm = masks_s == cid
            area = int(cm.sum())
            if area < min_area:
                continue
            overlap_with_primary = int(np.sum(final[cm] > 0))
            if overlap_with_primary > overlap_thr * area:
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
        overlap_with_final = int(np.sum(final[c["mask"]] > 0))
        if overlap_with_final > overlap_thr * c["area"]:
            stats["rej"] += 1
            continue
        final[c["mask"] & (final == 0)] = nid
        nid += 1
        stats["rescued"] += 1

    return final, stats


# ==================== Eval runner ====================

def eval_configs(model, dataset_name, img_dir, lbl_dir, configs,
                 iou_thr=0.5, class_filter=None, max_edge=None):
    """Evaluate multiple segmentation configs on a dataset."""
    print(f"\n{'='*100}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*100}")

    exts = (".jpg", ".jpeg", ".png")
    all_imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    print(f"Images: {len(all_imgs)}, IoU threshold: {iou_thr}")

    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "ious": []})
    pamsr_stats = defaultdict(lambda: {"primary": 0, "rescued": 0, "rej": 0})

    t0 = time.time()
    n_valid = 0
    for idx, ip in enumerate(all_imgs):
        lp = lbl_dir / (ip.stem + ".txt")
        try:
            img = load_image(str(ip), max_edge=max_edge)
        except Exception as e:
            print(f"  SKIP {ip.name}: {e}", flush=True)
            continue
        h, w = img.shape[:2]
        gt = load_gt_masks(lp, h, w, class_filter=class_filter)
        if not gt:
            continue
        n_valid += 1

        for cname, cfg in configs:
            try:
                if cfg.get("type") == "pamsr":
                    masks, stats = pamsr(model, img, **cfg["params"])
                    tp, fp, fn, ious = match(gt, masks, iou_thr=iou_thr)
                    for k, v in stats.items():
                        pamsr_stats[cname][k] += v
                    del masks
                else:
                    res = model.eval([img], **cfg["params"])
                    masks = res[0][0]
                    tp, fp, fn, ious = match(gt, masks, iou_thr=iou_thr)
                    del masks, res

                agg[cname]["tp"] += tp
                agg[cname]["fp"] += fp
                agg[cname]["fn"] += fn
                agg[cname]["ious"].extend(ious)
            except Exception as e:
                print(f"  ERROR {cname} on {ip.name}: {e}", flush=True)

            gc.collect()

        if (idx + 1) % 20 == 0 or (idx + 1) == len(all_imgs):
            spd = (idx + 1) / (time.time() - t0)
            print(f"  [{idx+1}/{len(all_imgs)}] {spd:.2f} img/s", flush=True)

        del img
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0

    print(f"\n{'─'*110}")
    print(f"RESULTS — {dataset_name} ({n_valid} valid images, {elapsed:.0f}s)")
    print(f"{'─'*110}")
    print(f"{'Method':<42} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} {'mIoU':>7}  Rescue")
    print("-" * 110)

    rows = []
    for cname, _ in configs:
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
                  if ps and ps.get('primary', 0) > 0 else "")
        is_best_ms = any(kw in n.lower() for kw in ["pamsr", "multi"])
        marker = " ★" if is_best_ms and f1 == max(x[6] for x in rows if any(kw in x[0].lower() for kw in ["pamsr", "multi"])) else ""
        print(f"{n:<42} {tp:>5} {fp:>5} {fn:>5} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {miou:>7.4f}  {ps_str}{marker}")

    single_best = max([r for r in rows if "pamsr" not in r[0].lower() and "multi" not in r[0].lower()],
                      key=lambda x: x[6])
    ms_best = max([r for r in rows if "pamsr" in r[0].lower() or "multi" in r[0].lower()],
                  key=lambda x: x[6], default=None)

    print(f"\n  Best single-scale: {single_best[0]} → F1={single_best[6]:.4f}")
    if ms_best:
        print(f"  Best multi-scale:  {ms_best[0]} → F1={ms_best[6]:.4f}")
        imp = ms_best[6] - single_best[6]
        print(f"  PAMSR improvement: ΔF1 = {imp:+.4f} ({imp/max(single_best[6],1e-9)*100:+.1f}%)")
        print(f"  Precision: {single_best[4]:.4f} → {ms_best[4]:.4f} ({ms_best[4]-single_best[4]:+.4f})")
        print(f"  Recall:    {single_best[5]:.4f} → {ms_best[5]:.4f} ({ms_best[5]-single_best[5]:+.4f})")

    return rows


def main():
    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    print("=" * 110)
    print("MULTI-SCALE (PAMSR) vs SINGLE-SCALE SEGMENTATION COMPARISON")
    print("=" * 110, flush=True)

    # ==================== data1_organized ====================
    # Best single-scale from sweep: d=60, cp=-2, ft=0.3
    data1_configs = [
        ("Single d=60 cp=-2 ft=0.3 (best)", {
            "type": "single",
            "params": {"diameter": 60.0, "cellprob_threshold": -2.0, "flow_threshold": 0.3}
        }),
        ("Single d=50 cp=-3 ft=0.3", {
            "type": "single",
            "params": {"diameter": 50.0, "cellprob_threshold": -3.0, "flow_threshold": 0.3}
        }),
        ("Single d=60 cp=-3", {
            "type": "single",
            "params": {"diameter": 60.0, "cellprob_threshold": -3.0}
        }),
        # PAMSR: primary=60, sweep secondary scales
        ("PAMSR p60 s[45,80] cp-2 ft0.3 cons", {
            "type": "pamsr",
            "params": {"primary_d": 60, "secondary_ds": [45, 80],
                       "cellprob": -2.0, "flow_thr": 0.3,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": True}
        }),
        ("PAMSR p60 s[40,80] cp-2 ft0.3 cons", {
            "type": "pamsr",
            "params": {"primary_d": 60, "secondary_ds": [40, 80],
                       "cellprob": -2.0, "flow_thr": 0.3,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": True}
        }),
        ("PAMSR p60 s[45,80] cp-2 ft0.3 nocons", {
            "type": "pamsr",
            "params": {"primary_d": 60, "secondary_ds": [45, 80],
                       "cellprob": -2.0, "flow_thr": 0.3,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": False}
        }),
        ("PAMSR p60 s[45,80] cp-2 ft0.3 p1", {
            "type": "pamsr",
            "params": {"primary_d": 60, "secondary_ds": [45, 80],
                       "cellprob": -2.0, "flow_thr": 0.3,
                       "rescue_prob_thr": 1.0, "rescue_need_consensus": True}
        }),
        ("PAMSR p60 s[40,75,90] cp-2 ft0.3", {
            "type": "pamsr",
            "params": {"primary_d": 60, "secondary_ds": [40, 75, 90],
                       "cellprob": -2.0, "flow_thr": 0.3,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": True}
        }),
        ("PAMSR p60 s[45,80] cp-2 ft0.3 ov0.1", {
            "type": "pamsr",
            "params": {"primary_d": 60, "secondary_ds": [45, 80],
                       "cellprob": -2.0, "flow_thr": 0.3,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": True,
                       "overlap_thr": 0.1}
        }),
        ("PAMSR p60 s[45,80] cp-2 ft0.3 p-1", {
            "type": "pamsr",
            "params": {"primary_d": 60, "secondary_ds": [45, 80],
                       "cellprob": -2.0, "flow_thr": 0.3,
                       "rescue_prob_thr": -1.0, "rescue_need_consensus": True}
        }),
    ]

    data1_rows = eval_configs(
        model, "data1_organized (clinical BALF, 7 classes)",
        DATA1_ROOT / "images" / "val",
        DATA1_ROOT / "labels_polygon" / "val",
        data1_configs,
        iou_thr=0.5, class_filter=None, max_edge=None
    )

    # ==================== WBC-Seg ====================
    # From EXPERIMENT_RESULTS_SUMMARY: d=30, cp=-2, IoU>=0.5
    wbc_configs = [
        ("Single d=30 cp=-2 (reported best)", {
            "type": "single",
            "params": {"diameter": 30.0, "cellprob_threshold": -2.0}
        }),
        ("Single d=30 cp=-3", {
            "type": "single",
            "params": {"diameter": 30.0, "cellprob_threshold": -3.0}
        }),
        ("Single d=25 cp=-2", {
            "type": "single",
            "params": {"diameter": 25.0, "cellprob_threshold": -2.0}
        }),
        # PAMSR: primary=30
        ("PAMSR p30 s[20,45] cp-2 cons", {
            "type": "pamsr",
            "params": {"primary_d": 30, "secondary_ds": [20, 45],
                       "cellprob": -2.0,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": True}
        }),
        ("PAMSR p30 s[20,40] cp-2 cons", {
            "type": "pamsr",
            "params": {"primary_d": 30, "secondary_ds": [20, 40],
                       "cellprob": -2.0,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": True}
        }),
        ("PAMSR p30 s[20,45] cp-2 nocons", {
            "type": "pamsr",
            "params": {"primary_d": 30, "secondary_ds": [20, 45],
                       "cellprob": -2.0,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": False}
        }),
        ("PAMSR p30 s[20,45] cp-2 p1", {
            "type": "pamsr",
            "params": {"primary_d": 30, "secondary_ds": [20, 45],
                       "cellprob": -2.0,
                       "rescue_prob_thr": 1.0, "rescue_need_consensus": True}
        }),
        ("PAMSR p30 s[15,40,55] cp-2 cons", {
            "type": "pamsr",
            "params": {"primary_d": 30, "secondary_ds": [15, 40, 55],
                       "cellprob": -2.0,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": True}
        }),
        ("PAMSR p30 s[20,45] cp-2 ov0.1", {
            "type": "pamsr",
            "params": {"primary_d": 30, "secondary_ds": [20, 45],
                       "cellprob": -2.0,
                       "rescue_prob_thr": 0.0, "rescue_need_consensus": True,
                       "overlap_thr": 0.1}
        }),
        ("PAMSR p30 s[20,45] cp-2 p-1", {
            "type": "pamsr",
            "params": {"primary_d": 30, "secondary_ds": [20, 45],
                       "cellprob": -2.0,
                       "rescue_prob_thr": -1.0, "rescue_need_consensus": True}
        }),
    ]

    wbc_rows = eval_configs(
        model, "WBC-Seg (external, high-res 3120x4160)",
        WBC_ROOT / "images" / "val",
        WBC_ROOT / "labels" / "val",
        wbc_configs,
        iou_thr=0.5, class_filter=None, max_edge=MAX_IMG_EDGE
    )

    # ==================== Cross-dataset summary ====================
    print(f"\n\n{'='*110}")
    print("CROSS-DATASET SUMMARY: Multi-Scale (PAMSR) vs Single-Scale")
    print(f"{'='*110}")
    print(f"{'Dataset':<35} {'Best Single':<20} {'F1':>7} {'Best PAMSR':<35} {'F1':>7} {'ΔF1':>8}")
    print("-" * 110)

    for dname, rows in [("data1_organized", data1_rows), ("WBC-Seg", wbc_rows)]:
        single_rows = [r for r in rows if "pamsr" not in r[0].lower()]
        ms_rows = [r for r in rows if "pamsr" in r[0].lower()]
        best_s = max(single_rows, key=lambda x: x[6])
        best_m = max(ms_rows, key=lambda x: x[6]) if ms_rows else None
        if best_m:
            imp = best_m[6] - best_s[6]
            print(f"{dname:<35} {best_s[0][:20]:<20} {best_s[6]:>7.4f} "
                  f"{best_m[0][:35]:<35} {best_m[6]:>7.4f} {imp:>+8.4f}")
        else:
            print(f"{dname:<35} {best_s[0][:20]:<20} {best_s[6]:>7.4f}  N/A")


if __name__ == "__main__":
    main()
