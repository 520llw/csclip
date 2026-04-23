#!/usr/bin/env python3
"""
CellposeSAM v3: Test lower cellprob thresholds + FP filtering strategies.
Since cpm3 (cellprob=-3.0) was best, test -4.0 and -5.0.
Also test post-processing FP filtering: area, circularity, intensity.
"""
import os, sys, time, gc
from pathlib import Path
import cv2
import numpy as np
from skimage import io as skio
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}


def load_gt_masks(label_path, h, w):
    anns = []
    if not label_path.exists(): return anns
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) < 7: continue
        cid = int(parts[0])
        if cid not in CLASS_NAMES: continue
        pts = [float(x) for x in parts[1:]]
        xs = [pts[i]*w for i in range(0, len(pts), 2)]
        ys = [pts[i]*h for i in range(1, len(pts), 2)]
        rr, cc = sk_polygon(ys, xs, shape=(h, w))
        if len(rr) == 0: continue
        mask = np.zeros((h, w), dtype=bool); mask[rr, cc] = True
        anns.append({"mask": mask})
    return anns


def match_and_score(gt_anns, pred_masks, iou_thr=0.3):
    n_gt, n_pred = len(gt_anns), len(pred_masks)
    if n_gt == 0 and n_pred == 0: return 0, 0, 0, []
    if n_gt == 0: return 0, n_pred, 0, []
    if n_pred == 0: return 0, 0, n_gt, []
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i in range(n_gt):
        for j in range(n_pred):
            inter = np.sum(gt_anns[i]["mask"] & pred_masks[j])
            union = np.sum(gt_anns[i]["mask"] | pred_masks[j])
            iou_matrix[i, j] = inter / union if union > 0 else 0
    matched_gt, matched_pred, matched_ious = set(), set(), []
    while len(matched_gt) < n_gt and len(matched_pred) < n_pred:
        rem = iou_matrix.copy()
        for i in matched_gt: rem[i, :] = -1
        for j in matched_pred: rem[:, j] = -1
        bi, bj = np.unravel_index(np.argmax(rem), rem.shape)
        if rem[bi, bj] < iou_thr: break
        matched_gt.add(bi); matched_pred.add(bj)
        matched_ious.append(rem[bi, bj])
    tp = len(matched_gt)
    return tp, n_pred - len(matched_pred), n_gt - tp, matched_ious


def filter_predictions(pred_masks, image, min_area=100, max_area=50000,
                        min_circularity=0.3, min_intensity_diff=10):
    """Filter FP predictions based on morphological criteria."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    bg_intensity = np.median(gray)
    filtered = []
    for mask in pred_masks:
        area = int(np.sum(mask))
        if area < min_area or area > max_area:
            continue
        coords = np.where(mask)
        if len(coords[0]) == 0: continue
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        bw, bh = x2-x1+1, y2-y1+1
        circularity = (4 * np.pi * area) / ((2*(bw+bh))**2 + 1e-8)
        if circularity < min_circularity:
            continue
        cell_intensity = np.mean(gray[mask])
        if abs(cell_intensity - bg_intensity) < min_intensity_diff:
            continue
        filtered.append(mask)
    return filtered


def run_eval_config(image_list, labels_dir, config_name, **kwargs):
    from labeling_tool.cellpose_utils import (
        _get_model, postprocess_segmentation, _label_map_to_polygons
    )
    cellprob = kwargs.get("cellprob_threshold", -3.0)
    diameter = kwargs.get("diameter", 30.0)
    min_area = kwargs.get("min_area", 100)
    flow_threshold = kwargs.get("flow_threshold", 0.4)
    do_filter = kwargs.get("do_filter", False)
    filter_kwargs = kwargs.get("filter_kwargs", {})

    model = _get_model(True)
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []

    for idx, img_path in enumerate(image_list):
        img = skio.imread(str(img_path))
        if img.ndim == 2: img = np.stack([img]*3, axis=-1)
        elif img.shape[-1] == 4: img = img[:,:,:3]
        h, w = img.shape[:2]

        result = model.eval([img], diameter=diameter, channels=[0,0],
                            cellprob_threshold=cellprob,
                            flow_threshold=flow_threshold)
        label_map = result[0][0]
        label_map = postprocess_segmentation(label_map, min_area=min_area)
        polys = _label_map_to_polygons(label_map, 0, min_area)

        pred_masks = []
        for p in polys:
            pts = p["points"]
            xs = [pts[i]*w for i in range(0, len(pts), 2)]
            ys = [pts[i]*h for i in range(1, len(pts), 2)]
            if len(xs) < 3: continue
            rr, cc = sk_polygon(ys, xs, shape=(h, w))
            if len(rr) == 0: continue
            m = np.zeros((h, w), dtype=bool); m[rr, cc] = True
            pred_masks.append(m)

        if do_filter:
            pred_masks = filter_predictions(pred_masks, img, **filter_kwargs)

        gt_anns = load_gt_masks(labels_dir / (img_path.stem + ".txt"), h, w)
        tp, fp, fn, ious = match_and_score(gt_anns, pred_masks)
        total_tp += tp; total_fp += fp; total_fn += fn
        all_ious.extend(ious)

        if idx % 10 == 0:
            print(f"  [{idx+1}/{len(image_list)}] GT={len(gt_anns)} Pred={len(pred_masks)}")

        del pred_masks, gt_anns, label_map

    prec = total_tp/(total_tp+total_fp) if total_tp+total_fp > 0 else 0
    rec = total_tp/(total_tp+total_fn) if total_tp+total_fn > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
    miou = float(np.mean(all_ious)) if all_ious else 0
    return {"prec": prec, "rec": rec, "f1": f1, "miou": miou,
            "tp": total_tp, "fp": total_fp, "fn": total_fn}


def main():
    images_dir = DATA_ROOT / "images" / "val"
    labels_dir = DATA_ROOT / "labels_polygon" / "val"
    image_list = sorted(images_dir.glob("*.png"))
    print(f"CellposeSAM v3 on {len(image_list)} images")

    configs = [
        ("cpm3_baseline", {"cellprob_threshold": -3.0}),
        ("cpm4", {"cellprob_threshold": -4.0}),
        ("cpm5", {"cellprob_threshold": -5.0}),
        ("cpm3_d40", {"cellprob_threshold": -3.0, "diameter": 40.0}),
        ("cpm3_d50", {"cellprob_threshold": -3.0, "diameter": 50.0}),
        ("cpm3_flow03", {"cellprob_threshold": -3.0, "flow_threshold": 0.3}),
        ("cpm3_flow02", {"cellprob_threshold": -3.0, "flow_threshold": 0.2}),
        ("cpm3_filter_basic", {"cellprob_threshold": -3.0, "do_filter": True,
                               "filter_kwargs": {"min_area": 150, "max_area": 40000,
                                                 "min_circularity": 0.3, "min_intensity_diff": 10}}),
        ("cpm3_filter_strict", {"cellprob_threshold": -3.0, "do_filter": True,
                                "filter_kwargs": {"min_area": 200, "max_area": 30000,
                                                  "min_circularity": 0.4, "min_intensity_diff": 15}}),
        ("cpm3_filter_vstrict", {"cellprob_threshold": -3.0, "do_filter": True,
                                 "filter_kwargs": {"min_area": 300, "max_area": 25000,
                                                   "min_circularity": 0.45, "min_intensity_diff": 20}}),
        ("cpm2_filter_strict", {"cellprob_threshold": -2.0, "do_filter": True,
                                "filter_kwargs": {"min_area": 200, "max_area": 30000,
                                                  "min_circularity": 0.4, "min_intensity_diff": 15}}),
    ]

    results = {}
    for name, kwargs in configs:
        print(f"\n>>> {name}")
        r = run_eval_config(image_list, labels_dir, name, **kwargs)
        results[name] = r
        print(f"  P={r['prec']:.4f} R={r['rec']:.4f} F1={r['f1']:.4f} IoU={r['miou']:.4f} "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']}")
        gc.collect()

    print(f"\n{'='*110}")
    print("CELLPOSESAM V3 RESULTS")
    print(f"{'='*110}")
    print(f"{'Config':<30} {'Prec':>7} {'Rec':>7} {'F1':>7} {'mIoU':>7}  {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 100)
    for name in sorted(results.keys(), key=lambda x: -results[x]["f1"]):
        r = results[name]
        print(f"{name:<30} {r['prec']:>7.4f} {r['rec']:>7.4f} {r['f1']:>7.4f} {r['miou']:>7.4f}  "
              f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5}")
    best = max(results.keys(), key=lambda x: results[x]["f1"])
    print(f"\nBEST: {best} -> F1={results[best]['f1']:.4f}")


if __name__ == "__main__":
    main()
