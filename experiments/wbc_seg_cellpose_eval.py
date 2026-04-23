"""
Evaluate CellposeSAM segmentation on WBC Seg yolo_seg_dataset.
Optimized parameters: cellprob_threshold=-3.0, diameter=50.0
Resizes large images to max 2048px to avoid GPU OOM.
"""
import os, sys, time, gc, traceback
import numpy as np
import cv2

sys.path.insert(0, "/home/xut/csclip")
from PIL import Image, ImageOps

DATA_ROOT = "/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset"
IOU_THR = 0.3
EVAL_SCALE = 1024
MAX_IMG_EDGE = 2048  # resize images larger than this before cellpose

def load_image_exif(path, max_edge=MAX_IMG_EDGE):
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode != "RGB": img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return np.array(img)

def read_yolo_polygons(label_path):
    anns = []
    if not os.path.isfile(label_path): return anns
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7: continue
            anns.append({"class_id": int(parts[0]), "ann_type": "polygon",
                          "points": [float(x) for x in parts[1:]]})
    return anns

def ann_to_mask(ann, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = ann["points"]
    if len(pts) >= 6:
        poly = np.array([(int(pts[i]*w), int(pts[i+1]*h)) for i in range(0, len(pts), 2)], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool)

def evaluate(pred_anns, gold_anns, img_h, img_w):
    scale = min(EVAL_SCALE / max(img_h, img_w), 1.0)
    eh, ew = int(img_h * scale), int(img_w * scale)
    pred_masks = [ann_to_mask(a, eh, ew) for a in pred_anns]
    gold_masks = [ann_to_mask(a, eh, ew) for a in gold_anns]
    matched, ious, dices = 0, [], []
    used = set()
    for pm in pred_masks:
        best_iou, best_gi = 0.0, -1
        for gi, gm in enumerate(gold_masks):
            if gi in used: continue
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou = float(inter) / max(float(union), 1)
            if iou > best_iou: best_iou, best_gi = iou, gi
        if best_iou > IOU_THR and best_gi >= 0:
            matched += 1; used.add(best_gi); ious.append(best_iou)
            dd = float(pm.sum() + gold_masks[best_gi].sum())
            dices.append(2.0 * float(np.logical_and(pm, gold_masks[best_gi]).sum()) / max(dd, 1))
    np_, ng = len(pred_masks), len(gold_masks)
    p = matched / max(np_, 1); r = matched / max(ng, 1)
    f1 = 2*p*r / max(p+r, 1e-9)
    return dict(n_pred=np_, n_gold=ng, matched=matched, precision=p, recall=r, f1=f1,
                mean_iou=float(np.mean(ious)) if ious else 0.0,
                mean_dice=float(np.mean(dices)) if dices else 0.0)

def run_cellpose(img_rgb, diameter, cellprob_thr, min_area):
    from cellpose import models
    if not hasattr(run_cellpose, '_model'):
        run_cellpose._model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    result = run_cellpose._model.eval([img_rgb], diameter=diameter, cellprob_threshold=cellprob_thr)
    lmap = result[0][0]
    h, w = lmap.shape[:2]
    anns = []
    for uid in np.unique(lmap):
        if uid == 0: continue
        mask = (lmap == uid).astype(np.uint8)
        if np.sum(mask) < min_area: continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area: continue
        eps = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3: continue
        pts = []
        for pt in approx:
            x, y = pt[0]; pts.extend([round(x/w, 6), round(y/h, 6)])
        anns.append({"class_id": 0, "ann_type": "polygon", "points": pts})
    return anns

def run_eval(subset, imgs_dir, lbls_dir, diameter, cellprob_thr, min_area):
    files = sorted(f for f in os.listdir(imgs_dir) if f.lower().endswith((".jpg",".jpeg",".png")))
    print(f"\n{'='*70}")
    print(f"[{subset}] {len(files)} images | d={diameter}, cp_thr={cellprob_thr}, min_area={min_area}")
    print(f"{'='*70}", flush=True)

    results = []; tp, tg, tm = 0, 0, 0
    errors = 0
    t0 = time.time()
    for i, fn in enumerate(files):
        gold = read_yolo_polygons(os.path.join(lbls_dir, os.path.splitext(fn)[0]+".txt"))
        if not gold: continue
        try:
            img = load_image_exif(os.path.join(imgs_dir, fn))
            h, w = img.shape[:2]
            pred = run_cellpose(img, diameter, cellprob_thr, min_area)
            del img; gc.collect()
            m = evaluate(pred, gold, h, w)
            results.append(m); tp += m["n_pred"]; tg += m["n_gold"]; tm += m["matched"]
        except Exception as e:
            errors += 1
            print(f"  ERROR on {fn}: {e}", flush=True)
            gc.collect()
            import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None
            continue
        if (i+1) % 10 == 0 or (i+1) == len(files):
            spd = (i+1)/(time.time()-t0)
            print(f"  [{i+1}/{len(files)}] {spd:.2f} img/s | "
                  f"P={m['n_pred']} G={m['n_gold']} M={m['matched']} F1={m['f1']:.3f} IoU={m['mean_iou']:.3f}", flush=True)

    elapsed = time.time() - t0
    op = tm/max(tp,1); orr = tm/max(tg,1); of1 = 2*op*orr/max(op+orr,1e-9)
    f1s = [r["f1"] for r in results]
    mi = [r["mean_iou"] for r in results if r["mean_iou"]>0]
    md = [r["mean_dice"] for r in results if r["mean_dice"]>0]
    ps = [r["precision"] for r in results]
    rs = [r["recall"] for r in results]

    print(f"\n{'─'*70}")
    print(f"RESULTS — {subset} ({len(results)} images, {errors} errors)")
    print(f"{'─'*70}")
    print(f"  Predictions:    {tp}")
    print(f"  Ground truth:   {tg}")
    print(f"  Matched:        {tm} (IoU>{IOU_THR})")
    print(f"  Precision:      {op:.4f}")
    print(f"  Recall:         {orr:.4f}")
    print(f"  F1 (overall):   {of1:.4f}")
    print(f"  F1 (per-img):   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    if mi: print(f"  IoU (matched):  {np.mean(mi):.4f} ± {np.std(mi):.4f}")
    if md: print(f"  Dice (matched): {np.mean(md):.4f} ± {np.std(md):.4f}")
    print(f"  Precision/img:  {np.mean(ps):.4f} ± {np.std(ps):.4f} (med={np.median(ps):.4f})")
    print(f"  Recall/img:     {np.mean(rs):.4f} ± {np.std(rs):.4f} (med={np.median(rs):.4f})")
    print(f"  Time:           {elapsed:.1f}s ({len(results)/max(elapsed,1):.2f} img/s)")
    print(f"{'─'*70}", flush=True)
    return results

if __name__ == "__main__":
    run_eval("val", DATA_ROOT+"/images/val", DATA_ROOT+"/labels/val", 50.0, -3.0, 100)
    run_eval("train_sample", DATA_ROOT+"/images/train", DATA_ROOT+"/labels/train", 50.0, -3.0, 100)
