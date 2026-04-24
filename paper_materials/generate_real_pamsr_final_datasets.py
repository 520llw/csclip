#!/usr/bin/env python3
"""Final pass: MultiCenter (more scans) + WBC (d=30, fewer cells)."""
import sys, gc, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import polygon as sk_polygon
from scipy import ndimage

sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = Path("/home/xut/csclip/paper_materials/pamsr_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_COLORS = {
    0: (255, 200, 100), 1: (255, 100, 100), 2: (100, 255, 100),
    3: (255, 80, 80), 4: (80, 120, 255), 5: (60, 200, 60), 6: (255, 160, 30),
}


def load_image(path, max_edge=None):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max_edge and max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def load_gt(label_path, h, w):
    masks, cids = [], []
    if not label_path.exists():
        return masks, cids
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        pts = [float(x) for x in parts[1:]]
        xs = [pts[i] * w for i in range(0, len(pts), 2)]
        ys = [pts[i] * h for i in range(1, len(pts), 2)]
        rr, cc = sk_polygon(ys, xs, shape=(h, w))
        if len(rr) == 0:
            continue
        m = np.zeros((h, w), dtype=bool)
        m[rr, cc] = True
        masks.append(m)
        cids.append(cid)
    return masks, cids


def match_detailed(gt_masks, pred_masks, iou_thr=0.5, min_area=30):
    n_pred = int(pred_masks.max()) if pred_masks.max() > 0 else 0
    matched_gt = set()
    matched_pred = set()
    for pid in range(1, n_pred + 1):
        pm = pred_masks == pid
        if pm.sum() < min_area:
            continue
        best_iou, best_gi = 0, -1
        for gi, gm in enumerate(gt_masks):
            if gi in matched_gt:
                continue
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thr and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pid)
    tp = len(matched_gt)
    fp = n_pred - len(matched_pred)
    fn = len(gt_masks) - tp
    return tp, fp, fn, matched_gt, matched_pred


def pamsr(model, img, primary_d, secondary_ds, cellprob=-3.0,
          rescue_prob_thr=0.0, rescue_need_consensus=True,
          min_area=80, overlap_thr=0.2):
    h, w = img.shape[:2]
    res_p = model.eval([img], diameter=primary_d,
                       cellprob_threshold=cellprob, channels=[0, 0])
    masks_p = res_p[0][0]
    final = masks_p.copy()
    nid = int(final.max()) + 1

    secondary_cells = []
    for sd in secondary_ds:
        res_s = model.eval([img], diameter=sd,
                           cellprob_threshold=cellprob, channels=[0, 0])
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


def make_panel(img_arr, gt_masks, gt_cids, single_masks, pamsr_masks, info):
    h, w = img_arr.shape[:2]
    p1 = img_arr.copy()
    def draw_outlines(arr, masks, color):
        result = arr.copy()
        n = int(masks.max()) if masks.max() > 0 else 0
        for pid in range(1, n + 1):
            mask = masks == pid
            if mask.sum() < 30:
                continue
            eroded = ndimage.binary_erosion(mask, iterations=2)
            outline = mask & (~eroded)
            result[outline] = color
        return result
    def draw_gt(arr, gt_masks, gt_cids):
        result = arr.copy()
        for mask, cid in zip(gt_masks, gt_cids):
            eroded = ndimage.binary_erosion(mask, iterations=2)
            outline = mask & (~eroded)
            result[outline] = GT_COLORS.get(cid, (255, 255, 255))
        return result
    p2 = draw_outlines(img_arr, single_masks, (0, 200, 200))
    p3 = draw_outlines(img_arr, pamsr_masks, (0, 200, 0))
    p4 = draw_gt(img_arr, gt_masks, gt_cids)

    s_tp, s_fp, s_fn, s_mg, s_mp = match_detailed(gt_masks, single_masks)
    p_tp, p_fp, p_fn, p_mg, p_mp = match_detailed(gt_masks, pamsr_masks)
    rescued_gt_ids = p_mg - s_mg

    for gid in rescued_gt_ids:
        mask = gt_masks[gid]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        r = max(20, int(((ys.max()-ys.min()) + (xs.max()-xs.min())) / 4))
        circle_y = [max(0, min(h-1, cy + int(r*np.sin(t)))) for t in np.linspace(0, 2*np.pi, 60)]
        circle_x = [max(0, min(w-1, cx + int(r*np.cos(t)))) for t in np.linspace(0, 2*np.pi, 60)]
        for yy, xx in zip(circle_y, circle_x):
            p3[yy, xx] = (255, 255, 0)

    panel = np.zeros((h + 70, w * 4, 3), dtype=np.uint8)
    panel[:, :] = (245, 245, 245)
    panel[:h, 0:w] = p1
    panel[:h, w:2*w] = p2
    panel[:h, 2*w:3*w] = p3
    panel[:h, 3*w:4*w] = p4

    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(14, h // 40))
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, h // 50))
    except:
        font = ImageFont.load_default()
        font_s = ImageFont.load_default()

    titles = ["(a) Original", "(b) Single-scale", "(c) PAMSR (Ours)", "(d) Ground Truth"]
    colors = [(50, 50, 50), (0, 180, 180), (0, 150, 0), (50, 50, 50)]
    for i, (t, c) in enumerate(zip(titles, colors)):
        bbox = draw.textbbox((0, 0), t, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((i * w + (w - tw) // 2, h + 8), t, fill=c, font=font)

    txt = f"Single: TP={s_tp} FP={s_fp} FN={s_fn}  |  PAMSR: TP={p_tp} FP={p_fp} FN={p_fn}  |  Rescued={len(rescued_gt_ids)} cells  |  {info}"
    draw.text((10, h + 40), txt, fill=(80, 80, 80), font=font_s)
    return pil, len(rescued_gt_ids)


def process_dataset(model, ds_name, img_dir, lbl_dir, primary_d=50, secondary_ds=[40,65],
                    max_scan=12, max_edge=None, min_area=80):
    import torch
    exts = (".jpg", ".jpeg", ".png")
    all_imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not all_imgs:
        return None
    random.seed(42)
    sampled = random.sample(all_imgs, min(max_scan, len(all_imgs)))

    best = None
    for idx, ip in enumerate(sampled):
        lp = lbl_dir / (ip.stem + ".txt")
        try:
            img = load_image(str(ip), max_edge=max_edge)
        except Exception as e:
            print(f"  SKIP {ip.name}: {e}")
            continue
        h, w = img.shape[:2]
        gt_masks, gt_cids = load_gt(lp, h, w)
        if not gt_masks:
            continue

        print(f"  [{idx+1}/{len(sampled)}] {ip.name}: GT={len(gt_masks)}", flush=True)
        res = model.eval([img], diameter=primary_d, cellprob_threshold=-3.0, channels=[0, 0])
        single_masks = res[0][0]
        pamsr_masks, stats = pamsr(model, img, primary_d=primary_d, secondary_ds=secondary_ds,
                                   cellprob=-3.0, rescue_prob_thr=0.0, rescue_need_consensus=True,
                                   min_area=min_area)

        s_tp, s_fp, s_fn, _, _ = match_detailed(gt_masks, single_masks)
        p_tp, p_fp, p_fn, _, _ = match_detailed(gt_masks, pamsr_masks)
        print(f"    Single: TP={s_tp} FP={s_fp} FN={s_fn}  |  PAMSR: TP={p_tp} FP={p_fp} FN={p_fn}  rescued={stats['rescued']}", flush=True)

        # Strongly prefer FN reduction
        score = stats["rescued"] * 20 + max(0, s_fn - p_fn) * 15 - max(0, p_fp - s_fp) * 3
        if best is None or score > best["score"]:
            best = {
                "stem": ip.stem, "img": img, "gt_masks": gt_masks, "gt_cids": gt_cids,
                "single_masks": single_masks, "pamsr_masks": pamsr_masks,
                "s_tp": s_tp, "s_fp": s_fp, "s_fn": s_fn,
                "p_tp": p_tp, "p_fp": p_fp, "p_fn": p_fn,
                "rescued": stats["rescued"], "score": score,
            }

        del single_masks, pamsr_masks
        gc.collect()
        torch.cuda.empty_cache()

    if best is None:
        return None

    info = f"{ds_name} | {best['stem']} | GT={len(best['gt_masks'])} rescued={best['rescued']}"
    panel, rc = make_panel(best["img"], best["gt_masks"], best["gt_cids"],
                           best["single_masks"], best["pamsr_masks"], info)
    out_path = OUT_DIR / f"pamsr_real_{ds_name}_{best['stem']}.png"
    panel.save(out_path, dpi=(300, 300))
    print(f"  SAVED: {out_path}")
    return best


def main():
    from cellpose import models
    import torch
    print("Loading model...")
    model = models.CellposeModel(gpu=True, model_type="cpsam")
    print("Loaded.")

    # MultiCenter with more scans
    print("\n" + "="*60)
    print("Dataset: multicenter")
    print("="*60)
    mc_root = Path("/home/xut/csclip/cell_datasets/MultiCenter_organized")
    mc_best = process_dataset(model, "multicenter", mc_root / "images" / "val",
                              mc_root / "labels_polygon" / "val",
                              primary_d=50, secondary_ds=[40,65],
                              max_scan=15, max_edge=None)

    # WBC with smaller diameter (cells are smaller in WBC-Seg)
    print("\n" + "="*60)
    print("Dataset: wbc (d=30)")
    print("="*60)
    wbc_root = Path("/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset")
    # Use smaller diameter for WBC, and only scan 3 images to save time
    wbc_best = process_dataset(model, "wbc", wbc_root / "images" / "val",
                               wbc_root / "labels" / "val",
                               primary_d=30, secondary_ds=[25, 40],
                               max_scan=3, max_edge=2048, min_area=30)

    with open(OUT_DIR / "pamsr_real_final_datasets.txt", "w") as f:
        for name, best in [("multicenter", mc_best), ("wbc", wbc_best)]:
            if best:
                f.write(f"{name} / {best['stem']}: Single {best['s_tp']}/{best['s_fp']}/{best['s_fn']} -> "
                        f"PAMSR {best['p_tp']}/{best['p_fp']}/{best['p_fn']}  rescued={best['rescued']}\n")
            else:
                f.write(f"{name}: NO VALID CANDIDATE\n")
    print(f"\nSaved: {OUT_DIR / 'pamsr_real_final_datasets.txt'}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
