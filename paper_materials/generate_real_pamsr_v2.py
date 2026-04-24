#!/usr/bin/env python3
"""Generate REAL PAMSR vs Single-scale segmentation comparisons on data2.
Run in cel conda env: /data/software/mamba/envs/cel/bin/python
"""
import sys, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import polygon as sk_polygon

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
OUT_DIR = Path("/home/xut/csclip/paper_materials/pamsr_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
GT_COLORS = {
    3: (255, 100, 100),   # Eos - red
    4: (100, 100, 255),   # Neu - blue
    5: (100, 255, 100),   # Lym - green
    6: (255, 180, 50),    # Mac - orange
}


def load_gt_with_class(label_path, h, w):
    masks = []
    cids = []
    if not label_path.exists():
        return masks, cids
    for line in open(label_path):
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
        masks.append(mask)
        cids.append(cid)
    return masks, cids


def match_detailed(gt_masks, pred_masks, iou_thr=0.5, min_area=30):
    """Return (tp, fp, fn, matched_gt_indices, matched_pred_indices, matched_ious)"""
    n_pred = int(pred_masks.max()) if pred_masks.max() > 0 else 0
    matched_gt = set()
    matched_pred = set()
    matched_ious = {}

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
            matched_ious[best_gi] = best_iou

    tp = len(matched_gt)
    fp = n_pred - len(matched_pred)
    fn = len(gt_masks) - tp
    return tp, fp, fn, matched_gt, matched_pred, matched_ious


def pamsr(model, img, primary_d=50, secondary_ds=[40, 65],
          cellprob=-3.0, rescue_prob_thr=0.0, rescue_need_consensus=True,
          min_area=80, overlap_thr=0.2):
    from cellpose import models
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


def draw_outlines_on_image(img_arr, masks, color_map, linewidth=2):
    """Draw outlines of labeled masks on image."""
    from scipy import ndimage
    result = img_arr.copy()
    n = int(masks.max()) if masks.max() > 0 else 0
    for pid in range(1, n + 1):
        mask = masks == pid
        if mask.sum() < 30:
            continue
        # erode to get outline
        eroded = ndimage.binary_erosion(mask, iterations=linewidth)
        outline = mask & (~eroded)
        color = color_map.get(pid, (255, 255, 255))
        result[outline] = color
    return result


def draw_gt_outlines(img_arr, gt_masks, gt_cids, linewidth=2):
    from scipy import ndimage
    result = img_arr.copy()
    for mask, cid in zip(gt_masks, gt_cids):
        eroded = ndimage.binary_erosion(mask, iterations=linewidth)
        outline = mask & (~eroded)
        color = GT_COLORS.get(cid, (255, 255, 255))
        result[outline] = color
    return result


def make_panel(img_arr, gt_masks, gt_cids, single_masks, pamsr_masks, title_info):
    h, w = img_arr.shape[:2]
    # Create 4 panels
    p1 = img_arr.copy()
    p2 = draw_outlines_on_image(img_arr, single_masks, {pid: (0, 255, 255) for pid in range(1, int(single_masks.max())+1)})
    p3 = draw_outlines_on_image(img_arr, pamsr_masks, {pid: (0, 255, 0) for pid in range(1, int(pamsr_masks.max())+1)})
    p4 = draw_gt_outlines(img_arr, gt_masks, gt_cids)

    panel = np.zeros((h + 60, w * 4, 3), dtype=np.uint8)
    panel[:, :] = (245, 245, 245)
    panel[:h, 0:w] = p1
    panel[:h, w:2*w] = p2
    panel[:h, 2*w:3*w] = p3
    panel[:h, 3*w:4*w] = p4

    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(14, h // 35))
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, h // 45))
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    titles = ["(a) Original", "(b) Single-scale (d=50)", "(c) PAMSR (Ours)", "(d) Ground Truth"]
    colors = [(50, 50, 50), (0, 200, 200), (0, 180, 0), (50, 50, 50)]
    for i, (t, c) in enumerate(zip(titles, colors)):
        bbox = draw.textbbox((0, 0), t, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((i * w + (w - tw) // 2, h + 8), t, fill=c, font=font)

    # stats
    s_tp, s_fp, s_fn, _, _, _ = match_detailed(gt_masks, single_masks)
    p_tp, p_fp, p_fn, _, _, _ = match_detailed(gt_masks, pamsr_masks)
    stats_text = f"Single: TP={s_tp} FP={s_fp} FN={s_fn} F1={2*s_tp/(2*s_tp+s_fp+s_fn):.3f}  |  PAMSR: TP={p_tp} FP={p_fp} FN={p_fn} F1={2*p_tp/(2*p_tp+p_fp+p_fn):.3f}  |  {title_info}"
    draw.text((10, h + 38), stats_text, fill=(80, 80, 80), font=font_small)

    return pil


def main():
    from cellpose import models
    print("Loading CellposeSAM model...")
    model = models.CellposeModel(gpu=True, model_type="cpsam")
    print("Model loaded.")

    img_dir = ROOT / "images" / "val"
    lbl_dir = ROOT / "labels_polygon" / "val"
    images = sorted(img_dir.glob("*.png"))

    candidates = []

    for idx, ip in enumerate(images):
        lp = lbl_dir / (ip.stem + ".txt")
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        gt_masks, gt_cids = load_gt_with_class(lp, h, w)
        if not gt_masks:
            continue

        print(f"\n[{idx+1}/{len(images)}] {ip.stem}: GT={len(gt_masks)} cells")

        # Single-scale
        single_masks, _, _ = model.eval(img, diameter=50, cellprob_threshold=-3.0, channels=[0, 0])
        s_tp, s_fp, s_fn, s_mg, s_mp, _ = match_detailed(gt_masks, single_masks)

        # PAMSR
        pamsr_masks, stats = pamsr(model, img, primary_d=50, secondary_ds=[40, 65],
                                   cellprob=-3.0, rescue_prob_thr=0.0,
                                   rescue_need_consensus=True)
        p_tp, p_fp, p_fn, p_mg, p_mp, _ = match_detailed(gt_masks, pamsr_masks)

        # Find rescued cells: FN in single but TP in PAMSR
        rescued = p_mg - s_mg
        rescued_ids = p_mg - s_mg  # set difference
        rescued_count = len(p_mg - s_mg)

        print(f"  Single: TP={s_tp} FP={s_fp} FN={s_fn}")
        print(f"  PAMSR:  TP={p_tp} FP={p_fp} FN={p_fn}  rescued={stats['rescued']}  primary={stats['primary']}")

        # Save candidate if PAMSR is clearly better
        if p_fn < s_fn and (rescued_count >= 1 or stats['rescued'] >= 2):
            f1_single = 2*s_tp/(2*s_tp+s_fp+s_fn) if (2*s_tp+s_fp+s_fn) > 0 else 0
            f1_pamsr = 2*p_tp/(2*p_tp+p_fp+p_fn) if (2*p_tp+p_fp+p_fn) > 0 else 0
            delta_f1 = f1_pamsr - f1_single
            candidates.append({
                "stem": ip.stem,
                "img": img,
                "gt_masks": gt_masks,
                "gt_cids": gt_cids,
                "single_masks": single_masks,
                "pamsr_masks": pamsr_masks,
                "f1_single": f1_single,
                "f1_pamsr": f1_pamsr,
                "delta_f1": delta_f1,
                "rescued": stats['rescued'],
                "s_fn": s_fn,
                "p_fn": p_fn,
            })

        del single_masks, pamsr_masks
        gc.collect()

    print(f"\n{'='*60}")
    print(f"Found {len(candidates)} candidate images with PAMSR improvement")
    print(f"{'='*60}")

    # Sort by delta_f1 descending, then by rescued count
    candidates.sort(key=lambda x: (-x["delta_f1"], -x["rescued"]))

    for i, c in enumerate(candidates[:6]):
        print(f"  {i+1}. {c['stem']}: F1 {c['f1_single']:.3f} -> {c['f1_pamsr']:.3f} (+{c['delta_f1']:.3f}), rescued={c['rescued']}, FN {c['s_fn']} -> {c['p_fn']}")

    # Generate panels for top candidates
    selected = candidates[:4]
    for i, c in enumerate(selected):
        panel = make_panel(
            c["img"], c["gt_masks"], c["gt_cids"],
            c["single_masks"], c["pamsr_masks"],
            f"{c['stem']} | rescued={c['rescued']} cells"
        )
        out_path = OUT_DIR / f"pamsr_real_compare_{c['stem']}.png"
        panel.save(out_path, dpi=(300, 300))
        print(f"Saved: {out_path}")

    # Save summary log
    with open(OUT_DIR / "pamsr_real_log.txt", "w") as f:
        f.write("Real PAMSR vs Single-scale comparison on data2_organized val\n")
        f.write("="*60 + "\n")
        for c in candidates:
            f.write(f"{c['stem']}: F1 {c['f1_single']:.4f} -> {c['f1_pamsr']:.4f} (+{c['delta_f1']:.4f}), "
                    f"rescued={c['rescued']}, FN {c['s_fn']} -> {c['p_fn']}\n")
    print(f"Saved log: {OUT_DIR / 'pamsr_real_log.txt'}")


if __name__ == "__main__":
    main()
