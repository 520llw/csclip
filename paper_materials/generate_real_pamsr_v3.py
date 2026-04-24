#!/usr/bin/env python3
"""Generate REAL PAMSR vs Single-scale for selected best candidates only."""
import sys, gc
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import polygon as sk_polygon
from scipy import ndimage

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
OUT_DIR = Path("/home/xut/csclip/paper_materials/pamsr_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
GT_COLORS = {
    3: (255, 80, 80),    # Eos - red
    4: (80, 120, 255),   # Neu - blue
    5: (60, 200, 60),    # Lym - green
    6: (255, 160, 30),   # Mac - orange
}

# Best candidates from the log (rescued>0, FN decreased or same, no big FP increase)
CANDIDATES = [
    ("2022-06-10-14-09-32-87353", "GT=77, rescued=3, FN 21→18"),
    ("2022-06-10-14-05-26-85638", "GT=66, rescued=1, FN 3→2"),
    ("2022-06-10-14-03-51-27123", "GT=67, rescued=1, FN 6→5"),
    ("2022-06-10-14-34-55-71733", "GT=61, rescued=1, FN 7→6"),
]


def load_gt_with_class(label_path, h, w):
    masks, cids = [], []
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


def pamsr(model, img, primary_d=50, secondary_ds=[40, 65],
          cellprob=-3.0, rescue_prob_thr=0.0, rescue_need_consensus=True,
          min_area=80, overlap_thr=0.2):
    h, w = img.shape[:2]
    masks_p, flows_p, _ = model.eval(img, diameter=primary_d,
                                     cellprob_threshold=cellprob, channels=[0, 0])
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


def draw_mask_outlines(img_arr, masks, color, linewidth=2):
    result = img_arr.copy()
    n = int(masks.max()) if masks.max() > 0 else 0
    for pid in range(1, n + 1):
        mask = masks == pid
        if mask.sum() < 30:
            continue
        eroded = ndimage.binary_erosion(mask, iterations=linewidth)
        outline = mask & (~eroded)
        result[outline] = color
    return result


def draw_gt_outlines(img_arr, gt_masks, gt_cids, linewidth=2):
    result = img_arr.copy()
    for mask, cid in zip(gt_masks, gt_cids):
        eroded = ndimage.binary_erosion(mask, iterations=linewidth)
        outline = mask & (~eroded)
        color = GT_COLORS.get(cid, (255, 255, 255))
        result[outline] = color
    return result


def make_panel(img_arr, gt_masks, gt_cids, single_masks, pamsr_masks, info):
    h, w = img_arr.shape[:2]
    p1 = img_arr.copy()
    p2 = draw_mask_outlines(img_arr, single_masks, (0, 200, 200))  # cyan
    p3 = draw_mask_outlines(img_arr, pamsr_masks, (0, 200, 0))     # green
    p4 = draw_gt_outlines(img_arr, gt_masks, gt_cids)

    # Highlight rescued cells: cells in pamsr that were FN in single
    s_tp, s_fp, s_fn, s_mg, s_mp = match_detailed(gt_masks, single_masks)
    p_tp, p_fp, p_fn, p_mg, p_mp = match_detailed(gt_masks, pamsr_masks)
    rescued_gt_ids = p_mg - s_mg

    # Add yellow highlight circles around rescued cells in PAMSR panel
    for gid in rescued_gt_ids:
        mask = gt_masks[gid]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        r = max(20, int(((ys.max()-ys.min()) + (xs.max()-xs.min())) / 4))
        # Draw circle with boundary clamping
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


def main():
    from cellpose import models
    print("Loading CellposeSAM...")
    model = models.CellposeModel(gpu=True, model_type="cpsam")
    print("Loaded.")

    img_dir = ROOT / "images" / "val"
    lbl_dir = ROOT / "labels_polygon" / "val"

    results = []
    for stem, info in CANDIDATES:
        ip = img_dir / f"{stem}_2048-1536.png"
        lp = lbl_dir / f"{stem}_2048-1536.txt"
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        gt_masks, gt_cids = load_gt_with_class(lp, h, w)
        print(f"\n{stem}: GT={len(gt_masks)} cells")

        single_masks, _, _ = model.eval(img, diameter=50, cellprob_threshold=-3.0, channels=[0, 0])
        pamsr_masks, stats = pamsr(model, img, primary_d=50, secondary_ds=[40, 65],
                                   cellprob=-3.0, rescue_prob_thr=0.0, rescue_need_consensus=True)

        s_tp, s_fp, s_fn, _, _ = match_detailed(gt_masks, single_masks)
        p_tp, p_fp, p_fn, _, _ = match_detailed(gt_masks, pamsr_masks)
        print(f"  Single: TP={s_tp} FP={s_fp} FN={s_fn}")
        print(f"  PAMSR:  TP={p_tp} FP={p_fp} FN={p_fn}  rescued={stats['rescued']}  primary={stats['primary']}")

        panel, rescued_count = make_panel(img, gt_masks, gt_cids, single_masks, pamsr_masks, info)
        out_path = OUT_DIR / f"pamsr_real_{stem}.png"
        panel.save(out_path, dpi=(300, 300))
        print(f"  Saved: {out_path}")

        results.append({
            "stem": stem,
            "single": (s_tp, s_fp, s_fn),
            "pamsr": (p_tp, p_fp, p_fn),
            "rescued": stats['rescued'],
        })

        del single_masks, pamsr_masks
        gc.collect()

    with open(OUT_DIR / "pamsr_real_summary.txt", "w") as f:
        for r in results:
            f.write(f"{r['stem']}: Single TP/FP/FN={r['single']} -> PAMSR TP/FP/FN={r['pamsr']}, rescued={r['rescued']}\n")
    print(f"\nDone. Saved summary to {OUT_DIR / 'pamsr_real_summary.txt'}")


if __name__ == "__main__":
    main()
