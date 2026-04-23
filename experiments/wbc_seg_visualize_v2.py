#!/usr/bin/env python3
"""Re-visualize WBC-Seg segmentation using CACHED CellposeSAM label maps.
No GPU needed - reads from /tmp/wbc_cpsam_cache/."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/xut/csclip/experiments")
from wbc_seg_benchmark import parse_yolo_seg_label, semantic_iou_dice, instance_tpfpfn
from wbc_seg_roi_eval import (instances_from_label_map, otsu_instances,
                                compute_roi, filter_instances_in_roi)

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset")
IMG_DIR = DATA_ROOT / "images" / "val"
LBL_DIR = DATA_ROOT / "labels" / "val"
CACHE = Path("/tmp/wbc_cpsam_cache")
FIG = Path("/home/xut/csclip/experiments/figures/wbc_seg_compare_v2.png")


def overlay(img, instances, alpha=0.35, outline=True, cmap_name="tab20"):
    out = img.copy().astype(np.float32)
    cmap = plt.get_cmap(cmap_name)
    for i, m in enumerate(instances):
        c = np.array(cmap(i % 20)[:3]) * 255
        out[m] = (1 - alpha) * out[m] + alpha * c
        if outline:
            cnts, _ = cv2.findContours(m.astype(np.uint8),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, c.tolist(), thickness=2)
    return out.clip(0, 255).astype(np.uint8)


def draw_roi_rect(ax, roi, color="yellow", label=None):
    if roi is None:
        return
    x1, y1, x2, y2 = roi
    import matplotlib.patches as patches
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor=color,
                              facecolor="none", linestyle="--")
    ax.add_patch(rect)
    if label:
        ax.text(x1 + 10, y1 + 30, label, color=color, fontsize=9,
                 bbox=dict(facecolor="black", alpha=0.5, pad=2))


def main():
    imgs = sorted(IMG_DIR.iterdir())
    imgs = [p for p in imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    # Pick 6 evenly spaced samples covering the val split
    n_picks = 6
    picks = [imgs[i] for i in np.linspace(0, len(imgs) - 1, n_picks, dtype=int)]

    fig, axes = plt.subplots(n_picks, 4, figsize=(20, 4.5 * n_picks))
    if n_picks == 1:
        axes = axes[np.newaxis, :]

    for row, p in enumerate(picks):
        img = np.array(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]

        # Downscale for fast rendering
        sc = 900 / max(h, w)
        disp = cv2.resize(img, (int(w * sc), int(h * sc)))
        H, W = disp.shape[:2]

        def rs(lst):
            return [cv2.resize(m.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                     for m in lst]

        # Load cached CellposeSAM label map
        cache_file = CACHE / f"{p.stem}.npz"
        if not cache_file.exists():
            print(f"  SKIP (no cache): {p.name}")
            continue
        lm = np.load(cache_file)["label_map"]
        ins_c = instances_from_label_map(lm)
        ins_o = otsu_instances(img)
        gt = parse_yolo_seg_label(LBL_DIR / (p.stem + ".txt"), w, h)
        roi = compute_roi(gt, (h, w))
        roi_disp = None
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            roi_disp = (int(rx1 * sc), int(ry1 * sc),
                        int(rx2 * sc), int(ry2 * sc))

        # Metrics
        so, do = semantic_iou_dice(ins_o, gt, (h, w))
        sc2, dc = semantic_iou_dice(ins_c, gt, (h, w))
        tpo, fpo, fno, _, _ = instance_tpfpfn(ins_o, gt, iou_thr=0.5)
        tpc, fpc, fnc, iousum, nm = instance_tpfpfn(ins_c, gt, iou_thr=0.5)
        miou_c = iousum / nm if nm > 0 else 0.0

        # col 0: original
        axes[row, 0].imshow(disp)
        axes[row, 0].set_title(f"Original  {w}×{h}\n{p.name[:30]}", fontsize=10)
        draw_roi_rect(axes[row, 0], roi_disp, "yellow", "GT ROI")

        # col 1: GT
        axes[row, 1].imshow(overlay(disp, rs(gt)))
        axes[row, 1].set_title(f"GT  ({len(gt)} polygons)\n"
                                f"Note: only covers ~57% of image",
                                fontsize=10, color="darkred")
        draw_roi_rect(axes[row, 1], roi_disp, "yellow")

        # col 2: Otsu
        axes[row, 2].imshow(overlay(disp, rs(ins_o)))
        axes[row, 2].set_title(
            f"Otsu  ({len(ins_o)} inst)\n"
            f"Semantic IoU={so:.3f} | Inst TP={tpo}/{len(gt)} FP={fpo}",
            fontsize=9)

        # col 3: CellposeSAM
        axes[row, 3].imshow(overlay(disp, rs(ins_c)))
        axes[row, 3].set_title(
            f"CellposeSAM  ({len(ins_c)} inst)\n"
            f"Semantic IoU={sc2:.3f} | Inst TP={tpc}/{len(gt)} "
            f"FP={fpc} | MatchedIoU={miou_c:.3f}",
            fontsize=9, color="darkgreen")

        for c in range(4):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])

        print(f"  row {row}: {p.name}  GT={len(gt)}  Otsu={len(ins_o)}  "
              f"cpsam={len(ins_c)}  cpsam_matched_IoU={miou_c:.3f}",
              flush=True)

    plt.suptitle("WBC-Seg val split: Ground Truth (flawed coverage) vs Otsu vs CellposeSAM\n"
                  "Yellow dashed box = GT annotation coverage ROI",
                  fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(FIG, dpi=110, bbox_inches="tight")
    print(f"\nSaved: {FIG}")


if __name__ == "__main__":
    main()
