#!/usr/bin/env python3
"""Visualize WBC-Seg segmentation: GT vs Otsu vs CellposeSAM."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/xut/csclip/experiments")
from wbc_seg_benchmark import (parse_yolo_seg_label, _patch_cellpose_numpy2,
                                instance_tpfpfn, semantic_iou_dice)
_patch_cellpose_numpy2()
from raabin_segment_eval import otsu_segment, cellpose_segment

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset")
IMG_DIR = DATA_ROOT / "images" / "val"
LBL_DIR = DATA_ROOT / "labels" / "val"
FIG = Path("/home/xut/csclip/experiments/figures/wbc_seg_compare.png")
FIG.parent.mkdir(parents=True, exist_ok=True)


def overlay(img, instances, alpha=0.4, outline=True):
    out = img.copy().astype(np.float32)
    cmap = plt.get_cmap("tab20")
    for i, m in enumerate(instances):
        c = np.array(cmap(i % 20)[:3]) * 255
        out[m] = (1 - alpha) * out[m] + alpha * c
        if outline:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, c.tolist(), thickness=3)
    return out.clip(0, 255).astype(np.uint8)


def main():
    from cellpose import models as cp_models
    print("[Loading CellposeSAM ...]", flush=True)
    model = cp_models.CellposeModel(gpu=True, pretrained_model="cpsam")

    imgs = sorted(IMG_DIR.iterdir())
    imgs = [p for p in imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    # Pick 4 evenly spaced samples
    picks = [imgs[i] for i in np.linspace(5, len(imgs) - 5, 4, dtype=int)]

    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, p in enumerate(picks):
        img = np.array(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]
        # downscale for visualization
        sc = 800 / max(h, w)
        disp = cv2.resize(img, (int(w * sc), int(h * sc)))
        H, W = disp.shape[:2]

        gt = parse_yolo_seg_label(LBL_DIR / (p.stem + ".txt"), w, h)
        ins_o = otsu_segment(img)
        ins_c = cellpose_segment(img, model)

        # Resize masks
        def rs(lst):
            return [cv2.resize(m.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                     for m in lst]

        gt_d = rs(gt); ot_d = rs(ins_o); cp_d = rs(ins_c)

        axes[row, 0].imshow(disp)
        axes[row, 0].set_title(f"Original  {w}x{h}", fontsize=10)
        axes[row, 1].imshow(overlay(disp, gt_d))
        axes[row, 1].set_title(f"GT  ({len(gt)} instances)", fontsize=10)
        axes[row, 2].imshow(overlay(disp, ot_d))
        so, _ = semantic_iou_dice(ins_o, gt, (h, w))
        tp_o, fp_o, fn_o, _, _ = instance_tpfpfn(ins_o, gt, iou_thr=0.5)
        axes[row, 2].set_title(f"Otsu  ({len(ins_o)} inst)\nIoU={so:.3f}  "
                                f"TP={tp_o} FP={fp_o} FN={fn_o}", fontsize=9)
        axes[row, 3].imshow(overlay(disp, cp_d))
        sc2, _ = semantic_iou_dice(ins_c, gt, (h, w))
        tp_c, fp_c, fn_c, _, _ = instance_tpfpfn(ins_c, gt, iou_thr=0.5)
        axes[row, 3].set_title(f"CellposeSAM  ({len(ins_c)} inst)\nIoU={sc2:.3f}  "
                                f"TP={tp_c} FP={fp_c} FN={fn_c}", fontsize=9)

        for c in range(4):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])

        print(f"  row {row}: {p.name}  GT={len(gt)}  Otsu={len(ins_o)}  "
              f"cpsam={len(ins_c)}", flush=True)

    plt.tight_layout()
    plt.savefig(FIG, dpi=110, bbox_inches="tight")
    print(f"\nSaved: {FIG}")


if __name__ == "__main__":
    main()
