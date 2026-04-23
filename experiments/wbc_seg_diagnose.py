#!/usr/bin/env python3
"""Diagnose why CellposeSAM detects few cells on WBC-Seg images."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

sys.path.insert(0, "/home/xut/csclip/experiments")
from wbc_seg_benchmark import parse_yolo_seg_label, _patch_cellpose_numpy2
_patch_cellpose_numpy2()

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/WBC Seg/yolo_seg_dataset")
IMG_DIR = DATA_ROOT / "images" / "val"
LBL_DIR = DATA_ROOT / "labels" / "val"

img_files = sorted(IMG_DIR.iterdir())
img_files = [p for p in img_files if p.suffix.lower() in (".jpg", ".jpeg", ".png")][:3]

from cellpose import models as cp_models
model = cp_models.CellposeModel(gpu=True, pretrained_model="cpsam")

for p in img_files:
    print(f"\n=== {p.name} ===")
    img = np.array(Image.open(p).convert("RGB"))
    h, w = img.shape[:2]
    print(f"Image shape: {h}x{w}")

    gt = parse_yolo_seg_label(LBL_DIR / (p.stem + ".txt"), w, h)
    print(f"GT instances: {len(gt)}")
    if gt:
        areas = [m.sum() for m in gt]
        diams = [2 * np.sqrt(a / np.pi) for a in areas]
        print(f"  GT mean area={np.mean(areas):.0f}, mean diameter={np.mean(diams):.1f}px, "
              f"min_d={np.min(diams):.1f}, max_d={np.max(diams):.1f}")

    # Try various diameters
    for diam in [0, 15, 20, 30, 50]:
        result = model.eval([img], diameter=float(diam) if diam > 0 else None,
                            channels=[0, 0], cellprob_threshold=-2.0)
        lm = result[0][0]
        ids = np.unique(lm); ids = ids[ids != 0]
        print(f"  cpsam diam={diam}: {len(ids)} instances detected")
        if len(ids) > 0:
            inst_areas = [int((lm == u).sum()) for u in ids]
            print(f"    area range: min={min(inst_areas)}, median={int(np.median(inst_areas))}, "
                  f"max={max(inst_areas)}")

print("\nDone")
