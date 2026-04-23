#!/usr/bin/env python3
"""
Visualize Otsu vs Cellpose vs Cellpose+SAM3 segmentation on Raabin samples.
Output: /home/xut/csclip/experiments/figures/raabin_seg_compare.png (+ individual PNGs).
"""
from __future__ import annotations
import os, sys, random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/xut/csclip")
sys.path.insert(0, "/home/xut/csclip/sam3")
sys.path.insert(0, "/home/xut/csclip/experiments")

from raabin_segment_eval import otsu_segment, cellpose_segment, sam3_refine

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/Raabin-WBC-top3")
IMG_DIR = DATA_ROOT / "images"
LBL_DIR = DATA_ROOT / "labels"
SAM3_CKPT = Path("/home/xut/csclip/labeling_tool/weights/sam3.pt")
FIG_DIR = Path("/home/xut/csclip/experiments/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "Eosinophil", 1: "Lymphocyte", 2: "Neutrophil"}
N_PER_CLASS = 3  # 3 samples per class for the figure
SEED = 7


def sample_images():
    pools = {c: [] for c in CLASS_NAMES}
    for img in sorted(IMG_DIR.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = LBL_DIR / (img.stem + ".txt")
        if not lbl.exists():
            continue
        with lbl.open() as f:
            line = f.readline().strip()
        if not line:
            continue
        cid = int(line.split()[0])
        if cid in pools:
            pools[cid].append(img)
    random.seed(SEED)
    selected = []
    for cid in CLASS_NAMES:
        pool = pools[cid]
        random.shuffle(pool)
        for p in pool[:N_PER_CLASS]:
            selected.append((p, cid))
    return selected


def overlay_instances(img_rgb, instances, cmap_name="tab10", alpha=0.45):
    """Draw translucent colored masks on top of the image."""
    overlay = img_rgb.copy().astype(np.float32)
    cmap = plt.get_cmap(cmap_name)
    for i, m in enumerate(instances):
        color = np.array(cmap(i % 10)[:3]) * 255
        overlay[m] = (1 - alpha) * overlay[m] + alpha * color
        # draw contour for clarity
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), thickness=2)
    return overlay.clip(0, 255).astype(np.uint8)


def main():
    print("[Loading Cellpose ...]", flush=True)
    from cellpose import models as cp_models
    cp_model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model="cpsam")
    print("[Loading SAM3 ...]", flush=True)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3_pkg = Path(sys.modules["sam3"].__file__).parent
    bpe = sam3_pkg / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe.exists():
        bpe = sam3_pkg.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_model = build_sam3_image_model(
        bpe_path=str(bpe), checkpoint_path=str(SAM3_CKPT),
        device=device, eval_mode=True, load_from_HF=False,
        enable_inst_interactivity=True)
    sam3_proc = Sam3Processor(sam3_model, confidence_threshold=0.3, device=device)

    samples = sample_images()
    print(f"Selected {len(samples)} samples ({N_PER_CLASS}/class)")

    n_rows = len(samples)
    n_cols = 4  # Original | Otsu | Cellpose | Cellpose+SAM3
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original",
                  "Otsu (simple threshold)",
                  "Cellpose (cpsam)",
                  "Cellpose + SAM3"]

    for row, (path, cid) in enumerate(samples):
        img = np.array(Image.open(path).convert("RGB"))
        ins_otsu = otsu_segment(img)
        ins_cp = cellpose_segment(img, cp_model)
        ins_sam = sam3_refine(img, ins_cp, sam3_proc, path)

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"{CLASS_NAMES[cid]}\n{path.name[:20]}",
                                 fontsize=10)
        axes[row, 1].imshow(overlay_instances(img, ins_otsu))
        axes[row, 1].set_title(f"n={len(ins_otsu)}", fontsize=9)
        axes[row, 2].imshow(overlay_instances(img, ins_cp))
        axes[row, 2].set_title(f"n={len(ins_cp)}", fontsize=9)
        axes[row, 3].imshow(overlay_instances(img, ins_sam))
        axes[row, 3].set_title(f"n={len(ins_sam)}", fontsize=9)

        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col] +
                                          (f"\nn={len([ins_otsu, ins_cp, ins_sam][col-1]) if col > 0 else ''}"
                                           if col > 0 else ""), fontsize=11)

        # Re-apply col titles on top row (the n= overrides broke them)
        if row == 0:
            for col, t in enumerate(col_titles):
                axes[0, col].set_title(t, fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "raabin_seg_compare.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved comparison figure: {out}")
    plt.close(fig)

    # Also save per-sample stats
    print("\nPer-sample instance counts (Otsu / Cellpose / CP+SAM3):")
    for (path, cid) in samples:
        img = np.array(Image.open(path).convert("RGB"))
        n_o = len(otsu_segment(img))
        n_c = len(cellpose_segment(img, cp_model))
        print(f"  {CLASS_NAMES[cid]:<12} {path.name[:40]:<42} {n_o:>3} / {n_c:>3}")


if __name__ == "__main__":
    main()
