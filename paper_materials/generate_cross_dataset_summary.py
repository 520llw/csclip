#!/usr/bin/env python3
"""Combine best PAMSR examples from data1, data2, MultiCenter into one figure."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path("/home/xut/csclip/paper_materials/pamsr_real")
OUT = ROOT / "pamsr_cross_dataset_best_cases.png"

# Best example from each dataset
CASES = [
    {
        "path": ROOT / "pamsr_real_data1_1131.png",
        "title": "Dataset: data1 (Clinical BALF, 853×640)",
        "stats": "Single: TP=3 FP=0 FN=1  →  PAMSR: TP=4 FP=0 FN=0  |  Rescued=1 cell",
    },
    {
        "path": ROOT / "pamsr_real_2022-06-10-14-09-32-87353.png",
        "title": "Dataset: data2 (Clinical BALF, 2048×1536)",
        "stats": "Single: TP=56 FP=13 FN=21  →  PAMSR: TP=59 FP=13 FN=18  |  Rescued=3 cells",
    },
    {
        "path": ROOT / "pamsr_real_multicenter_1222.png",
        "title": "Dataset: MultiCenter (Multi-site BALF, 853×640)",
        "stats": "Single: TP=10 FP=1 FN=5  →  PAMSR: TP=12 FP=2 FN=3  |  Rescued=2 cells",
    },
]

# Target width for each row (scale down if needed)
TARGET_W = 2400

images = []
for case in CASES:
    img = Image.open(case["path"])
    # Resize if too wide
    w, h = img.size
    if w > TARGET_W:
        scale = TARGET_W / w
        img = img.resize((TARGET_W, int(h * scale)), Image.LANCZOS)
        w, h = img.size
    images.append((img, w, h, case))

# Compute layout
max_w = max(i[1] for i in images)
total_h = sum(i[2] for i in images) + len(images) * 80  # 80px per row for titles

# Create canvas
canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    font_stats = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except:
    font_title = ImageFont.load_default()
    font_stats = ImageFont.load_default()

y_offset = 20
for img, w, h, case in images:
    # Center horizontally
    x_offset = (max_w - w) // 2
    canvas.paste(img, (x_offset, y_offset + 50))
    
    # Title
    draw.text((x_offset + 10, y_offset), case["title"], fill=(30, 30, 30), font=font_title)
    # Stats
    draw.text((x_offset + 10, y_offset + h + 55), case["stats"], fill=(80, 80, 80), font=font_stats)
    
    y_offset += h + 80

# Crop to actual content
bbox = canvas.getbbox()
canvas = canvas.crop((0, 0, max_w, y_offset + 10))

# Add main figure title at top
title_canvas = Image.new("RGB", (canvas.width, canvas.height + 60), (255, 255, 255))
title_canvas.paste(canvas, (0, 60))
tdraw = ImageDraw.Draw(title_canvas)
try:
    font_main = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
except:
    font_main = ImageFont.load_default()

title_text = "Fig. PAMSR Qualitative Results Across Three Datasets (Real CellposeSAM Runs)"
tdraw.text((20, 15), title_text, fill=(20, 20, 20), font=font_main)

# Add WBC note at bottom
note_text = "WBC-Seg (external blood smear): CellposeSAM baseline too low (TP<10/146) for per-image rescue demo; dataset-level Precision↑0.025, Recall↑0.028."
try:
    font_note = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 16)
except:
    font_note = ImageFont.load_default()
tdraw.text((20, title_canvas.height - 30), note_text, fill=(120, 120, 120), font=font_note)

title_canvas.save(OUT, dpi=(300, 300))
print(f"Saved: {OUT} ({title_canvas.width}×{title_canvas.height})")
