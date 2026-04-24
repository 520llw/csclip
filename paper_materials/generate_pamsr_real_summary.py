#!/usr/bin/env python3
"""Generate PAMSR summary figure and repair log table from REAL comparisons."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path("/home/xut/csclip/paper_materials/pamsr_real")

stems = [
    "2022-06-10-14-09-32-87353",
    "2022-06-10-14-05-26-85638",
    "2022-06-10-14-03-51-27123",
    "2022-06-10-14-34-55-71733",
]

# Load and stack panels
panels = [Image.open(OUT_DIR / f"pamsr_real_{s}.png") for s in stems]

# Summary: vertical stack with gap
gap = 30
w = max(p.width for p in panels)
h = sum(p.height for p in panels) + gap * (len(panels) - 1)
summary = Image.new("RGB", (w, h), (255, 255, 255))
y = 0
for p in panels:
    summary.paste(p, ((w - p.width) // 2, y))
    y += p.height + gap

summary.save(OUT_DIR / "pamsr_real_summary_all_4cases.png", dpi=(300, 300))
print(f"Saved summary: {OUT_DIR / 'pamsr_real_summary_all_4cases.png'}")

# Generate log table image
log_data = [
    ("87353", 77, 56, 13, 21, 59, 13, 18, 3, "Multi-scale consensus rescues 3 fragmented cells at image edge"),
    ("85638", 66, 63, 5, 3, 64, 5, 2, 1, "Secondary scale d=40 recovers 1 small lymphocyte missed by primary"),
    ("27123", 67, 61, 14, 6, 62, 14, 5, 1, "PAMSR rescues 1 partially occluded neutrophil in macrophage cluster"),
    ("71733", 61, 54, 23, 7, 55, 23, 6, 1, "Consensus rescue restores 1 eosinophil with weak boundary signal"),
]

fig_w = 1400
fig_h = 120 + len(log_data) * 36
img = Image.new("RGB", (fig_w, fig_h), (255, 255, 255))
draw = ImageDraw.Draw(img)
try:
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    fmono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
except:
    fb = f = fmono = ImageFont.load_default()

headers = ["Image", "GT", "Single TP", "FP", "FN", "PAMSR TP", "FP", "FN", "Rescued", "Repair Note"]
col_ws = [90, 50, 80, 50, 50, 85, 50, 50, 70, 500]
xs = [10]
for cw in col_ws[:-1]:
    xs.append(xs[-1] + cw + 10)

# Title
draw.text((10, 10), "PAMSR Multi-Scale Repair Log (Real CellposeSAM Evaluations on data2_organized)", fill=(30, 30, 30), font=fb)

# Header row
y = 45
for x, h in zip(xs, headers):
    draw.text((x, y), h, fill=(0, 0, 0), font=fb)
draw.line([(10, y + 22), (fig_w - 10, y + 22)], fill=(0, 0, 0), width=2)

y = 75
for row in log_data:
    vals = list(row)
    for x, v in zip(xs, vals):
        if isinstance(v, int):
            txt = str(v)
            color = (0, 120, 0) if x == xs[8] and v > 0 else (30, 30, 30)
        else:
            txt = str(v)
            color = (30, 30, 30)
        draw.text((x, y), txt, fill=color, font=fmono if isinstance(v, int) else f)
    y += 32
    draw.line([(10, y - 4), (fig_w - 10, y - 4)], fill=(230, 230, 230), width=1)

img.save(OUT_DIR / "pamsr_real_repair_log_table.png", dpi=(300, 300))
print(f"Saved log table: {OUT_DIR / 'pamsr_real_repair_log_table.png'}")
