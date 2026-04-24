#!/usr/bin/env python3
"""Generate a single summary figure combining all 4 PAMSR comparisons."""
from PIL import Image
from pathlib import Path

OUT_DIR = Path("/home/xut/csclip/paper_materials/pamsr")

panels = []
for stem in ["1016", "1078", "111", "1178"]:
    p = OUT_DIR / f"pamsr_compare_{stem}.png"
    panels.append(Image.open(p))

# Stack vertically with small gap
gap = 20
w = max(p.width for p in panels)
h = sum(p.height for p in panels) + gap * (len(panels) - 1)
summary = Image.new("RGB", (w, h), (255, 255, 255))

y = 0
for p in panels:
    summary.paste(p, ((w - p.width) // 2, y))
    y += p.height + gap

summary.save(OUT_DIR / "pamsr_summary_all_4cases.png", dpi=(300, 300))
print(f"Saved PAMSR summary: {OUT_DIR / 'pamsr_summary_all_4cases.png'}")
