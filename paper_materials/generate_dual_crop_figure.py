#!/usr/bin/env python3
"""Generate Fig. 2b: Dual-scale crop diagram for BALF-Analyzer paper."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Input image
IMG_PATH = Path("/home/xut/csclip/cell_datasets/data1_organized/images/train/561.jpg")
LBL_PATH = Path("/home/xut/csclip/cell_datasets/data1_organized/labels_polygon/train/561.txt")
OUT_DIR = Path("/home/xut/csclip/paper_materials/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load image
img = Image.open(IMG_PATH)
if img.mode != "RGB":
    img = img.convert("RGB")
W, H = img.size  # 853 x 640

# Parse labels
CLASS_NAMES = {4: "Neutrophil", 6: "Macrophage"}
CLASS_COLORS = {4: (52, 152, 219), 6: (231, 76, 60)}  # Blue, Red

cells = []
for line in open(LBL_PATH):
    parts = line.strip().split()
    if len(parts) < 7:
        continue
    cid = int(parts[0])
    pts = [float(x) for x in parts[1:]]
    xs = [pts[i] * W for i in range(0, len(pts), 2)]
    ys = [pts[i] * H for i in range(1, len(pts), 2)]
    cx = np.mean(xs)
    cy = np.mean(ys)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bw = maxx - minx
    bh = maxy - miny
    cells.append({
        "cid": cid, "cx": cx, "cy": cy, "bw": bw, "bh": bh,
        "minx": minx, "maxx": maxx, "miny": miny, "maxy": maxy,
        "name": CLASS_NAMES.get(cid, f"Class {cid}"),
        "color": CLASS_COLORS.get(cid, (100, 100, 100))
    })

# Sort by y position (top to bottom)
cells.sort(key=lambda c: c["cy"])

# Select the bottom cell (Macrophage) for detailed crop demo
demo_cell = cells[-1]  # Macrophage, larger, more visually distinctive

# Crop calculations
def make_crop(img, cx, cy, bw, bh, scale_factor, label_text, crop_label):
    """Create a crop with the given scale factor (1.1 = 110%, 1.5 = 150%)."""
    half_w = bw * scale_factor / 2
    half_h = bh * scale_factor / 2
    
    left = max(0, int(cx - half_w))
    top = max(0, int(cy - half_h))
    right = min(W, int(cx + half_w))
    bottom = min(H, int(cy + half_h))
    
    crop = img.crop((left, top, right, bottom))
    
    # Draw border and label on the crop
    draw = ImageDraw.Draw(crop)
    color = demo_cell["color"]
    
    # Border color: green for cell crop (1.1), purple for context crop (1.5)
    border_color = (46, 204, 113) if scale_factor < 1.2 else (155, 89, 182)
    draw.rectangle([(0, 0), (crop.size[0]-1, crop.size[1]-1)], outline=border_color, width=4)
    
    # Label at top
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, crop.size[1]//12))
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, crop.size[1]//15))
    except:
        font = ImageFont.load_default()
        font_s = ImageFont.load_default()
    
    # Text background
    bbox = draw.textbbox((0, 0), label_text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.rectangle([(5, 5), (10 + tw, 10 + th + 2)], fill=(255, 255, 255, 200))
    draw.text((7, 5), label_text, fill=color, font=font)
    
    # Scale label at bottom
    bbox2 = draw.textbbox((0, 0), crop_label, font=font_s)
    tw2, th2 = bbox2[2]-bbox2[0], bbox2[3]-bbox2[1]
    draw.rectangle([(5, crop.size[1]-th2-10), (10+tw2, crop.size[1]-5)], fill=(255, 255, 255, 200))
    draw.text((7, crop.size[1]-th2-8), crop_label, fill=(80, 80, 80), font=font_s)
    
    return crop, (left, top, right, bottom)

# Create original image with bounding boxes overlay
img_orig = img.copy()
draw_orig = ImageDraw.Draw(img_orig)
try:
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    font_mid = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font_big = ImageFont.load_default()
    font_mid = ImageFont.load_default()

for c in cells:
    color = c["color"]
    # Original bbox (100%)
    draw_orig.rectangle(
        [(c["minx"], c["miny"]), (c["maxx"], c["maxy"])],
        outline=color, width=2
    )
    # Label
    draw_orig.text((c["minx"], c["miny"]-20), c["name"], fill=color, font=font_mid)
    
    # Center dot
    r = 3
    draw_orig.ellipse([(c["cx"]-r, c["cy"]-r), (c["cx"]+r, c["cy"]+r)], fill=color)

# Highlight demo cell with larger annotations
c = demo_cell
color = c["color"]
# 110% bbox
cell_half_w = c["bw"] * 1.1 / 2
cell_half_h = c["bh"] * 1.1 / 2
draw_orig.rectangle(
    [(c["cx"]-cell_half_w, c["cy"]-cell_half_h), (c["cx"]+cell_half_w, c["cy"]+cell_half_h)],
    outline=(46, 204, 113), width=2  # Green
)
# 150% bbox
ctx_half_w = c["bw"] * 1.5 / 2
ctx_half_h = c["bh"] * 1.5 / 2
draw_orig.rectangle(
    [(c["cx"]-ctx_half_w, c["cy"]-ctx_half_h), (c["cx"]+ctx_half_w, c["cy"]+ctx_half_h)],
    outline=(155, 89, 182), width=2  # Purple
)

# Legend on original
legend_y = 10
legend_items = [
    ("— Original BBox", c["color"]),
    ("— Cell Crop (110%)", (46, 204, 113)),
    ("— Context Crop (150%)", (155, 89, 182)),
]
for text, col in legend_items:
    draw_orig.text((10, legend_y), text, fill=col, font=font_mid)
    legend_y += 18

# Generate crops
cell_crop, cell_coords = make_crop(img, c["cx"], c["cy"], c["bw"], c["bh"], 1.10, c["name"], "Cell Crop (110%)")
ctx_crop, ctx_coords = make_crop(img, c["cx"], c["cy"], c["bw"], c["bh"], 1.50, c["name"], "Context Crop (150%)")

# Resize crops for display (make them similar height)
target_h = 350
cell_crop_r = cell_crop.resize((int(cell_crop.size[0] * target_h / cell_crop.size[1]), target_h), Image.LANCZOS)
ctx_crop_r = ctx_crop.resize((int(ctx_crop.size[0] * target_h / ctx_crop.size[1]), target_h), Image.LANCZOS)

# Build final panel
panel_w = W + 40 + cell_crop_r.size[0] + 40 + ctx_crop_r.size[0] + 40
panel_h = max(H, target_h + 120)
panel = Image.new("RGB", (panel_w, panel_h), (250, 250, 250))

# Title
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except:
    font_title = ImageFont.load_default()
draw_panel = ImageDraw.Draw(panel)
title = "Fig. 2(b): Dual-Scale Cropping Strategy"
draw_panel.text((panel_w//2 - 220, 10), title, fill=(20, 20, 20), font=font_title)

# Paste original (with some top offset for title)
y_offset = 45
panel.paste(img_orig, (20, y_offset))

# Paste crops
x_crop = W + 60
draw_panel.text((x_crop, y_offset - 30), "(b) Extracted Crops", fill=(50, 50, 50), font=font_big)

panel.paste(cell_crop_r, (x_crop, y_offset))
panel.paste(ctx_crop_r, (x_crop + cell_crop_r.size[0] + 40, y_offset))

# Add sub-labels below crops
sub_y = y_offset + target_h + 12
draw_panel.text((x_crop + cell_crop_r.size[0]//2 - 35, sub_y), 
                f"Cell\n({demo_cell['name']})", fill=(46, 204, 113), font=font_mid)
draw_panel.text((x_crop + cell_crop_r.size[0] + 40 + ctx_crop_r.size[0]//2 - 55, sub_y),
                f"Cell + Context\n({demo_cell['name']})", fill=(155, 89, 182), font=font_mid)

# Bottom annotation
ann_y = sub_y + 45
draw_panel.text((20, ann_y), 
    "Each detected instance generates two crops: (1) a tight cell crop at 110% of the bounding box, "
    "and (2) a context crop at 150% that includes surrounding neighborhood. Both are fed into the multi-backbone encoder.",
    fill=(80, 80, 80), font=font_mid)

# Save
out_path = OUT_DIR / "fig2b_dual_scale_crop.png"
panel.save(out_path, dpi=(300, 300))
print(f"Saved: {out_path} ({panel.size[0]}x{panel.size[1]})")
