#!/usr/bin/env python3
"""Generate a mockup screenshot of the FastAPI labeling tool review queue / SAM3 prompt interface."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

OUT_DIR = Path("/home/xut/csclip/paper_materials/ui_mockup")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load a real BALF thumbnail for the cell list
THUMB_PATH = Path("/home/xut/csclip/cell_datasets/data1_organized/images/val/1016.jpg")
thumb = Image.open(THUMB_PATH).convert("RGB").resize((80, 60), Image.LANCZOS)

W, H = 1600, 1000

# Create canvas with subtle gradient background
img = Image.new("RGB", (W, H), (245, 247, 250))
draw = ImageDraw.Draw(img)

# Header bar
draw.rectangle([0, 0, W, 56], fill=(37, 99, 235))  # blue header

try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    font_nav = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    font_mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
except Exception:
    font_title = ImageFont.load_default()
    font_nav = ImageFont.load_default()
    font_body = ImageFont.load_default()
    font_bold = ImageFont.load_default()
    font_small = ImageFont.load_default()
    font_mono = ImageFont.load_default()

# Title
draw.text((20, 16), "BALF Cell Labeling Tool  —  Review Queue (SAM3 Prompt Mode)", fill=(255, 255, 255), font=font_title)

# Nav tabs
tabs = ["Dataset", "Auto-Segment", "Review Queue", "SAM3 Prompt", "Export"]
tx = W - 620
for t in tabs:
    color = (255, 255, 255) if t == "Review Queue" else (200, 220, 255)
    draw.text((tx, 20), t, fill=color, font=font_nav)
    tx += 118

# Left sidebar: cell list
sidebar_w = 320
draw.rectangle([0, 56, sidebar_w, H], fill=(255, 255, 255))
draw.line([(sidebar_w, 56), (sidebar_w, H)], fill=(220, 224, 230), width=2)

draw.text((16, 72), "Pending Review (12)", fill=(30, 30, 30), font=font_bold)
draw.text((16, 100), "Confirmed (45)  ·  Rejected (3)", fill=(100, 100, 100), font=font_small)

# List items
list_y = 140
status_colors = {
    "pending": (250, 204, 21),   # yellow
    "review": (239, 68, 68),     # red
    "confirmed": (34, 197, 94),  # green
}

cells = [
    ("1016_0", "Macrophage", "pending", "Single-scale miss"),
    ("1078_1", "RBC", "review", "Fragmented"),
    ("111_0", "Macrophage", "pending", "Over-merge risk"),
    ("1178_1", "Neutrophil", "confirmed", "OK"),
    ("1204_0", "Eosinophil", "review", "Low confidence"),
    ("1483_0", "Macrophage", "pending", "—"),
    ("1483_1", "Neutrophil", "confirmed", "—"),
]

for idx, (cid, cls, status, note) in enumerate(cells):
    y = list_y + idx * 72
    # hover highlight for first item
    if idx == 0:
        draw.rectangle([8, y, sidebar_w - 8, y + 64], fill=(239, 246, 255))
        draw.rectangle([8, y, 10, y + 64], fill=(37, 99, 235), width=3)
    # thumbnail
    img.paste(thumb, (20, y + 2))
    # text
    draw.text((110, y + 6), f"{cid}  ·  {cls}", fill=(30, 30, 30), font=font_body)
    # status badge
    sc = status_colors[status]
    bw = 80 if status == "confirmed" else (60 if status == "pending" else 55)
    draw.rounded_rectangle([110, y + 30, 110 + bw, y + 48], radius=10, fill=sc)
    label = {"pending": "Pending", "review": "Review", "confirmed": "Confirmed"}[status]
    draw.text((114, y + 31), label, fill=(255, 255, 255), font=font_small)
    draw.text((110 + bw + 10, y + 32), note, fill=(100, 100, 100), font=font_small)
    # separator
    draw.line([(20, y + 68), (sidebar_w - 20, y + 68)], fill=(230, 230, 230), width=1)

# Main content area: image viewer + SAM3 prompt panel
main_x = sidebar_w + 20

# Image viewer frame
viewer_y = 80
viewer_w = 720
viewer_h = 540
draw.rounded_rectangle([main_x, viewer_y, main_x + viewer_w, viewer_y + viewer_h], radius=8, outline=(200, 200, 200), width=2)

# Placeholder image with grid
placeholder = Image.new("RGB", (viewer_w - 4, viewer_h - 4), (240, 240, 240))
ph_draw = ImageDraw.Draw(placeholder)
for gx in range(0, viewer_w, 40):
    ph_draw.line([(gx, 0), (gx, viewer_h)], fill=(230, 230, 230), width=1)
for gy in range(0, viewer_h, 40):
    ph_draw.line([(0, gy), (viewer_w, gy)], fill=(230, 230, 230), width=1)
ph_draw.text((viewer_w // 2 - 180, viewer_h // 2 - 10), "[BALF Microscopy Image — 2048×1536 px]", fill=(150, 150, 150), font=font_body)
ph_draw.text((viewer_w // 2 - 120, viewer_h // 2 + 15), "Drag to pan  ·  Scroll to zoom", fill=(150, 150, 150), font=font_small)
img.paste(placeholder, (main_x + 2, viewer_y + 2))

# Overlay: SAM3 prompt markers on the placeholder
marker_positions = [(220, 180), (480, 320), (350, 420)]
marker_labels = ["P1", "P2", "P3"]
for (mx, my), ml in zip(marker_positions, marker_labels):
    abs_x = main_x + 2 + mx
    abs_y = viewer_y + 2 + my
    # crosshair
    draw.line([(abs_x - 15, abs_y), (abs_x + 15, abs_y)], fill=(239, 68, 68), width=2)
    draw.line([(abs_x, abs_y - 15), (abs_x, abs_y + 15)], fill=(239, 68, 68), width=2)
    # circle
    draw.ellipse([abs_x - 12, abs_y - 12, abs_x + 12, abs_y + 12], outline=(239, 68, 68), width=2)
    # label
    draw.rounded_rectangle([abs_x + 10, abs_y - 22, abs_x + 34, abs_y - 2], radius=4, fill=(239, 68, 68))
    draw.text((abs_x + 14, abs_y - 20), ml, fill=(255, 255, 255), font=font_small)

# Toolbar below image
tool_y = viewer_y + viewer_h + 12
tools = ["Pan", "Zoom", "Positive Point", "Negative Point", "Box Prompt", "Auto-mask", "Clear"]
tx = main_x
for t in tools:
    active = t == "Positive Point"
    bg = (37, 99, 235) if active else (255, 255, 255)
    fg = (255, 255, 255) if active else (50, 50, 50)
    border = (37, 99, 235) if active else (200, 200, 200)
    tw = 110 if t == "Positive Point" else (90 if t in ["Negative Point", "Auto-mask"] else 60)
    draw.rounded_rectangle([tx, tool_y, tx + tw, tool_y + 32], radius=6, outline=border, fill=bg, width=1)
    draw.text((tx + 8, tool_y + 7), t, fill=fg, font=font_small)
    tx += tw + 8

# Right panel: SAM3 / Classification panel
right_x = main_x + viewer_w + 20
right_w = W - right_x - 20
draw.rounded_rectangle([right_x, viewer_y, right_x + right_w, viewer_y + 420], radius=8, outline=(200, 200, 200), width=2)
draw.text((right_x + 14, viewer_y + 14), "SAM3 Prompt & Classification", fill=(30, 30, 30), font=font_bold)

# Prompt list
prompt_y = viewer_y + 50
prompts = [
    ("P1", "Positive point", "(312, 452)", "—"),
    ("P2", "Positive point", "(628, 741)", "—"),
    ("P3", "Negative point", "(471, 852)", "exclude debris"),
]
draw.text((right_x + 14, prompt_y), "Active Prompts", fill=(80, 80, 80), font=font_small)
prompt_y += 22
for pid, ptype, coord, note in prompts:
    draw.text((right_x + 20, prompt_y), f"• {pid}: {ptype}  {coord}  {note}", fill=(50, 50, 50), font=font_body)
    prompt_y += 22

# Action buttons
btn_y = prompt_y + 16
draw.rounded_rectangle([right_x + 14, btn_y, right_x + 130, btn_y + 34], radius=6, fill=(37, 99, 235))
draw.text((right_x + 24, btn_y + 7), "Run SAM3", fill=(255, 255, 255), font=font_body)
draw.rounded_rectangle([right_x + 146, btn_y, right_x + 260, btn_y + 34], radius=6, outline=(200, 200, 200), fill=(255, 255, 255))
draw.text((right_x + 156, btn_y + 7), "Undo", fill=(50, 50, 50), font=font_body)

# Classification result section
res_y = btn_y + 60
draw.line([(right_x + 14, res_y), (right_x + right_w - 14, res_y)], fill=(220, 220, 220), width=1)
draw.text((right_x + 14, res_y + 10), "Classification Result", fill=(30, 30, 30), font=font_bold)

pred_classes = [
    ("Macrophage", 0.87, "High"),
    ("Neutrophil", 0.09, "Low"),
    ("Lymphocyte", 0.03, "Low"),
    ("Eosinophil", 0.01, "Low"),
]
bar_y = res_y + 42
for cls, score, conf in pred_classes:
    bar_w = int(score * (right_w - 120))
    draw.text((right_x + 14, bar_y), cls, fill=(50, 50, 50), font=font_body)
    draw.rounded_rectangle([right_x + 130, bar_y + 2, right_x + 130 + bar_w, bar_y + 18], radius=4, fill=(37, 99, 235) if score > 0.5 else (200, 200, 200))
    draw.text((right_x + 130 + bar_w + 8, bar_y), f"{score:.2f}", fill=(80, 80, 80), font=font_body)
    bar_y += 26

# Confidence & actions
draw.text((right_x + 14, bar_y + 6), "Confidence: HIGH  |  Morphology check: PASS", fill=(34, 197, 94), font=font_bold)
act_y = bar_y + 36
draw.rounded_rectangle([right_x + 14, act_y, right_x + 100, act_y + 32], radius=6, fill=(34, 197, 94))
draw.text((right_x + 22, act_y + 7), "Confirm", fill=(255, 255, 255), font=font_body)
draw.rounded_rectangle([right_x + 114, act_y, right_x + 200, act_y + 32], radius=6, outline=(200, 200, 200), fill=(255, 255, 255))
draw.text((right_x + 122, act_y + 7), "Reject", fill=(50, 50, 50), font=font_body)
draw.rounded_rectangle([right_x + 214, act_y, right_x + 310, act_y + 32], radius=6, outline=(200, 200, 200), fill=(255, 255, 255))
draw.text((right_x + 222, act_y + 7), "Re-prompt", fill=(50, 50, 50), font=font_body)

# Footer status bar
draw.rectangle([0, H - 30, W, H], fill=(240, 240, 240))
draw.line([(0, H - 30), (W, H - 30)], fill=(200, 200, 200), width=1)
draw.text((16, H - 24), "Connected  ·  BiomedCLIP + Cellpose + SAM3  ·  1316 cached features", fill=(80, 80, 80), font=font_small)

# Save
out_path = OUT_DIR / "fastapi_review_queue_sam3_mockup.png"
img.save(out_path, dpi=(150, 150))
print(f"Saved UI mockup: {out_path}")

# Also create a smaller composite for figure embedding (scale down)
img_small = img.resize((1200, 750), Image.LANCZOS)
img_small.save(OUT_DIR / "fastapi_review_queue_sam3_mockup_1200.png")
print("Saved 1200px version")
