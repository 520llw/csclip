#!/usr/bin/env python3
"""Generate BALF field-of-view image with segmentation mask and center coordinates."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from skimage.draw import polygon as sk_polygon

ROOT = Path("/home/xut/csclip/cell_datasets/data1_organized")
SPLIT = "train"
STEM = "1483"  # Macrophage + Neutrophil, 2 cells, well-centered
CLASS_NAMES = {0: "CCEC", 1: "RBC", 2: "SEC", 3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
COLORS = {
    0: (255, 200, 100),   # CCEC - yellow
    1: (255, 100, 100),   # RBC - red
    2: (100, 255, 100),   # SEC - green
    3: (255, 100, 255),   # Eosinophil - magenta
    4: (100, 100, 255),   # Neutrophil - blue
    5: (100, 255, 255),   # Lymphocyte - cyan
    6: (255, 180, 50),    # Macrophage - orange
}


def load_polygon_annotations(label_path, img_w, img_h):
    masks = []
    bboxes = []
    cids = []
    centers = []
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        pts = [float(x) for x in parts[1:]]
        xs = [pts[i] * img_w for i in range(0, len(pts), 2)]
        ys = [pts[i] * img_h for i in range(1, len(pts), 2)]
        rr, cc = sk_polygon(ys, xs, shape=(img_h, img_w))
        if len(rr) == 0:
            continue
        mask = np.zeros((img_h, img_w), dtype=bool)
        mask[rr, cc] = True
        masks.append(mask)
        cids.append(cid)
        # bbox
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bboxes.append((x_min, y_min, x_max, y_max))
        # center of mass
        cy = np.mean(rr)
        cx = np.mean(cc)
        centers.append((cx, cy))
    return masks, bboxes, cids, centers


def main():
    img_path = ROOT / "images" / SPLIT / f"{STEM}.jpg"
    lbl_path = ROOT / "labels_polygon" / SPLIT / f"{STEM}.txt"
    out_dir = Path("/home/xut/csclip/paper_materials/balf_field")
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    print(f"Image: {img_path.name}, size={img_w}x{img_h}")

    masks, bboxes, cids, centers = load_polygon_annotations(lbl_path, img_w, img_h)
    print(f"Loaded {len(masks)} polygon annotations")

    # Save original
    img.save(out_dir / f"{STEM}_original.jpg", quality=95)

    # Save mask overlay
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    mask_img = np.zeros((img_h, img_w, 4), dtype=np.uint8)

    for mask, bbox, cid, center in zip(masks, bboxes, cids, centers):
        color = COLORS.get(cid, (200, 200, 200))
        # Fill mask with transparency
        mask_img[mask] = (*color, 120)
        # Draw bbox
        draw.rectangle(bbox, outline=color, width=3)
        # Draw center cross
        cx, cy = center
        r = 6
        draw.line([(cx-r, cy), (cx+r, cy)], fill=(255, 0, 0), width=2)
        draw.line([(cx, cy-r), (cx, cy+r)], fill=(255, 0, 0), width=2)
        # Label
        label = f"{CLASS_NAMES.get(cid, '?')} ({int(cx)}, {int(cy)})"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        # text background
        text_x, text_y = int(bbox[0]), int(bbox[1]) - 20
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        draw.rectangle([text_x, text_y, text_x + tw + 4, text_y + th + 2], fill=(0, 0, 0, 180))
        draw.text((text_x + 2, text_y), label, fill=(255, 255, 255), font=font)

    # Composite mask
    mask_pil = Image.fromarray(mask_img, "RGBA")
    composite = Image.alpha_composite(img.convert("RGBA"), mask_pil)
    composite.save(out_dir / f"{STEM}_mask_overlay.png")

    # Also save mask overlay with bbox/center on top
    final = Image.alpha_composite(composite, overlay.convert("RGBA"))
    final.save(out_dir / f"{STEM}_annotated.png")

    # Save binary mask (single channel, class-colored for visibility)
    mask_color = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for mask, cid in zip(masks, cids):
        color = COLORS.get(cid, (200, 200, 200))
        mask_color[mask] = color
    Image.fromarray(mask_color).save(out_dir / f"{STEM}_mask_only.png")

    # Save center coordinates as text
    with open(out_dir / f"{STEM}_cell_centers.txt", "w") as f:
        f.write(f"Image: {STEM}.jpg ({img_w}x{img_h})\n")
        f.write("cell_id,class_name,class_id,center_x,center_y,bbox_x1,bbox_y1,bbox_x2,bbox_y2\n")
        for idx, (cid, center, bbox) in enumerate(zip(cids, centers, bboxes)):
            f.write(f"{idx},{CLASS_NAMES.get(cid, '?')},{cid},{center[0]:.1f},{center[1]:.1f},"
                    f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}\n")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
