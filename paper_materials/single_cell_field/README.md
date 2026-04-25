# BALF Field-of-View with Isolated Target Cell

> **Image**: `balf_field.png` (2048 × 1536 px)  
> **Source**: `data2_organized/images/train/2022-06-10-14-24-58-44011_2048-1536.png`  
> **Dataset**: data2 (main BALF dataset, 180 images, 7 classes)

---

## Target Cell

| Property | Value |
|----------|-------|
| **Class** | Macrophage (class_id = 6) |
| **BBox (relative)** | `cx=0.439209, cy=0.321940, bw=0.162598, bh=0.215495` |
| **BBox (pixels)** | `(732, 328, 1066, 660)` — 334 × 332 px |
| **Left margin** | 732 px (35.7 %) |
| **Right margin** | 982 px (48.0 %) |
| **Top margin** | 328 px (21.4 %) |
| **Bottom margin** | 876 px (57.0 %) |

All margins ≥ 20 % of image dimension. The nearest neighboring cell is 0.332 image-widths away (cell #4, another Macrophage at upper-right), so there is no visual overlap or crowding.

---

## Why this image

- **From data2** — the paper's primary evaluation dataset.
- **High resolution** — 2048 × 1536, enough for crisp figure rendering.
- **Clean context** — the target Macrophage sits in an open region with plenty of background, making it ideal for:
  - Cell-vs-context crop illustrations
  - Attention-map overlays
  - Bounding-box / segmentation mask visualizations

---

## Other cells in the same field

| # | Class | Relative BBox | Distance from target |
|---|-------|---------------|----------------------|
| 0 | Eosinophil (3) | (0.118, 0.187, 0.119, 0.176) | 0.348 |
| 1 | Lymphocyte (5) | (0.818, 0.392, 0.049, 0.081) | 0.385 |
| 2 | Neutrophil (4) | (0.960, 0.527, 0.078, 0.128) | 0.560 |
| 4 | Macrophage (6) | (0.640, 0.586, 0.209, 0.313) | 0.332 |
| 5 | Macrophage (6) | (0.924, 0.269, 0.152, 0.234) | 0.488 |

*(Target is cell #3 in the label file.)*

---

## Usage

```python
from PIL import Image
import matplotlib.patches as patches

img = Image.open("balf_field.png")
fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
ax.imshow(img)

# Target Macrophage bbox
bbox = (732, 328, 1066, 660)  # x1, y1, x2, y2
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                         linewidth=2, edgecolor='lime', facecolor='none')
ax.add_patch(rect)
ax.set_title("BALF Field — Isolated Macrophage (class 6)")
plt.axis('off')
plt.tight_layout()
plt.show()
```
