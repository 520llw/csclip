# Figure 3 — 6-Panel Mosaic Materials

> **Purpose**: Source images and data for the 6-panel workflow figure (input → segmentation → features → classification → HITL → output).
> **Panel size**: ~93 × 56 pt (388 × 233 px @ 300 DPI), aspect ratio 1.66 : 1.

---

## Panel 1 — Input: BALF Microscope Field

| Property | Value |
|----------|-------|
| **File** | `panel1_balf_field_crop.jpg` |
| **Source** | `data1_organized/images/train/1483.jpg` (853×640) |
| **Crop** | Center region, 450×400 px |
| **Content** | Real BALF field with 5–8 visible cells, unstained background |

**Why this image**: data1 image has larger cells than data2 (853×640 vs 2048×1536), making individual cells visually identifiable even after downsizing to 388 px width.

---

## Panel 2 — Segmentation: Cellpose-SAM + PAMSR Output

| Property | Value |
|----------|-------|
| **File** | `panel2_pamsr_column.png` |
| **Source** | `pamsr_real/pamsr_real_data1_1131.png` (3412×710) |
| **Crop** | Column (c) — PAMSR (green contours), 853×710 px |
| **Content** | Sparse BALF field with PAMSR-rescued segmentation |

**Why this image**: data1_1131 is the sparsest of the three candidates, ensuring readability at small panel size. One cell is clearly rescued (green contour where single-scale cyan misses it).

---

## Panel 3 — Features: Feature Representation

| Property | Value |
|----------|-------|
| **File** | `data2_val_tsne.csv` |
| **Shape** | 1316 rows × 3 columns |
| **Columns** | `x`, `y`, `class` |
| **Method** | t-SNE (perplexity=30, iter=1000) on BiomedCLIP 512-D features |

**Classes**: 3=Eosinophil, 4=Neutrophil, 5=Lymphocyte, 6=Macrophage

Use this CSV to render a scatter plot with class-colored clusters.

---

## Panel 4 — Classification: Decision Evidence

| Property | Value |
|----------|-------|
| **File** | `confusion_matrix_normalized_seed42.png` |
| **Source** | `confusion/confusion_matrix_normalized_seed42.png` |
| **Content** | Row-normalized confusion matrix for AFP-OD (seed=42) |

**Why normalized**: Row normalization preserves the diagonal-dominant pattern across classes with unequal sample sizes, making the correct-class pattern readable at small size.

---

## Panel 5 — HITL Review: Interface Screenshot

| Property | Value |
|----------|-------|
| **File** | `panel5_hitl_canvas.png` |
| **Source** | `ui_mockup/balf_analyzer_real_ui.png` (1871×906) |
| **Crop** | Central canvas region, 1010×688 px |
| **Content** | Real BALF-Analyzer UI showing cell bbox annotations + class labels + review queue |

**Why this crop**: The full UI becomes unreadable at 388 px width. The canvas crop focuses on the annotated cells and label list, which carries the HITL narrative.

---

## Panel 6 — Output: Classified Cells

| Property | Value |
|----------|-------|
| **File** | `panel6_classified_cells_final.png` |
| **Source** | `panel6_classified_cells.png` (2048×1536) |
| **Crop** | 1000×800 px region with high class diversity |
| **Content** | BALF field with each cell outlined in its **predicted** class color |

**Color legend**:
| Class | Color | Outline |
|-------|-------|---------|
| Eosinophil | Deep Orange | 🔶 |
| Neutrophil | Dodger Blue | 🔷 |
| Lymphocyte | Lime Green | 🟢 |
| Macrophage | Medium Purple | 🟣 |

**Image**: `data2_organized/val/2022-06-10-14-34-55-71733_2048-1536.png` (61 cells, all 4 classes present).

**Predictions**: From `confusion/per_sample_predictions_seed42.csv` (indices 592–652).

---

## Quick Reference

| Panel | File | Dimensions | Ready to crop? |
|-------|------|-----------|----------------|
| 1 | `panel1_balf_field_crop.jpg` | 450×400 | ✅ Yes |
| 2 | `panel2_pamsr_column.png` | 853×710 | ✅ Yes |
| 3 | `data2_val_tsne.csv` | 1316 rows | ✅ Render from CSV |
| 4 | `confusion_matrix_normalized_seed42.png` | ~1200×900 | ✅ Yes |
| 5 | `panel5_hitl_canvas.png` | 1010×688 | ✅ Yes |
| 6 | `panel6_classified_cells_final.png` | 1000×800 | ✅ Yes |

All files are pre-cropped or pre-computed to the correct aspect ratio (~1.66:1) and can be directly scaled to 388×233 px for the final figure.
