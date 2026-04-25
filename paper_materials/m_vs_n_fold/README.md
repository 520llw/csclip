# Data2 Fold: Macrophage vs Neutrophil (M vs N) Support & Query Features

> **Purpose**: Binary-classification feature arrays for covariance-estimation visualization (sample covariance, LW shrinkage, AFP-OD).  
> **Dataset**: `data2_organized` (BALF cell images, train/val split).  
> **Model**: Frozen BiomedCLIP image encoder + dual-scale fusion (cell 85 % + context 15 %).  
> **Generated**: 2026-04-25 by `experiments/extract_m_vs_n_fold.py`

---

## Files

| File | Shape | Description |
|------|-------|-------------|
| `support_M.npy` | (10, 512) | 10-shot support features for **Macrophage** (class_id = 6) |
| `support_N.npy` | (10, 512) | 10-shot support features for **Neutrophil** (class_id = 4) |
| `query_M.npy` | (409, 512) | Validation query samples with ground-truth **Macrophage** label |
| `query_N.npy` | (127, 512) | Validation query samples with ground-truth **Neutrophil** label |

- **Feature dimension D = 512**: raw output of BiomedCLIP image encoder (no PCA / no dimensionality reduction).
- **Support selection**: random 10 samples per class from `train` split, `seed = 42`.
- **Query selection**: all cells of the corresponding class from `val` split.

---

## Feature Extraction Details

1. **Cell crop**: bbox + 10 % margin, background filled with gray (128).
2. **Context crop**: bbox + 30 % margin.
3. **Encoding**: each crop is passed through BiomedCLIP `encode_image`, L2-normalized.
4. **Fusion**: `fused = 0.85 * cell_feat + 0.15 * context_feat`, then L2-normalized again.

The exact code is in `experiments/extract_m_vs_n_fold.py` (root of this repository).

---

## Usage Example

```python
import numpy as np

support_M = np.load("support_M.npy")   # (10, 512)
support_N = np.load("support_N.npy")   # (10, 512)
query_M   = np.load("query_M.npy")     # (409, 512)
query_N   = np.load("query_N.npy")     # (127, 512)

# PCA is only for 2-D visualization;
# F1 scores should be computed in the original 512-D space.
```

---

## Citation / Context

These arrays were created for the BALF cell classification project (BiomedCLIP-based few-shot adaptive classifier).  
They are intended for:

- Ellipse drawing from covariance estimates
- Decision-boundary comparison (sample covariance vs Ledoit-Wolf vs AFP-OD)
- PCA scatter plots **with metric strip F1 computed in full 512-D space**

If you use these arrays in a publication, please refer to the main project repository:  
`https://github.com/520llw/csclip`
