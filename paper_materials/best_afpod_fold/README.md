# Best AFP-OD Fold: Eosinophil vs Neutrophil (seed=456)

> **Purpose**: Feature arrays for a fold where AFP-OD **outperforms** baseline prototype.
> **Selected because**: across all 5 seeds × 6 class pairs, this fold gives the largest AFP-OD gain (+2.62 pp macro-F1).

---

## Files

| File | Shape | Description |
|------|-------|-------------|
| `support_E.npy` | (10, 512) | 10-shot support for **Eosinophil** (class_id = 3) |
| `support_N.npy` | (10, 512) | 10-shot support for **Neutrophil** (class_id = 4) |
| `query_E.npy` | (114, 512) | Validation queries, true **Eosinophil** |
| `query_N.npy` | (127, 512) | Validation queries, true **Neutrophil** |

- **Feature dimension D = 512**: BiomedCLIP raw output, unit-norm.
- **Support**: random 10 samples from `train` split, `seed = 456`.
- **Query**: all corresponding cells from `val` split.

---

## Why this fold

| Metric | Baseline (prototype, sample-cov) | AFP-OD (LW + α=0.2) | Δ |
|--------|----------------------------------|---------------------|---|
| Macro-F1 | **0.1567** | **0.1829** | **+0.0262** |

This is the **only fold** in the full 5-seed × 6-pair sweep where AFP-OD delivers a clear win.  
All other folds show either a small loss or near-zero change, confirming that AFP-OD's value is **class-pair and fold-dependent** — exactly what the paper's multi-fold ablation table already shows.

---

## Extraction pipeline

Same as `paper_materials/m_vs_n_fold/`:
1. Cell crop (bbox + 10 % margin, bg=128)
2. Context crop (bbox + 30 % margin)
3. BiomedCLIP `encode_image`, L2-normalized
4. Fusion: `0.85 * cell + 0.15 * context`, L2-normalized

Full script: `experiments/find_best_afpod_fold.py`

---

## Usage

```python
import numpy as np

support_E = np.load("support_E.npy")   # (10, 512)
support_N = np.load("support_N.npy")   # (10, 512)
query_E   = np.load("query_E.npy")     # (114, 512)
query_N   = np.load("query_N.npy")     # (127, 512)
```
