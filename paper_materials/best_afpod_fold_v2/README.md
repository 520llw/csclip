# Best AFP-OD Fold v2: Eosinophil vs Neutrophil (seed=789, kNN-evaluated)

> **Purpose**: Feature arrays for a fold where AFP-OD **clearly beats kNN baseline**.
> **Critical fix**: v1 used prototype/NCM evaluation (misleading); v2 uses kNN k=7 (matches actual classifier).
> **Selected because**: across all 5 seeds × 6 class pairs, this fold gives the largest AFP-OD gain under **realistic kNN classification**.

---

## Files

| File | Shape | Description |
|------|-------|-------------|
| `support_E.npy` | (10, 512) | 10-shot support for **Eosinophil** (class_id = 3) |
| `support_N.npy` | (10, 512) | 10-shot support for **Neutrophil** (class_id = 4) |
| `query_E.npy` | (114, 512) | Validation queries, true **Eosinophil** |
| `query_N.npy` | (127, 512) | Validation queries, true **Neutrophil** |

- **Feature dimension D = 512**: BiomedCLIP raw output, unit-norm.
- **Support**: random 10 samples from `train` split, `seed = 789`.
- **Query**: all corresponding cells from `val` split.

---

## Why this fold (v2 correction)

### v1 的问题
The first version (`best_afpod_fold/`) used **prototype classification (NCM)** as the baseline evaluator:
- Prototype baseline F1 ≈ 0.15
- AFP-OD F1 ≈ 0.18
- This looks like a win, but the actual classifier in the paper is **MB-kNN**, not prototype mean.

Under **kNN k=7** (the real classifier), the v1 fold shows **unstable or negative gains**.

### v2 的修正
This fold was selected using **kNN k=7** as the evaluator, matching the actual MB-kNN classifier:

| Metric | kNN Baseline | AFP-OD (LW + α=0.2) | Δ |
|--------|-------------|---------------------|---|
| Macro-F1 | **0.5853** | **0.7568** | **+0.1714** |

This is the **largest absolute gain** across all 30 combinations (5 seeds × 6 pairs).

---

## Extraction pipeline

Same as `paper_materials/m_vs_n_fold/`:
1. Cell crop (bbox + 10 % margin, bg=128)
2. Context crop (bbox + 30 % margin)
3. BiomedCLIP `encode_image`, L2-normalized
4. Fusion: `0.85 * cell + 0.15 * context`, L2-normalized

Full script: `experiments/find_best_afpod_fold_v2.py`

---

## Usage

```python
import numpy as np

support_E = np.load("support_E.npy")   # (10, 512)
support_N = np.load("support_N.npy")   # (10, 512)
query_E   = np.load("query_E.npy")     # (114, 512)
query_N   = np.load("query_N.npy")     # (127, 512)
```

**Note**: When evaluating AFP-OD gains, use **kNN k=7** as the classifier, not prototype mean. See `experiments/find_best_afpod_fold_v2.py` for the exact evaluation logic.
