# AGENTS.md — BALF Cell Annotation System

Medical image annotation web app with AI segmentation. FastAPI backend + vanilla JS frontend.

## Critical: How to Run

**ALWAYS run from parent directory (`D:\VM_share`), NOT from inside `labeling_tool/`:**

```bash
cd D:\VM_share
conda activate research_assistant
python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** `main.py` has **no `if __name__ == "__main__"` block** — always use uvicorn.

Or use the batch script: `D:\VM_share\labeling_tool\run_server.bat`

Then open: http://localhost:8000

## Environment

- **Conda env:** `research_assistant` (Python 3.11.15)
- **GPU:** CUDA 12.8 (PyTorch 2.10.0+cu128)
- **No build system:** Plain Python modules + static JS files (no pyproject.toml, package.json, etc.)

## Architecture

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app (~2600 lines). Image discovery, annotation I/O, API endpoints. |
| `model.py` | SAM3 wrapper (box/text/point prompts) |
| `cellpose_utils.py` | CellposeSAM segmentation |
| `fewshot_biomedclip.py` | Few-shot BiomedCLIP classifier |
| `hybrid_classifier.py` | Hybrid adaptive classifier |
| `paths.py` | Weight file path resolution |
| `static/` | Frontend: index.html (Tailwind), app.js (Konva.js canvas). **No build step.** |
| `weights/` | sam3.pt (~3.4GB), biomedclip/ weights |

## External Dependencies (MedSAM Project)

The parent directory must contain (via `MEDSAM_ROOT` env var, defaults to parent of `labeling_tool/`):
- `sam3/` — SAM3 model source
- `biomedclip_fewshot_support_experiment/` — few-shot modules
- `biomedclip_zeroshot_cell_classify/` — classification modules
- `biomedclip_hybrid_adaptive_classifier/` — hybrid classifier

**Server starts without these**, but ML endpoints return 503 errors.

## Key Conventions

- **YOLO format:** `.txt` files alongside images. Polygon: `class_id x1 y1 x2 y2 ...` (normalized). BBox: `class_id cx cy w h`.
- **Dataset discovery:** Scans `MEDSAM_ROOT` recursively for `data*.yaml` files.
- **Lazy model loading:** Models load on first use (10–30s cold start). GPU OOM → automatic CPU fallback.
- **Batch jobs:** Cellpose batch uses start → poll → commit pattern.

## Adding Endpoints

1. Define Pydantic model near existing ones (~lines 390–550 in main.py)
2. Place route with logically related endpoints
3. Use helpers: `_get_group()`, `_get_label_set()`, `_get_dirs()`
4. Return plain `dict` objects

## Testing

**No formal test suite.** Verify manually:

```bash
python check_environment.py  # Dependency check
curl http://localhost:8000/api/groups  # API test
```

## Common Issues

| Problem | Cause |
|---------|-------|
| `ModuleNotFoundError: labeling_tool` | Running from wrong directory. Must be in `D:\VM_share` |
| `ModuleNotFoundError: biomedclip_*` | Missing MedSAM project in parent directory |
| Port 8000 in use | Use `--port 8001` |
| CUDA OOM | Automatic CPU fallback; first request may be slow |
| Static files cached | Hard-refresh browser (Ctrl+F5) |

## File-to-Concern Quick Ref

- Server/routing: `main.py`
- SAM3 segmentation: `model.py`
- Cellpose: `cellpose_utils.py`
- Few-shot classify: `fewshot_biomedclip.py` + `default_supports.json`
- Hybrid classify: `hybrid_classifier.py` + `default_supports.json`
- Frontend: `static/index.html`, `static/app.js`
- Paths: `paths.py`
- Launch: `run_server.bat`, `run_server.sh`

---

*Last updated: April 7, 2026*
