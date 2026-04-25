# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**csclip** is the BALF (bronchoalveolar lavage fluid) cell segmentation + few-shot classification stack
behind a patent and a CAS Tier-2 paper. Two halves share four root-level Python modules:

- **Research half** — `experiments/` + `paper_materials/`, run on Linux GPU box, conda env `cel`,
  triggered through root-level `run_*.sh` wrappers that redirect output to `/tmp/*_output.txt`.
- **Production half** — `labeling_tool/`, a FastAPI + vanilla-JS/Konva.js web annotation tool that
  serves the same algorithms to a human-in-the-loop labeler.

The bridge between them is `labeling_tool/export_default_supports_from_experiment.py`, which re-runs
the experimental support-selection pipeline (`SHOTS_PER_CLASS=10`, `SEED=42`, `STRATEGY="center_picked"`)
and writes the chosen 40 cells to `labeling_tool/default_supports.json` — the exact format the FastAPI
server reads on startup.

## The four core modules (do-not-break zone)

The project root contains four modules imported by both halves; running
`python test_imports.py` smoke-tests all of them plus SAM3 and the labeling-tool wrappers.

| File | Role |
|---|---|
| `biomedclip_zeroshot_cell_classify.py` | `InstanceInfo` dataclass, `resolve_device("auto")`, `ensure_local_biomedclip_dir` |
| `biomedclip_fewshot_support_experiment.py` | Dual-scale BiomedCLIP encoding: masked cell crop (margin 0.15, bg=128) at weight 0.90 + unmasked context crop (margin 0.30) at weight 0.10, fused & L2-normed |
| `biomedclip_query_adaptive_classifier.py` | 12-D morphology vector (log_area, log_perimeter, circularity, aspect_ratio, solidity, mean R/G/B, std_intensity, eccentricity, extent, equiv_diameter); `SupportRecord`/`QueryRecord`/`SupportCandidate` dataclasses |
| `biomedclip_hybrid_adaptive_classifier.py` | Margin-aware fusion scoring: `final = global_image_weight·proto + global_text_weight·text + adaptive_scale·(adaptive_image_weight·support_affinity)`. `adaptive_scale` ramps between `adaptive_scale_max` and `adaptive_scale_min` based on top-1/top-2 margin (`margin_low`=0.03 → `margin_high`=0.12) |

`labeling_tool/AGENTS.md` and the root `AGENTS.md` flag these as the highest-priority "do not break"
zone. After editing any of them, **always run `python test_imports.py` first** before launching
experiments or the web server.

## Environments

| Use | Conda env | Python | Notes |
|---|---|---|---|
| Experiments / research | `cel` (`/data/software/mamba/envs/cel/bin/python`) | 3.9.21 | PyTorch 2.5.1 + CUDA 12.x. All `run_*.sh` invoke this absolute path. |
| Web tool (`labeling_tool/`) | `research_assistant` | 3.11.15 | PyTorch 2.10.0+cu128. Originally on Windows `D:\VM_share`; FastAPI ≤ 0.128.x for Py 3.9 compatibility. |

`MEDSAM_ROOT` env var overrides project root resolution (see `labeling_tool/paths.py`); defaults to
parent of `labeling_tool/`. The repo has hardcoded `/home/xut/csclip` paths in many `experiments/*.py`
scripts and the `run_*.sh` wrappers — **adjust or `cd` accordingly when running on a different
checkout location**.

## Common commands

### Running experiments

All experiments are launched by root-level `run_*.sh` wrappers that `cd /home/xut/csclip`, call
the `cel` env's Python, and redirect stdout+stderr to `/tmp/<name>_output.txt`. **Tail those files
to read results — the wrappers do not echo to your terminal.**

```bash
python test_imports.py         # smoke test the 4 core modules + SAM3 + wrappers

bash run_baseline.sh           # → /tmp/baseline_output.txt + experiments/baseline_results.json
bash run_10shot.sh             # 5 seeds × ~14 strategies → ten_shot_results.json
bash run_10shot_v2.sh
bash run_ablation.sh           # A baseline → B +text → C +morph → D +CoOp → E full
bash run_advanced.sh
bash run_diagnosis.sh
bash run_knn.sh
bash run_optimize.sh
bash run_cellpose_test.sh      # CellposeSAM evaluation
bash run_cellpose_bench.sh     # diameter / cellprob_threshold sweep
```

To re-bake the production support set from the latest winning experiment:

```bash
python labeling_tool/export_default_supports_from_experiment.py
# → labeling_tool/default_supports.json (40 entries, 10 per class)
```

### Pre-extracting features (one-time per backbone)

`experiments/extract_features.py` (and its variants `data1_extract_features.py`,
`dinobloom_extract.py`, `phikon_extract.py`, `pbc_extract_features.py`, …) writes `.npz` caches
to `experiments/feature_cache/` keyed `{model}_{dataset}_{split}.npz` with arrays
`feats`/`morphs`/`labels`. Downstream classifier scripts (e.g. `ensemble_final.py`) load these
caches instead of re-encoding images.

### Running the web tool

There is **no `if __name__ == "__main__"` block** in `labeling_tool/main.py`; always launch via
uvicorn from the project root (one level above `labeling_tool/`):

```bash
conda activate research_assistant
python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000
```

Or use `labeling_tool/run_server.sh` / `run_server.bat`. There is no formal test suite — verify
manually with `python labeling_tool/check_environment.py` and `curl http://localhost:8000/api/groups`.

## Architecture: the bridge

```
experiments/  ──► 4 core modules ──► labeling_tool/
   │                                       │
   │ pre-extracts features               loads default_supports.json on /api/supports
   │ runs strategy comparisons           wraps prepare_classifier / predict_annotations
   │ writes baseline_results.json        in /api/fewshot/* and /api/hybrid/*
   │                                       │
   └─► export_default_supports_from_experiment.py ─► default_supports.json
       (re-uses the experiment's exact selection pipeline)
```

### Web request flow (production prediction loop)

1. `/api/predict_batch` — SAM3 box-prompt batch (image set once, iterates many bboxes; OOM →
   automatic CPU fallback) **or** `/api/cellpose_segment` / `/api/cellpose_batch/*` (async
   start → status → preview → commit pattern; mask polygons sit in `pending_outputs` until
   committed).
2. `/api/hybrid/predict_current` with `support_items` from `default_supports.json` — returns
   per-instance `pred_class_id` + `confidence`. Uses `BEST_HYBRID_WEIGHTS` (the locked-in
   experiment finding: `global_image_weight=1.0, global_text_weight=0.0, adaptive_scale_max=0.25,
   margin_low=0.03, margin_high=0.12, final_temperature=0.03`). Optional `PairwiseSizeRefiner`
   (off by default) nudges between top-1/top-2 only when their morphology size separation ≥ 0.3 σ.
3. Polygons render onto the Konva canvas; user accepts/edits, then `POST /api/save`:
   - Backs up old `.txt` to `.backup/<name>.<timestamp>.bak` (cleanup keeps last N)
   - Writes YOLO polygon lines `class_id x1 y1 x2 y2 …` (normalized 0–1)
   - `database.save_annotation_version(...)` — full snapshot, ≤ 50 per image
   - `database.record_daily_stat(...)`, `database.log_action(...)` — audit row in SQLite WAL

### Support-set lifecycle

`/api/supports` resolves in priority order: per-dataset `<dataset>/.fewshot_supports.json` →
bundled `labeling_tool/default_supports.json`. Each support is keyed by
`(filename, subset, annotation_uid OR rounded-points signature)`.
`_validate_supports_against_labels` calls `_rebind_support_against_annotations` to re-anchor
supports to new geometry whenever a label file changes — without this, edits silently invalidate
the few-shot pool.

## Data format

YOLO polygon `.txt` line: `class_id x1 y1 x2 y2 ...` (normalized 0–1).
YOLO bbox: `class_id cx cy w h` (normalized 0–1).
`labeling_tool/main.py` discovers image groups by recursively scanning `MEDSAM_ROOT` for
`data*.yaml` configs. Class IDs in BALF scope: `3 Eosinophil, 4 Neutrophil, 5 Lymphocyte,
6 Macrophage` (data2 has 4 effective classes; distribution is highly skewed —
Lymph 54.9 / Macro 31.1 / Neut 9.7 / **Eos 4.4 %**).

Datasets layout (under `cell_datasets/`, gitignored):

```
data2_organized/    # 180 imgs (144 train + 36 val), main experimental set
├── images/{train,val}/*.png
└── labels_polygon/{train,val}/*.txt
data1_organized/    # 2,698 labeled
MultiCenter_organized/  # 2,372 multi-site
Tao_Divide/         # 20,580 unlabeled, domain-shift sanity check
```

Weights (gitignored under `model_weights/` and `labeling_tool/weights/`): `sam3.pt` (~3.4 GB),
`biomedclip/`, `dinov2_vitb14_pretrain.pth`, `dinov2_vits14_pretrain.pth`, `dinobloom_vitb14.pth`,
`phikon_v2/`, `plip/`. The BiomedCLIP local dir must contain `open_clip_config.json` whose
`hf_model_name`/`hf_tokenizer_name` point to that same local directory (not an HF hub id).

## Locked-in research findings (don't re-litigate)

These are baked into `BEST_HYBRID_WEIGHTS`, `BALF_CELL_PROFILES`, the cellpose defaults, and the
`.cursorrules` notes. Confirm with experiments before changing.

- **Text prototypes are dead** — BiomedCLIP cross-class text similarity is 0.86–0.96, image-text
  alignment only 0.31–0.35. Pure zero-shot abandoned; `global_text_weight=0.0`.
- **Best ensemble (locked):** BiomedCLIP(0.42) + Phikon-v2(0.18) + DINOv2-S(0.07) + morph(0.33),
  with transductive (2 iter, top5, conf>0.025) + cascade (thr=0.012, morph_w=0.45) →
  Acc 87.13 %, mF1 0.755, Eos F1 0.466.
- **Eosinophil is the bottleneck** (baseline precision ≈ 18.9 %); morphology hard constraints
  + Eos-specialist + cascade exist specifically for it.
- **Lymphocyte is easy** (F1 ≈ 0.89).
- **Non-parametric kNN (k=7) > SVM/LR/MLP** at 40 samples — parametric models overfit.
- **Feature rectification/centering hurts** — preserves BiomedCLIP's native geometry.
- **CellposeSAM:** `cellprob_threshold=-3.0, diameter=50.0` (BALF cells are larger than
  default 30 px) → F1 0.7575, +32.5 % vs default. `flow_threshold=0.2` raises precision to 0.70
  but drops recall to 0.60. Adaptive preprocessing (CLAHE/denoise) and FP post-filtering are
  net-harmful.

## Lab-notes knowledge base

`lab-notes/` exists at the project root. The `.cursor/rules/lab-notes.mdc` rule treats this as
hard-enabled: when the user submits a structured "lab-notes" entry (experiment / insight /
literature), files **must** be written to `lab-notes/{experiments,insights,literature}/` with
the prescribed YAML frontmatter (`date`, `type`, `tags`, plus `exp_id`/`status`/`seed`/`commit`
for experiments), and `lab-notes/INDEX.md` must be appended on top. Never delete historical
entries; mark superseded ones with a new insight that cites the old. `HANDOFF.md` is the
sole file that may be overwritten (preserve a digest of the previous version at the bottom).

## Common gotchas

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: labeling_tool` | Must run uvicorn from project root, not from inside `labeling_tool/` |
| `ModuleNotFoundError: biomedclip_*` | Project root not on `sys.path`; experiment scripts insert it via `sys.path.insert(0, '/home/xut/csclip')` — adjust if checkout lives elsewhere |
| `BiomedCLIP local directory not found` | `MEDSAM_ROOT` not set, or `labeling_tool/weights/biomedclip/` missing `open_clip_config.json` |
| `iopath` / `pycocotools` missing | Required by SAM3 in addition to its `pyproject.toml` deps |
| CUDA OOM in web tool | Auto-falls back to CPU on first request (slow); both SAM3 wrapper and BiomedCLIP loader handle this |
| Stale supports after label edits | `/api/supports?validate=true` triggers `_rebind_support_against_annotations` |
| Cellpose batch results not on disk | Async pattern requires explicit `/api/cellpose_batch/commit` after `awaiting_save` status |
