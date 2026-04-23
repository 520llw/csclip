# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BALF Cell Annotation System** — a medical image annotation tool with AI-assisted segmentation. Python FastAPI backend + vanilla JavaScript/Konva.js frontend.

## Running the Server

**CRITICAL: Always run from the parent directory (`D:\VM_share`), NOT from inside `labeling_tool/`:**

```bash
cd D:\VM_share
conda activate research_assistant
python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the batch script: `D:\VM_share\labeling_tool\run_server.bat`

There is no `if __name__ == "__main__"` block — always use uvicorn directly.

## Environment

- **Conda env:** `research_assistant` (Python 3.11.15)
- **GPU:** CUDA 12.8 (PyTorch 2.10.0+cu128)
- **No build system:** Plain Python modules + static JS files (no pyproject.toml, package.json, etc.)

## Architecture

### Backend (`main.py` — ~5800 lines, the core)

Single-file FastAPI application managing:
- **Image group discovery**: scans `MEDSAM_ROOT` for `data*.yaml` (YOLO-format dataset configs) to build image groups with train/val splits
- **Annotation I/O**: reads/writes YOLO-format `.txt` label files alongside images
- **Model orchestration**: lazy-loads and caches ML models; handles GPU OOM by falling back to CPU
- **Dataset management**: enhanced dataset metadata, progress tracking, bulk operations (see Dataset Management section)

### ML Modules

| File | Role |
|------|------|
| `model.py` | SAM3 wrapper — box/text/13-point prompt segmentation |
| `fewshot_biomedclip.py` | BiomedCLIP few-shot classifier using support prototypes |
| `hybrid_classifier.py` | Adaptive classifier combining morphological features + image features + text prompts |
| `cellpose_utils.py` | CellposeSAM segmentation, converts label maps to polygons |
| `paths.py` | Resolves weight file paths (checks `weights/` dir first, then `$MEDSAM_ROOT/assets/`) |

### Frontend (`static/`)

- `index.html`: dark-themed UI with Tailwind CSS
- `app.js` (~228KB): all frontend logic — Konva.js canvas, annotation state, API calls, polygon tool, copy/paste, space-drag pan, visualization modes, image flag filtering, keyboard shortcuts, undo/redo. No build step; plain JS served statically.
- `data-manager.js` (~54KB): enterprise data management panel UI (datasets, projects with dashboard, stats with class distribution charts, export, audit)
- `dataset-manager.js`: legacy dataset management panel (deprecated)

### Enterprise Database (`database.py`)

SQLite-backed enterprise features:
- **Projects**: CRUD, dataset associations, status tracking
- **Annotation versions**: automatic versioning on save (up to 50 versions per image)
- **Audit log**: all operations recorded (annotation, export, project)
- **Session persistence**: user state saved/restored across page loads
- **Daily statistics**: annotation counts, AI assists, active days
- **Image flags**: review status, problem marking
- **Export history**: track all data exports
- **User preferences**: keyboard shortcuts, UI state

### External Dependencies

The parent directory (resolved via `MEDSAM_ROOT`, defaults to parent of `labeling_tool/`) must contain:
- `sam3/` — SAM3 model source code (imported at runtime)
- `biomedclip_fewshot_support_experiment/` — few-shot modules
- `biomedclip_zeroshot_cell_classify/` — classification modules
- `biomedclip_hybrid_adaptive_classifier/` — hybrid classifier
- `data*.yaml` dataset configuration files

Server starts without these, but ML endpoints return 503 errors.

### Weight Files

```
weights/
├── sam3.pt           (3.4GB SAM3 checkpoint)
└── biomedclip/       (BiomedCLIP model weights)
```

### Data Format

**YOLO polygon format:** `.txt` files alongside images with normalized coordinates:
- Polygon: `class_id x1 y1 x2 y2 ...` (normalized 0-1)
- BBox: `class_id cx cy w h` (normalized 0-1)

### Key API Patterns

| Endpoint | Purpose |
|----------|---------|
| `/api/groups` | Image group listing from YAML configs |
| `/api/annotations` + `/api/save` | YOLO-format label file read/write |
| `/api/predict` | SAM3 segmentation (box/text/point prompts) |
| `/api/fewshot/*` | Few-shot classification using `default_supports.json` |
| `/api/hybrid/*` | Hybrid classifier endpoints |
| `/api/cellpose_batch/*` | Async batch segmentation (start/status/cancel/commit pattern) |
| `/api/datasets/enhanced` | Enhanced dataset list with metadata and progress |
| `/api/datasets/{id}/metadata` | Dataset metadata CRUD |
| `/api/images/paginated` | Paginated image list with filtering |
| `/api/projects` | Project CRUD (enterprise) |
| `/api/session` | Session persistence (save/restore) |
| `/api/image_flags` | Image flag system (review status) |
| `/api/audit_log` | Audit log queries |
| `/api/stats/*` | Statistics (daily, summary, overview) |
| `/api/export/*` | Multi-format export (YOLO/COCO/VOC/Mask/CSV) |
| `/api/export_history` | Export history tracking |
| `/api/version_compare` | Annotation version diff (compare two versions) |
| `/api/datasets/stats` | Per-dataset annotation statistics & class distribution |
| `/api/datasets/compare` | Multi-dataset comparison |

### Data Flow

1. Server scans for `data*.yaml` configs → builds image groups
2. Frontend fetches groups → loads images + existing `.txt` annotations
3. User annotates manually or triggers AI segmentation
4. AI returns polygon/mask predictions → user accepts/edits
5. `POST /api/save` writes back to YOLO-format `.txt` files

## Enterprise Data Management (April 2026)

### Database Layer (`database.py` — SQLite)
- Projects, annotation versions, audit log, sessions, daily stats, image flags, export history, user preferences
- WAL mode, foreign keys, automatic schema versioning

### Dataset Management
- **Metadata storage:** `dataset_metadata.json` (separate from `datasets.json`)
- **Progress tracking:** Auto-calculated annotation progress per dataset
- **Bulk operations:** Hide/restore datasets, batch tag updates

### Data Management Center (Frontend)
- **数据集标签页**: 概览卡片、数据集列表、搜索过滤
- **项目标签页**: 项目创建/编辑/归档/仪表盘（含类别分布图和进度统计）
- **统计标签页**: 30天活动总览、每日活动柱状图、系统概览、类别分布饼图+横条图
- **导出标签页**: 多格式导出(YOLO/COCO/VOC/Mask/CSV)、导出历史
- **审计日志标签页**: 操作记录查询与分类过滤

### Annotation Tools
- **Select** (V): 选择标注
- **Adjust** (A): 调整BBox
- **Rectangle** (R): 画矩形框
- **Polygon** (P): 多边形绘制（点击添加顶点、双击/Enter闭合、Backspace回退）
- **SAM** (S): AI智能分割

### Visualization Modes
- **边界+BBox**: 标准模式
- **仅轮廓**: 只显示边界线
- **填充+轮廓**: 半透明填充 + 边界
- **蒙版(仅填充)**: 只显示填充色块，无边界
- **仅BBox**: 只显示包围框

### Annotation List Features
- 按类别分组的分布摘要（可点击过滤）
- 文本搜索过滤（类别名/编号）
- 批量修改类别
- Checkbox多选 + 拖拽选择

### Image List Features
- 状态筛选下拉：全部/已标注/未标注/已标记
- 标记状态图标（✓）显示在图片列表
- 标注进度条
- 文件名搜索

### Keyboard Shortcuts
- Full keyboard shortcut system (press ? to see all)
- Ctrl+Z/Y undo/redo with redo stack (100 steps)
- **Ctrl+C/V**: 复制/粘贴标注
- **Space+拖拽**: 画布平移
- **Ctrl+1**: 实际像素缩放(1:1)
- **Ctrl+Shift+←→**: 跳转到已标记图像
- 1-9 quick class switch, Shift+1-9 to set class on selected
- Ctrl+A select all, Ctrl+D duplicate, Home/End navigation
- Shift+F: 标记当前图像, M: data manager, F: fullscreen canvas

## Notes

- GPU memory is managed manually: models are cleared from VRAM between heavy operations; OOM errors trigger CPU fallback
- `default_supports.json` holds pre-selected few-shot support instances used by both `fewshot_biomedclip.py` and `hybrid_classifier.py`
- Lazy model loading: Models load on first use (10–30s cold start)
- The Cellpose batch endpoint is asynchronous — poll `/api/cellpose_batch/status` until complete, then commit
- Dataset discovery scans `MEDSAM_ROOT` recursively for `data*.yaml` files

## Common Issues

| Problem | Cause |
|---------|-------|
| `ModuleNotFoundError: labeling_tool` | Running from wrong directory. Must be in `D:\VM_share` |
| `ModuleNotFoundError: biomedclip_*` | Missing MedSAM project in parent directory |
| Port 8000 in use | Use `--port 8001` |
| CUDA OOM | Automatic CPU fallback; first request may be slow |
| Static files cached | Hard-refresh browser (Ctrl+F5) |
