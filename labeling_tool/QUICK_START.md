# BALF Cell Annotation System - Quick Reference

## Start Server (Pick One)

### Windows CMD/PowerShell
```cmd
conda activate research_assistant
cd D:\VM_share
python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000
```

### Git Bash
```bash
source /c/Users/Administrator/anaconda3/etc/profile.d/conda.sh
conda activate research_assistant
cd D:/VM_share
python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000
```

### Double-Click (Windows)
```
Run: D:\VM_share\labeling_tool\run_server.bat
```

## Access Application
→ Open browser: **http://localhost:8000**

## Verify Environment
```cmd
conda activate research_assistant
python D:\VM_share\labeling_tool\check_environment.py
```

## Environment Info

| Item | Value |
|------|-------|
| Name | `research_assistant` |
| Python | 3.11.15 |
| PyTorch | 2.10.0+cu128 (GPU Ready) |
| FastAPI | 0.135.2 |
| Cellpose | 4.1.1 |
| Status | ✓ Ready |

## Installed Packages Summary

✓ Web: FastAPI, Uvicorn  
✓ ML: PyTorch, TorchVision  
✓ Images: OpenCV, Pillow, scikit-image  
✓ Models: Cellpose, open-clip, SAM  
✓ Utils: NumPy, PyYAML, scikit-learn  

**Full list:** requirements.txt (325+ packages)

## Documentation

- **Detailed Setup:** ENVIRONMENT_SETUP.md
- **Full Status:** ENVIRONMENT_READY.md
- **CLI & Architecture:** CLAUDE.md
- **UI/UX Changes:** Right sidebar reorganized by workflow

## Files in This Directory

```
labeling_tool/
├── main.py                      [Server code]
├── model.py                     [SAM3 wrapper]
├── fewshot_biomedclip.py       [Few-shot classifier]
├── hybrid_classifier.py         [Hybrid classifier]
├── cellpose_utils.py           [Cellpose wrapper]
├── static/
│   ├── index.html              [Web UI - REDESIGNED]
│   └── app.js                  [Frontend logic]
├── weights/
│   ├── sam3.pt                 [SAM3 checkpoint]
│   └── biomedclip/             [BiomedCLIP weights]
├── run_server.bat              [Windows launcher]
├── run_server.sh               [Bash launcher]
├── check_environment.py        [Verification script]
├── requirements.txt            [All dependencies]
├── ENVIRONMENT_SETUP.md        [Setup guide]
├── ENVIRONMENT_READY.md        [Status report]
├── CLAUDE.md                   [Architecture guide]
└── CLAUDE.md-quick-ref.txt     [This file]
```

## Recent Changes

✓ **Right Sidebar Reorganized** (UI)
  - New workflow progress indicator (分割 → 细化 → 分类)
  - CellposeSAM moved to step 1 (expanded by default)
  - SAM3 moved to step 2 (collapsed by default)
  - Smart Classification moved to step 3 (collapsed by default)
  - Step badges on each section header

✓ **Environment Setup Complete**
  - Conda environment configured with all dependencies
  - Created launcher scripts for easy startup
  - Environment verification tools

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError: biomedclip_fewshot_support_experiment" | Normal - MedSAM not installed. Server still works for UI. |
| Port 8000 already in use | Change port: `--port 8001` |
| CUDA errors | Check: `python -c "import torch; print(torch.cuda.is_available())"` |
| Import errors | Run: `python check_environment.py` |

## Tips

- Use `--reload` flag for development (auto-restart on code changes)
- Remove `--reload` for production
- Access static files directly: http://localhost:8000 → loads `/static/index.html`
- Check logs in terminal for errors
- Maximum image size is limited by available VRAM

---

**Setup Date:** April 2, 2026  
**Environment:** research_assistant (conda)  
**Status:** Production Ready ✓
