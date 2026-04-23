# Conda Environment Setup Guide

## Current Environment Status

**Environment Name:** `research_assistant`  
**Python Version:** 3.11.15  
**Location:** `C:\Users\Administrator\anaconda3\envs\research_assistant`

## Installed Core Packages

✓ **Web Framework**
- FastAPI 0.135.2
- Uvicorn 0.42.0
- Starlette 1.0.0 (via FastAPI)

✓ **Machine Learning & Deep Learning**
- PyTorch 2.10.0+cu128
- TorchVision 0.25.0+cu128
- TorchAudio 2.10.0+cu128
- NumPy 2.3.5
- SciPy (included with scikit-learn)

✓ **Image Processing**
- OpenCV (opencv-python) 4.13.0
- Pillow (PIL) 12.0.0
- scikit-image (latest)

✓ **Segmentation & Classification**
- Cellpose 4.1.1
- open-clip-torch 3.3.0
- Segment Anything 1.0

✓ **Utilities**
- PyYAML 6.0.3
- scikit-learn 1.8.0

## Full Dependency List

All packages are listed in `requirements.txt` (auto-generated from pip freeze)

## Quick Start

### Option 1: Activate Environment & Run Server

```bash
# On Windows PowerShell or CMD
conda activate research_assistant
cd D:\VM_share
uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Use Provided Script

On Git Bash:
```bash
bash D:\VM_share\labeling_tool\run_server.sh
```

### Option 3: Python Import Test

```bash
conda activate research_assistant
cd D:\VM_share
python -c "import labeling_tool.main as m; print('OK')"
```

## External Dependencies (MedSAM Project)

The labeling_tool imports modules from an external MedSAM project:

**Missing Module:** `biomedclip_fewshot_support_experiment`  
**Expected Location:** `D:\VM_share\` or `$MEDSAM_ROOT/`

To enable full functionality, you need to either:
1. Clone/place the MedSAM project in the parent directory
2. Set `MEDSAM_ROOT` environment variable:
   ```bash
   export MEDSAM_ROOT="path/to/MedSAM"
   ```

## Adding Packages to Environment

If you need to add new packages in the future:

```bash
conda activate research_assistant
pip install <package-name>
```

Or use conda:
```bash
conda install -c conda-forge <package-name>
```

Then update requirements.txt:
```bash
pip freeze > requirements.txt
```

## Recreating the Environment (If Needed)

To recreate this environment on another machine:

```bash
# Create new environment from requirements.txt
conda create -n labeling_env python=3.11
conda activate labeling_env
pip install -r requirements.txt
```

## Troubleshooting

### Issue: "conda: command not found"
**Solution:** Initialize conda in your shell:
```bash
source /c/Users/Administrator/anaconda3/etc/profile.d/conda.sh
conda activate research_assistant
```

### Issue: "ModuleNotFoundError: No module named 'labeling_tool'"
**Solution:** Make sure you're running from the `D:\VM_share` directory, not from inside `labeling_tool/`:
```bash
cd D:\VM_share
python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000
```

### Issue: Missing biomedclip_fewshot_support_experiment
**Solution:** Ensure MedSAM project is in parent directory or set MEDSAM_ROOT

## GPU Support

This environment is configured for CUDA 12.8 (PyTorch with cu128). The setup includes:
- ✓ CUDA support for PyTorch
- ✓ GPU acceleration for Cellpose
- ✓ GPU acceleration for OpenCV (if compiled with CUDA)

Verify GPU is available:
```bash
conda activate research_assistant
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

## Environment Size

The `research_assistant` environment is approximately 2-3GB due to:
- PyTorch with CUDA libraries
- Pre-trained model weights for Cellpose and SAM
- Full scientific Python stack

## Last Updated

Environment was configured and tested on: **April 2, 2026**

## Next Steps

1. Place the MedSAM project in `D:\VM_share\` or set MEDSAM_ROOT
2. Run the server with: `conda activate research_assistant && cd D:\VM_share && uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000 --reload`
3. Open browser to `http://localhost:8000`
