
============================================================
BALF Cell Annotation System - Environment Check
============================================================

[Web Framework]
  [OK] FastAPI              0.135.2
  [OK] Uvicorn              0.42.0
  [OK] Starlette            1.0.0

[Deep Learning]
  [OK] PyTorch              2.10.0+cu128
  [OK] TorchVision          0.25.0+cu128

[Image Processing]
  [OK] OpenCV               4.13.0
  [OK] Pillow               12.0.0
  [OK] scikit-image         0.26.0

[ML Utilities]
  [OK] NumPy                2.3.5
  [OK] PyYAML               6.0.3
  [OK] scikit-learn         1.8.0

[Advanced Models]
  [OK] Cellpose             unknown
  [OK] open-clip            3.3.0
  [OK] SAM                  unknown

[Data Processing]
  [OK] Pydantic             2.12.5

============================================================
[SUCCESS] All required packages are installed!

You can now start the server:
  conda activate research_assistant
  cd D:\VM_share
  python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000
