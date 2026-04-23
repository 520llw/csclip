@echo off
set PYTHONPATH=D:\VM_share
cd /d D:\VM_share\labeling_tool
"C:\Users\Administrator\anaconda3\envs\research_assistant\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
