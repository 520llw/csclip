@echo off
REM Labeling Tool Server Launcher for Windows
REM This batch script activates the conda environment and starts the FastAPI server

setlocal enabledelayedexpansion

echo ===============================================
echo BALF Cell Annotation System - Server Launcher
echo ===============================================
echo.

REM Initialize conda
call C:\Users\Administrator\anaconda3\Scripts\activate.bat

REM Activate research_assistant environment
conda activate research_assistant

if errorlevel 1 (
    echo Error: Failed to activate conda environment
    echo Make sure research_assistant environment exists
    pause
    exit /b 1
)

REM Navigate to VM_share directory
cd /d "D:\VM_share"

echo.
echo Environment: research_assistant
echo Working Directory: %CD%
echo.
echo ===============================================
echo Starting server on http://localhost:8000
echo ===============================================
echo.

REM Start the FastAPI server
python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo Error: Failed to start server
    echo Check ENVIRONMENT_SETUP.md for troubleshooting
    pause
)
