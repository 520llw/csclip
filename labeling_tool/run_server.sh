#!/bin/bash

# Labeling Tool Server Launcher
# This script activates the conda environment and starts the FastAPI server

# Initialize conda
source /c/Users/Administrator/anaconda3/etc/profile.d/conda.sh

# Activate research_assistant environment
conda activate research_assistant

# Set MEDSAM_ROOT if needed (adjust path as necessary)
# export MEDSAM_ROOT="/path/to/MedSAM"

# Navigate to VM_share directory
cd "D:/VM_share"

# Start the FastAPI server
echo "==============================================="
echo "Starting BALF Cell Annotation System"
echo "Environment: research_assistant"
echo "==============================================="
uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000 --reload
