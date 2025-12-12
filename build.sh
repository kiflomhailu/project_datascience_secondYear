#!/bin/bash
# Render build script for Flask API

echo "Installing Python dependencies..."
pip install -r dashboard/api/requirements.txt

echo "Verifying models exist..."
if [ ! -f "latest/seismic_event_occurrence_model_v2.cbm" ]; then
    echo "ERROR: Model files not found!"
    exit 1
fi

echo "Build complete!"
