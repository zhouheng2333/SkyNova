#!/bin/bash
set -euo pipefail

echo "Stopping SkyNova services..."

pkill -f "controller.py" || true
pkill -f "model_worker.py" || true
pkill -f "streamlit run app.py" || true
pkill -f "sd_worker.py" || true

echo "Stop command sent."
