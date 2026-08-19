#!/usr/bin/env bash
set -o errexit

echo "--- 1. Installing production dependencies ---"
pip install -r requirements.txt

echo "--- 2. Installing temporary CPU-only PyTorch for ONNX conversion ---"
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "--- 3. Running export script to generate onnx_model/ on the server ---"
python lyriclens/export_to_onnx.py

echo "--- Build process finished successfully! ---"