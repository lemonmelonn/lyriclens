#!/usr/bin/env bash
set -o errexit

echo "--- 1. Installing temporary CPU-only PyTorch and onnxscript first ---"
pip install torch onnxscript --index-url https://download.pytorch.org/whl/cpu

echo "--- 2. Installing remaining production requirements ---"
pip install -r requirements.txt

echo "--- 3. Running export script to generate onnx_model/ on the server ---"
python lyriclens/export_to_onnx.py

echo "--- Build process finished successfully! ---"