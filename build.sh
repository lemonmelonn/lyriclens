#!/usr/bin/env bash
set -o errexit

echo "--- 1. Installing temporary CPU-only PyTorch from PyTorch index ---"
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "--- 2. Installing onnxscript and remaining production requirements from PyPI ---"
pip install onnxscript
pip install -r requirements.txt

echo "--- 3. Running export script to generate onnx_model/ on the server ---"
python lyriclens/export_to_onnx.py

echo "--- Build process finished successfully! ---"