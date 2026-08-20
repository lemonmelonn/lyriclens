import onnx

# Check the IR version of your exported model
model = onnx.load("./onnx_model/model.onnx")
print("fgcfhjn")
print(f"Model IR version: {model.ir_version}")  # Should be 9