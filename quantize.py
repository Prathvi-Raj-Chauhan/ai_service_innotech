import os
import shutil

print("[Quantizer]: Starting INT8 quantization...")

fp32_model = "/app/yolo26x_fp32.onnx"
int8_model = "/app/yolo26x.onnx"

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    if os.path.exists(fp32_model):
        quantize_dynamic(fp32_model, int8_model, weight_type=QuantType.QUInt8)
        os.remove(fp32_model)
        print("[Quantizer]: Successfully quantized model to INT8 (shrunk to ~55MB)!")
    else:
        print("[Quantizer]: fp32 model not found, skipping quantization.")
except Exception as e:
    print(f"[Quantizer Warning]: Quantization failed: {e}. Keeping original model.")
    if os.path.exists(fp32_model):
        shutil.move(fp32_model, int8_model)
