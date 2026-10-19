FROM python:3.11-slim

# Install system curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at Docker build time (Build environment has multiple GB of RAM)
ARG MODEL_DOWNLOAD_URL="https://github.com/Prathvi-Raj-Chauhan/ai_service_innotech/releases/download/0.0.1/yolo26x.onnx"
RUN curl -L -f -o /app/yolo26x_fp32.onnx "${MODEL_DOWNLOAD_URL}"

# Quantize model to INT8 during build to shrink from 213MB to ~55MB (drastically cuts RAM for 512MB instances)
RUN pip install --no-cache-dir onnx && \
    python -c "try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print('Quantizing model to INT8...')
    quantize_dynamic('/app/yolo26x_fp32.onnx', '/app/yolo26x.onnx', weight_type=QuantType.QUInt8)
    import os; os.remove('/app/yolo26x_fp32.onnx')
    print('Quantization completed successfully!')
except Exception as e:
    print('Quantization skipped/error:', e)
    import shutil; shutil.move('/app/yolo26x_fp32.onnx', '/app/yolo26x.onnx')" && \
    pip uninstall -y onnx

# Copy application files
COPY . .

# Prevent glibc memory fragmentation
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV PORT=8000
EXPOSE 8000

# Start single-worker Uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
