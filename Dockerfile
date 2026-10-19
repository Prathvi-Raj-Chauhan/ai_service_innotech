FROM python:3.11-slim

# Install system curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at Docker build time
ARG MODEL_DOWNLOAD_URL="https://github.com/Prathvi-Raj-Chauhan/ai_service_innotech/releases/download/0.0.1/yolo26x.onnx"
RUN curl -L -f -o /app/yolo26x_fp32.onnx "${MODEL_DOWNLOAD_URL}"

# Copy quantize script and quantize model to INT8 (shrinks from 213MB to ~55MB in build stage)
COPY quantize.py .
RUN pip install --no-cache-dir onnx && \
    python quantize.py && \
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
