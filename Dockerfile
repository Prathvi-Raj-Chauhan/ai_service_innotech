FROM python:3.11-slim

# Install system curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model during BUILD time so Python runtime starts with clean RAM (0 download overhead)
ARG MODEL_DOWNLOAD_URL="https://github.com/Prathvi-Raj-Chauhan/ai_service_innotech/releases/download/0.0.1/yolo26x.onnx"
RUN curl -L -f -o /app/yolo26x.onnx "${MODEL_DOWNLOAD_URL}" || echo "Build-time download skipped, will fallback to runtime if needed"

# Copy source code and config
COPY . .

# Prevent memory fragmentation in glibc
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV PORT=8000
EXPOSE 8000

# Start single-worker Uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
