FROM python:3.11-slim

# Install system dependencies for image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model, config, and source code
COPY . .

ENV PORT=8000
EXPOSE 8000

# Start Uvicorn with 1 worker to conserve RAM on cloud instances
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
