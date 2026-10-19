# Innotech E-Waste Photo Detector Microservice

This microservice runs YOLO inference on uploaded photos to detect **Mobile**, **Laptop**, or **TV** e-waste items and returns bounding box coordinates, confidence scores, and area ratios.

---

## 📁 Directory Layout

```
ai_service/
├── yolo26x.onnx         <- Put your model file here (213 MB)
├── labels.json          <- Target classes (mobile, laptop, tv) and thresholds
├── detector.py          <- Loads model once and executes detection
├── app.py               <- FastAPI server with /predict and /health
├── requirements.txt     <- Python dependencies
├── Dockerfile           <- Container configuration for cloud deployment
├── test_client.py       <- CLI tool to test /predict
└── README.md
```

---

## 🚀 Quickstart (Local)

### 1. Install Dependencies
```bash
cd ai_service
pip install -r requirements.txt
```

### 2. Add Model File
Place `yolo26x.onnx` directly inside the `ai_service/` directory.  
*(If not present, the service will fall back to downloading standard `yolo11n.pt` for local debugging).*

### 3. Start Server
```bash
python app.py
# Or:
uvicorn app:app --host 0.0.0.0 --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 4. Test Detection
```bash
python test_client.py sample_phone.jpg
```

---

## 🌐 Cloud Deployment (e.g., Render)

### Deploying as a Docker Web Service on Render:
1. In your GitHub repo, push the `ai_service/` folder.
2. Go to **Render Dashboard** → **New +** → **Web Service**.
3. Select your repository.
4. Set **Root Directory** to `ai_service`.
5. Set **Environment** to `Docker`.
6. Select Instance Type (Render free tier or Starter with 1GB+ RAM).
7. Once deployed, copy your service URL (e.g., `https://innotech-ai-detector.onrender.com`).

---

## 🔗 Connecting to Innotech Backend

In `backend/.env` (and in your Render backend Environment Variables):
```env
ENABLE_EXTERNAL_AI=true
AI_SERVICE_URL=https://innotech-ai-detector.onrender.com
```
