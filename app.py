import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from detector import detect, LABELS, IDS, CONF_THRESHOLD, MIN_AREA_RATIO, IMGSZ, IOU_THRESHOLD

app = FastAPI(
    title="Innotech E-Waste Photo Detector API",
    description="Identifies e-waste devices (Mobile, Laptop, TV) using YOLO with largest bounding box selection.",
    version="1.0.0"
)

# Enable CORS for mobile app, web app, and backend services
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {
        "service": "Innotech E-Waste Photo Detector",
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "labels": "/labels",
            "predict": "/predict (POST, multipart/form-data with 'file')"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/labels")
def get_labels():
    return {
        "classes": LABELS,
        "ids": IDS,
        "imgsz": IMGSZ,
        "thresholds": {
            "confidence": CONF_THRESHOLD,
            "min_area_ratio": MIN_AREA_RATIO,
            "iou": IOU_THRESHOLD
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file data
    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # Validate file size: 10 MB max
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    # Decode and transpose according to EXIF orientation
    try:
        raw_img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(raw_img).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid image file")

    # Run YOLO detection
    try:
        result = detect(img)
        # width, height
        result["image_size"] = list(img.size)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
