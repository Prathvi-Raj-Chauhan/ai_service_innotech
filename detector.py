import os
import json
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
LABELS_FILE = BASE_DIR / "labels.json"

# Defaults
LABELS = {"cell phone": "mobile", "laptop": "laptop", "tv": "tv"}
CONF_THRESHOLD = 0.4
MIN_AREA_RATIO = 0.01
IOU_THRESHOLD = 0.7
EXPLICIT_IDS = None

# Load labels, COCO IDs, and thresholds dynamically from labels.json
if LABELS_FILE.exists():
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        if isinstance(config, dict):
            # 1. Class mappings
            if "classes" in config and isinstance(config["classes"], dict):
                LABELS = config["classes"]
            elif any(k in config for k in ["cell phone", "laptop", "tv", "mobile"]):
                LABELS = {k: v for k, v in config.items() if isinstance(v, str)}

            # 2. Thresholds (confidence, min_area_ratio, iou)
            thresholds = config.get("thresholds", {})
            if isinstance(thresholds, dict):
                CONF_THRESHOLD = float(thresholds.get("confidence", CONF_THRESHOLD))
                MIN_AREA_RATIO = float(thresholds.get("min_area_ratio", MIN_AREA_RATIO))
                IOU_THRESHOLD = float(thresholds.get("iou", IOU_THRESHOLD))

            # 3. Explicit COCO IDs (e.g. {"tv": 62, "laptop": 63, "cell phone": 67})
            coco_ids = config.get("coco_ids", {})
            if isinstance(coco_ids, dict) and coco_ids:
                EXPLICIT_IDS = [int(v) for v in coco_ids.values()]
                print(f"[Detector]: Using explicit COCO IDs from labels.json: {EXPLICIT_IDS}")
    except Exception as e:
        print(f"[Detector Warning]: Error reading labels.json, using defaults: {e}")

MODEL_PATH = BASE_DIR / "yolo26x.onnx"

if not MODEL_PATH.exists():
    # If yolo26x.onnx is not yet present, check for any other .onnx or .pt file
    alt_models = list(BASE_DIR.glob("*.onnx")) + list(BASE_DIR.glob("*.pt"))
    if alt_models:
        MODEL_PATH = alt_models[0]
        print(f"[Detector]: 'yolo26x.onnx' not found. Using detected model: {MODEL_PATH.name}")
    else:
        # Fallback to standard lightweight YOLO that downloads automatically on first run
        MODEL_PATH = "yolo11n.pt"
        print(f"[Detector]: 'yolo26x.onnx' not yet placed in {BASE_DIR}. Using '{MODEL_PATH}' as standard fallback.")

print(f"[Detector]: Loading YOLO model from {MODEL_PATH} ...")
model = YOLO(str(MODEL_PATH), task="detect")

# Find class IDs matching target labels
if EXPLICIT_IDS:
    IDS = EXPLICIT_IDS
else:
    IDS = [i for i, n in model.names.items() if n in LABELS or LABELS.get(n) in ["mobile", "laptop", "tv"]]

print(f"[Detector]: Target classes active: {[model.names.get(i, f'ID:{i}') for i in IDS]} with IDs: {IDS}")
print(f"[Detector]: Thresholds active: conf={CONF_THRESHOLD}, min_area={MIN_AREA_RATIO}, iou={IOU_THRESHOLD}")

def detect(image, conf=CONF_THRESHOLD, min_area=MIN_AREA_RATIO, iou=IOU_THRESHOLD):
    """
    Detects target e-waste devices (mobile, laptop, tv) in the input image.
    Args:
        image: PIL Image, numpy array, or file path.
        conf: minimum confidence score threshold (default from labels.json or 0.4).
        min_area: minimum area ratio threshold (default from labels.json or 0.01).
        iou: NMS IoU threshold (default from labels.json or 0.7).
    Returns:
        dict: {"is_target": bool, "device": dict or None, "all_detections": list}
    """
    results = model.predict(image, classes=IDS, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
    h, w = results.orig_shape
    dets = []

    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1) / (w * h)
        if area >= min_area:
            raw_cls_name = results.names[int(b.cls[0])]
            dets.append({
                "label": LABELS.get(raw_cls_name, raw_cls_name),
                "confidence": round(float(b.conf[0]), 3),
                "box": [round(v) for v in (x1, y1, x2, y2)],
                "area_ratio": round(area, 3)
            })

    # If several are present, pick the one covering the largest area
    largest = max(dets, key=lambda d: d["area_ratio"], default=None)
    return {
        "is_target": largest is not None,
        "device": largest,
        "all_detections": dets
    }
