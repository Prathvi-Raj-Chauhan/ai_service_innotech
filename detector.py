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
IMGSZ = 640
IOU_THRESHOLD = 0.7
IDS = [62, 63, 67]

# Load from labels.json (supports names, coco_to_ours, conf, min_area_ratio, imgsz)
if LABELS_FILE.exists():
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        if isinstance(config, dict):
            # 1. Label mapping (coco_to_ours or classes)
            if "coco_to_ours" in config and isinstance(config["coco_to_ours"], dict):
                LABELS = config["coco_to_ours"]
            elif "classes" in config and isinstance(config["classes"], dict):
                LABELS = config["classes"]

            # 2. Confidence threshold
            if "conf" in config:
                CONF_THRESHOLD = float(config["conf"])
            elif "thresholds" in config and "confidence" in config["thresholds"]:
                CONF_THRESHOLD = float(config["thresholds"]["confidence"])

            # 3. Minimum area ratio
            if "min_area_ratio" in config:
                MIN_AREA_RATIO = float(config["min_area_ratio"])
            elif "thresholds" in config and "min_area_ratio" in config["thresholds"]:
                MIN_AREA_RATIO = float(config["thresholds"]["min_area_ratio"])

            # 4. Image size (imgsz)
            if "imgsz" in config:
                IMGSZ = int(config["imgsz"])

            # 5. IoU threshold
            if "iou" in config:
                IOU_THRESHOLD = float(config["iou"])
            elif "thresholds" in config and "iou" in config["thresholds"]:
                IOU_THRESHOLD = float(config["thresholds"]["iou"])

            # 6. Extract class IDs matching target labels
            if "names" in config and isinstance(config["names"], dict):
                names_dict = config["names"]
                matched_ids = [int(k) for k, v in names_dict.items() if v in LABELS]
                if matched_ids:
                    IDS = matched_ids
            elif "coco_ids" in config and isinstance(config["coco_ids"], dict):
                IDS = [int(v) for v in config["coco_ids"].values()]
    except Exception as e:
        print(f"[Detector Warning]: Error reading labels.json, using defaults: {e}")

MODEL_PATH = BASE_DIR / "yolo26x.onnx"

# 1. Check if model can be downloaded from MODEL_URL environment variable
model_url = os.environ.get("MODEL_URL")
if not MODEL_PATH.exists() and model_url:
    print(f"[Detector]: Downloading model from {model_url} ...")
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        print(f"[Detector]: Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"[Detector Error]: Failed to download model from {model_url}: {e}")

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

# Verify active class IDs against loaded model
active_target_names = [model.names.get(i, str(i)) for i in IDS]
print(f"[Detector]: Active target classes: {active_target_names} with IDs: {IDS}")
print(f"[Detector]: Active thresholds: conf={CONF_THRESHOLD}, min_area={MIN_AREA_RATIO}, imgsz={IMGSZ}, iou={IOU_THRESHOLD}")

def detect(image, conf=CONF_THRESHOLD, min_area=MIN_AREA_RATIO, imgsz=IMGSZ, iou=IOU_THRESHOLD):
    """
    Detects target e-waste devices (mobile, laptop, tv) in the input image.
    Args:
        image: PIL Image, numpy array, or file path.
        conf: minimum confidence score threshold (default from labels.json: 0.4).
        min_area: minimum area ratio threshold (default from labels.json: 0.01).
        imgsz: inference image size (default from labels.json: 640).
        iou: NMS IoU threshold (default: 0.7).
    Returns:
        dict: {"is_target": bool, "device": dict or None, "all_detections": list}
    """
    results = model.predict(image, classes=IDS, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    h, w = results.orig_shape
    dets = []

    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1) / (w * h)
        if area >= min_area:
            cls_id = int(b.cls[0])
            raw_cls_name = results.names.get(cls_id, str(cls_id))
            target_label = LABELS.get(raw_cls_name, raw_cls_name)
            dets.append({
                "label": target_label,
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
