import os
import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
LABELS_FILE = BASE_DIR / "labels.json"

# Default configuration
LABELS = {"cell phone": "mobile", "laptop": "laptop", "tv": "tv"}
COCO_IDS = {62: "tv", 63: "laptop", 67: "cell phone"}
CONF_THRESHOLD = 0.4
MIN_AREA_RATIO = 0.01
IMGSZ = 640
IOU_THRESHOLD = 0.7

# Load dynamically from labels.json
if LABELS_FILE.exists():
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        if isinstance(config, dict):
            # 1. Label mapping (coco_to_ours)
            if "coco_to_ours" in config and isinstance(config["coco_to_ours"], dict):
                LABELS = config["coco_to_ours"]

            # 2. Thresholds
            if "conf" in config:
                CONF_THRESHOLD = float(config["conf"])
            if "min_area_ratio" in config:
                MIN_AREA_RATIO = float(config["min_area_ratio"])
            if "imgsz" in config:
                IMGSZ = int(config["imgsz"])
            if "iou" in config:
                IOU_THRESHOLD = float(config["iou"])

            # 3. Target class IDs from 'names'
            if "names" in config and isinstance(config["names"], dict):
                COCO_IDS = {}
                for k, v in config["names"].items():
                    if v in LABELS:
                        COCO_IDS[int(k)] = v
    except Exception as e:
        print(f"[Detector Warning]: Error loading labels.json: {e}")

MODEL_PATH = BASE_DIR / "yolo26x.onnx"

# Auto-download from MODEL_URL if needed
model_url = os.environ.get("MODEL_URL")
if not MODEL_PATH.exists() and model_url:
    print(f"[Detector]: Downloading model in low-memory stream from {model_url} ...")
    try:
        import urllib.request
        import gc
        with urllib.request.urlopen(model_url) as response, open(MODEL_PATH, "wb") as out_file:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                out_file.write(chunk)
        gc.collect()
        print(f"[Detector]: Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"[Detector Error]: Failed to download model from {model_url}: {e}")

if not MODEL_PATH.exists():
    # If yolo26x.onnx is missing, look for any .onnx in the folder
    alt_onnx = list(BASE_DIR.glob("*.onnx"))
    if alt_onnx:
        MODEL_PATH = alt_onnx[0]
        print(f"[Detector]: Using available model: {MODEL_PATH.name}")
    else:
        raise FileNotFoundError(f"No ONNX model found at {MODEL_PATH} and no MODEL_URL specified.")

print(f"[Detector]: Loading ONNX model with ultra-low memory footprint from {MODEL_PATH} ...")

# Configure ONNX Runtime to minimize RAM usage for 512MB environments
session_options = ort.SessionOptions()
session_options.enable_cpu_mem_arena = False  # Disable memory arena pre-allocation
session_options.enable_mem_pattern = False  # Disable tensor memory pattern pre-allocation
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

session = ort.InferenceSession(str(MODEL_PATH), session_options, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
gc.collect()

TARGET_IDS = list(COCO_IDS.keys())
# In YOLO output, columns 0..3 are cx, cy, w, h. The 80 classes start at column 4!
TARGET_COLS = [4 + cid for cid in TARGET_IDS]
print(f"[Detector]: Loaded ONNX Runtime successfully. Target IDs: {TARGET_IDS}, Target Columns: {TARGET_COLS}")
print(f"[Detector]: Thresholds: conf={CONF_THRESHOLD}, min_area={MIN_AREA_RATIO}, imgsz={IMGSZ}, iou={IOU_THRESHOLD}")

def _nms(boxes, scores, iou_thresh):
    """Fast vectorized Non-Maximum Suppression (NMS)."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

def detect(image, conf=CONF_THRESHOLD, min_area=MIN_AREA_RATIO, imgsz=IMGSZ, iou=IOU_THRESHOLD):
    """
    Pure ONNX Runtime inference: zero PyTorch overhead, runs in <200MB RAM.
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")

    orig_w, orig_h = img.size

    # 1. Letterbox resize keeping aspect ratio with padding
    r = min(imgsz / orig_h, imgsz / orig_w)
    new_w = int(round(orig_w * r))
    new_h = int(round(orig_h * r))

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    # 2. Normalize and format tensor: [1, 3, imgsz, imgsz] float32
    blob = np.array(canvas, dtype=np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]

    # 3. ONNX inference
    out = session.run([output_name], {input_name: blob})[0]
    preds = out[0].T  # Transpose [84, 8400] -> [8400, 84]

    # 4. Filter scores for target classes (Columns 0..3 are box coords; classes start at index 4)
    scores_target = preds[:, TARGET_COLS]
    max_subidx = np.argmax(scores_target, axis=1)
    max_scores = np.max(scores_target, axis=1)

    mask = max_scores >= conf
    if not np.any(mask):
        return {"is_target": False, "device": None, "all_detections": []}

    filtered_preds = preds[mask]
    filtered_scores = max_scores[mask]
    filtered_cls = np.array(TARGET_IDS)[max_subidx[mask]]

    # 5. Decode box coordinates (cx, cy, w, h) -> undo letterbox to original pixels
    cx = filtered_preds[:, 0]
    cy = filtered_preds[:, 1]
    bw = filtered_preds[:, 2]
    bh = filtered_preds[:, 3]

    x1 = np.clip((cx - bw / 2 - pad_x) / r, 0, orig_w)
    y1 = np.clip((cy - bh / 2 - pad_y) / r, 0, orig_h)
    x2 = np.clip((cx + bw / 2 - pad_x) / r, 0, orig_w)
    y2 = np.clip((cy + bh / 2 - pad_y) / r, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 6. Apply Non-Maximum Suppression (NMS)
    keep_indices = _nms(boxes, filtered_scores, iou)

    dets = []
    for idx in keep_indices:
        b = [round(float(v)) for v in boxes[idx]]
        area_ratio = (b[2] - b[0]) * (b[3] - b[1]) / (orig_w * orig_h)

        if area_ratio >= min_area:
            raw_cls = COCO_IDS.get(int(filtered_cls[idx]), "device")
            target_label = LABELS.get(raw_cls, raw_cls)

            dets.append({
                "label": target_label,
                "confidence": round(float(filtered_scores[idx]), 3),
                "box": b,
                "area_ratio": round(float(area_ratio), 3)
            })

    # Pick largest device covering photo area
    largest = max(dets, key=lambda d: d["area_ratio"], default=None)

    return {
        "is_target": largest is not None,
        "device": largest,
        "all_detections": dets
    }

# Export values for FastAPI /labels endpoint
IDS = TARGET_IDS
