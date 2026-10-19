import sys
import json
import requests

SERVER_URL = "http://localhost:8000"

def test_health():
    try:
        res = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"[Health Check]: Status {res.status_code} -> {res.json()}")
        return res.status_code == 200
    except Exception as e:
        print(f"[Health Check Failed]: Is the server running at {SERVER_URL}? Error: {e}")
        return False

def test_predict(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            res = requests.post(f"{SERVER_URL}/predict", files=files, timeout=30)
            print(f"[Prediction Status]: {res.status_code}")
            print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"[Prediction Failed]: {e}")

def test_labels():
    try:
        res = requests.get(f"{SERVER_URL}/labels", timeout=5)
        print(f"[Active Labels & Thresholds]: Status {res.status_code}")
        print(json.dumps(res.json(), indent=2))
        return res.status_code == 200
    except Exception as e:
        print(f"[Labels Check Failed]: {e}")
        return False

if __name__ == "__main__":
    print("Testing Innotech E-Waste Detector API...")
    ok = test_health()
    if ok:
        test_labels()
    if len(sys.argv) > 1:
        test_predict(sys.argv[1])
    else:
        print("\nTip: Run with an image path to test prediction:")
        print("     python test_client.py test_phone.jpg")
