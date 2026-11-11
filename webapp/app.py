import os
import sys
import time
from pathlib import Path
from typing import Generator

import cv2
from flask import Flask, Response, jsonify, render_template, request

# Allow `python webapp/app.py` to import modules from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DetectionConfig
from src.detector import PotholeDetector


app = Flask(__name__, template_folder="templates")

MODEL_PATH = os.getenv("POTHOLE_MODEL_PATH", "models/pothole_best.pt")
INPUT_SOURCE = os.getenv("POTHOLE_INPUT_SOURCE", "0")
CONFIDENCE = float(os.getenv("POTHOLE_CONFIDENCE", "0.35"))
DEVICE = os.getenv("POTHOLE_DEVICE", "cpu")

stats = {
    "detections": 0,
    "fps": 0.0,
    "status": "starting",
    "error": "",
    "source": INPUT_SOURCE,
    "model": MODEL_PATH,
    "confidence": CONFIDENCE,
    "device": DEVICE,
}


def _resolve_params() -> tuple[str, float, str, str]:
    """Resolve per-request parameters with safe fallbacks."""
    source_arg = request.args.get("source", INPUT_SOURCE)
    model_arg = request.args.get("model", MODEL_PATH)
    conf_arg = request.args.get("conf", None)
    device_arg = request.args.get("device", DEVICE)

    source_arg = source_arg if source_arg else INPUT_SOURCE
    model_arg = model_arg if model_arg else MODEL_PATH

    try:
        conf_value = float(conf_arg) if conf_arg is not None else CONFIDENCE
    except ValueError:
        conf_value = CONFIDENCE

    return source_arg, conf_value, model_arg, device_arg


def frame_stream(
    source_raw: str,
    confidence: float,
    model_path: str,
    device: str,
) -> Generator[bytes, None, None]:
    source = DetectionConfig.normalize_source(source_raw)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        stats["status"] = "error"
        stats["error"] = f"Unable to open input source: {source_raw}"
        return

    try:
        detector = PotholeDetector(model_path, confidence, device)
    except Exception as exc:
        stats["status"] = "error"
        stats["error"] = str(exc)
        cap.release()
        return

    stats["status"] = "running"
    stats["error"] = ""
    stats["source"] = source_raw
    stats["model"] = model_path
    stats["confidence"] = confidence
    stats["device"] = device
    previous_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            stats["status"] = "ended"
            break

        result = detector.detect(frame)
        boxes = getattr(result, "boxes", None)
        stats["detections"] = int(len(boxes)) if boxes is not None else 0

        annotated = result.plot()

        current_time = time.time()
        elapsed = max(current_time - previous_time, 1e-6)
        stats["fps"] = round(1.0 / elapsed, 2)
        previous_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {stats['fps']}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Detections: {stats['detections']}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        success, buffer = cv2.imencode(".jpg", annotated)
        if not success:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        default_model=MODEL_PATH,
        default_source=INPUT_SOURCE,
        default_conf=CONFIDENCE,
    )


@app.route("/video_feed")
def video_feed() -> Response:
    source_arg, conf_arg, model_arg, device_arg = _resolve_params()
    return Response(
        frame_stream(source_arg, conf_arg, model_arg, device_arg),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def get_stats() -> Response:
    source_arg, conf_arg, model_arg, device_arg = _resolve_params()
    stats["source"] = source_arg
    stats["model"] = model_arg
    stats["confidence"] = conf_arg
    stats["device"] = device_arg
    return jsonify(stats)


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
