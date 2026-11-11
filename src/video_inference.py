import argparse
import time
from pathlib import Path

import cv2

from src.config import DetectionConfig
from src.detector import PotholeDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time pothole detection")
    parser.add_argument("--model", default=None, help="Path to YOLO model (.pt)")
    parser.add_argument(
        "--source",
        default=None,
        help="Input source (webcam index like 0, video file path, or stream URL)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Detection confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (e.g., 0 for GPU, cpu for CPU)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save annotated output video",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DetectionConfig:
    cfg = DetectionConfig()
    if args.model is not None:
        cfg.model_path = args.model
    if args.source is not None:
        cfg.input_source = args.source
    if args.confidence is not None:
        cfg.confidence = args.confidence
    if args.device is not None:
        cfg.device = args.device
    if args.output is not None:
        cfg.output_path = args.output
    return cfg


def ensure_output_dir(output_path: str) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def run() -> None:
    args = parse_args()
    config = build_config(args)

    detector = PotholeDetector(config.model_path, config.confidence, config.device)
    source = DetectionConfig.normalize_source(config.input_source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input source: {config.input_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    writer = None
    if config.output_path:
        ensure_output_dir(config.output_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(config.output_path, fourcc, fps, (width, height))

    prev_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.detect(frame)
        annotated = result.plot()

        current_time = time.time()
        frame_fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {frame_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(annotated)

        cv2.imshow(config.window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
