from pathlib import Path
from typing import Any


class PotholeDetector:
    """Thin wrapper around a YOLO model for pothole detection."""

    def __init__(self, model_path: str, confidence: float = 0.35, device: str = "0") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'ultralytics'. Install requirements with: "
                "pip install -r requirements.txt"
            ) from exc

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_file}. "
                "Train/export a model and place it at this path or override with --model."
            )

        self.model = YOLO(str(model_file))
        self.confidence = confidence
        self.device = device

    def detect(self, frame: Any):
        """Run inference on a frame and return the first YOLO result object."""
        results = self.model.predict(frame, conf=self.confidence, device=self.device, verbose=False)
        return results[0]
