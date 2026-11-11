import os
from dataclasses import dataclass


@dataclass
class DetectionConfig:
    """Configuration used for real-time pothole detection."""

    model_path: str = os.getenv("POTHOLE_MODEL_PATH", "models/pothole_best.pt")
    confidence: float = float(os.getenv("POTHOLE_CONFIDENCE", "0.35"))
    input_source: str = os.getenv("POTHOLE_INPUT_SOURCE", "0")
    device: str = os.getenv("POTHOLE_DEVICE", "cpu")
    output_path: str = os.getenv("POTHOLE_OUTPUT_PATH", "")
    window_name: str = "Real-Time Pothole Detection"

    @staticmethod
    def normalize_source(source: str):
        """Use webcam index if source is numeric, otherwise treat it as file/stream path."""
        if source.isdigit():
            return int(source)
        return source
