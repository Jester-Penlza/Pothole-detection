# Pothole Detection

A YOLO-based pothole detector with two ways to run:

- **Browser app:** upload a road image or use a camera. The ONNX model runs entirely on the device and is deployed with GitHub Pages.
- **Python app:** run real-time inference from a webcam, video, or stream using Ultralytics, OpenCV, and Flask.

## Live demo

After the first successful Pages deployment, the app is available at:

<https://jester-penlza.github.io/Pothole-detection/>

Images and camera frames stay in the browser. They are not uploaded to a server.

## Accuracy

The saved `pothole_run50` validation results report:

| Metric | Result | Practical meaning |
| --- | ---: | --- |
| Precision | **92.5%** | About 92.5 of every 100 reported pothole detections were correct on the validation set. |
| Recall | **80.7%** | The model found about 80.7 of every 100 labeled potholes and missed about 19.3. |
| mAP@0.5 | **88.3%** | Overall detection quality when a predicted box needs at least 50% overlap with the labeled box. |
| mAP@0.5:0.95 | **49.5%** | Stricter average quality across overlap thresholds from 50% to 95%. |

These figures were measured on the project's **130-image validation split**, which contains **161 labeled pothole boxes**. They describe performance on that dataset—not guaranteed accuracy on every road. Real-world results can be lower in darkness, rain, glare, heavy shadows, motion blur, unusual camera angles, distant or partly hidden potholes, and road surfaces that differ from the training data.

## Detection specifications

| Specification | Value |
| --- | --- |
| Task | Object detection with bounding boxes |
| Architecture | Ultralytics YOLOv8n |
| Detection classes | 1: `pothole` |
| Parameters | 3,005,843 |
| Compute | Approximately 8.1 GFLOPs per 640×640 image |
| Model input | One RGB image, resized and letterboxed to `1 × 3 × 640 × 640` |
| Model output | Up to 300 detections in `[x1, y1, x2, y2, confidence, class]` format |
| Default confidence threshold | 35% |
| Browser threshold range | 25%–90%, adjustable by the user |
| PyTorch weights | `models/pothole_best.pt`, 5.96 MiB |
| Browser weights | `site/models/pothole_best.onnx`, 11.70 MiB |
| ONNX export | IR version 8, opset 17, built-in non-maximum suppression |
| Browser runtime | ONNX Runtime Web 1.23.2 using WebAssembly on the visitor's device |
| Python runtime | Ultralytics + OpenCV; CPU by default, CUDA GPU optional |
| Browser inputs | JPG, PNG, WEBP, drag-and-drop, or live device camera |
| Python inputs | Webcam index, video file, or compatible OpenCV stream URL |
| Privacy | Browser images and camera frames are processed locally and are not uploaded |
| Deployment | Static HTTPS site hosted by GitHub Pages |

### Speed

The saved Ultralytics GPU validation benchmark recorded approximately **7.0 ms preprocessing + 9.3 ms inference + 9.5 ms postprocessing = 25.8 ms per image**, or about **38.8 FPS**. This is a hardware-specific benchmark, not a guaranteed speed. During verification of the deployed WebAssembly version, the two sample images took approximately **481–564 ms per image** in the test browser. Browser speed varies substantially by CPU, device, browser, and power mode.

### Current scope

The detector identifies and draws boxes around visible potholes. It does **not** currently estimate physical distance, depth, dimensions, or severity; identify cracks as a separate class; attach GPS coordinates; or guarantee safe driving decisions. Those capabilities require additional sensors, labels, calibration, and evaluation.

## Run the browser app locally

The browser blocks model loading from a `file://` URL, so serve the site over HTTP:

```powershell
python -m http.server 8000 --directory site
```

Open <http://localhost:8000>.

## Run the Python detector

Requirements: Python 3.10+ and a webcam or video source.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Web dashboard:

```powershell
python webapp\app.py
```

Open <http://127.0.0.1:5000>.

Command-line webcam inference:

```powershell
python -m src.video_inference --source 0 --model models/pothole_best.pt --device cpu
```

Use `--source path\to\video.mp4` for a video file. Add `--output outputs\annotated.mp4` to save the annotated result.

## Repository structure

```text
models/pothole_best.pt        Original Ultralytics model
site/                         Static GitHub Pages browser app
src/                          Python detector and video inference
webapp/                       Flask camera dashboard
.github/workflows/pages.yml   Automatic Pages deployment
```

The large training dataset, training runs, temporary environments, and downloaded base weights are intentionally excluded from Git. The dataset configuration identifies the source as Roboflow project `pothole-vhmow-hilwi`, version 1, licensed CC BY 4.0.

## Deployment

Every push to `main` runs `.github/workflows/pages.yml`, uploads `site/`, and deploys it to GitHub Pages. Pages must use **GitHub Actions** as its publishing source in the repository settings.
