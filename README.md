# Pothole Detection

A YOLO-based pothole detector with two ways to run:

- **Browser app:** upload a road image or use a camera. The ONNX model runs entirely on the device and is deployed with GitHub Pages.
- **Python app:** run real-time inference from a webcam, video, or stream using Ultralytics, OpenCV, and Flask.

## Live demo

After the first successful Pages deployment, the app is available at:

<https://jester-penlza.github.io/Pothole-detection/>

Images and camera frames stay in the browser. They are not uploaded to a server.

## Model results

The included model was trained for one class, `pothole`, using a YOLOv8n detector.

| Metric | Validation result |
| --- | ---: |
| Precision | 0.925 |
| Recall | 0.807 |
| mAP@0.5 | 0.883 |
| mAP@0.5:0.95 | 0.495 |

These results come from the saved `pothole_run50` validation output. Performance can change with lighting, weather, camera angle, road surface, and hardware.

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
