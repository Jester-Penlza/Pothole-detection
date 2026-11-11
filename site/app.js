"use strict";

const MODEL_SIZE = 640;
const MODEL_URL = "models/pothole_best.onnx";
const RUNTIME_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
const CAMERA_INTERVAL_MS = 350;

const elements = {
  cameraButton: document.querySelector("#cameraButton"),
  cameraVideo: document.querySelector("#cameraVideo"),
  confidence: document.querySelector("#confidence"),
  confidenceValue: document.querySelector("#confidenceValue"),
  detectionCount: document.querySelector("#detectionCount"),
  dropZone: document.querySelector("#dropZone"),
  emptyState: document.querySelector("#emptyState"),
  fileInput: document.querySelector("#fileInput"),
  inferenceTime: document.querySelector("#inferenceTime"),
  modelCanvas: document.querySelector("#modelCanvas"),
  modelStatus: document.querySelector("#modelStatus"),
  resultMessage: document.querySelector("#resultMessage"),
  sampleButtons: document.querySelectorAll(".sample-button"),
  stopButton: document.querySelector("#stopButton"),
  topConfidence: document.querySelector("#topConfidence"),
  uploadLabel: document.querySelector("#uploadLabel"),
  viewerCanvas: document.querySelector("#viewerCanvas"),
  workingState: document.querySelector("#workingState"),
};

const state = {
  session: null,
  stream: null,
  cameraRunning: false,
  inferenceBusy: false,
  lastCameraFrame: 0,
  currentImage: null,
};

function setModelStatus(message, type = "loading") {
  elements.modelStatus.classList.toggle("ready", type === "ready");
  elements.modelStatus.classList.toggle("error", type === "error");
  elements.modelStatus.lastElementChild.textContent = message;
}

function setWorking(isWorking) {
  elements.workingState.hidden = !isWorking;
}

function setControlsEnabled(enabled) {
  elements.cameraButton.disabled = !enabled;
  elements.fileInput.disabled = !enabled;
  elements.sampleButtons.forEach((button) => { button.disabled = !enabled; });
  elements.uploadLabel.setAttribute("aria-disabled", String(!enabled));
}

async function loadModel() {
  try {
    setControlsEnabled(false);
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 1)
      : 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = RUNTIME_BASE;

    state.session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    setModelStatus("Detector ready", "ready");
    setControlsEnabled(true);
  } catch (error) {
    console.error(error);
    setModelStatus("Model failed to load", "error");
    elements.resultMessage.textContent = "Refresh the page and check your connection.";
  }
}

function getSourceSize(source) {
  if (source instanceof HTMLVideoElement) {
    return { width: source.videoWidth, height: source.videoHeight };
  }
  return { width: source.naturalWidth, height: source.naturalHeight };
}

function prepareInput(source) {
  const { width, height } = getSourceSize(source);
  if (!width || !height) throw new Error("The selected image has no readable dimensions.");

  const scale = Math.min(MODEL_SIZE / width, MODEL_SIZE / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  const padX = Math.round((MODEL_SIZE - resizedWidth) / 2 - 0.1);
  const padY = Math.round((MODEL_SIZE - resizedHeight) / 2 - 0.1);

  const context = elements.modelCanvas.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "rgb(114, 114, 114)";
  context.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, padX, padY, resizedWidth, resizedHeight);

  const pixels = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const planeSize = MODEL_SIZE * MODEL_SIZE;
  const tensorData = new Float32Array(planeSize * 3);

  for (let pixel = 0, offset = 0; pixel < planeSize; pixel += 1, offset += 4) {
    tensorData[pixel] = pixels[offset] / 255;
    tensorData[planeSize + pixel] = pixels[offset + 1] / 255;
    tensorData[(planeSize * 2) + pixel] = pixels[offset + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", tensorData, [1, 3, MODEL_SIZE, MODEL_SIZE]),
    transform: { width, height, scale, padX, padY },
  };
}

function parseDetections(output, transform) {
  const threshold = Number(elements.confidence.value) / 100;
  const detections = [];
  const values = output.data;

  for (let offset = 0; offset + 5 < values.length; offset += 6) {
    const score = values[offset + 4];
    if (score < threshold) continue;

    const x1 = Math.max(0, Math.min(transform.width, (values[offset] - transform.padX) / transform.scale));
    const y1 = Math.max(0, Math.min(transform.height, (values[offset + 1] - transform.padY) / transform.scale));
    const x2 = Math.max(0, Math.min(transform.width, (values[offset + 2] - transform.padX) / transform.scale));
    const y2 = Math.max(0, Math.min(transform.height, (values[offset + 3] - transform.padY) / transform.scale));

    if (x2 <= x1 || y2 <= y1) continue;
    detections.push({ x1, y1, x2, y2, score });
  }

  return detections;
}

function drawResult(source, detections) {
  const { width, height } = getSourceSize(source);
  const canvas = elements.viewerCanvas;
  const context = canvas.getContext("2d");
  canvas.width = width;
  canvas.height = height;
  context.drawImage(source, 0, 0, width, height);

  const lineWidth = Math.max(3, Math.round(width / 320));
  const fontSize = Math.max(16, Math.round(width / 38));
  context.lineWidth = lineWidth;
  context.font = `700 ${fontSize}px Manrope, sans-serif`;
  context.textBaseline = "top";

  detections.forEach((detection) => {
    const boxWidth = detection.x2 - detection.x1;
    const boxHeight = detection.y2 - detection.y1;
    const label = `Pothole ${Math.round(detection.score * 100)}%`;
    const labelWidth = context.measureText(label).width + 18;
    const labelHeight = fontSize + 12;
    const labelY = Math.max(0, detection.y1 - labelHeight);

    context.strokeStyle = "#cbe82f";
    context.strokeRect(detection.x1, detection.y1, boxWidth, boxHeight);
    context.fillStyle = "#cbe82f";
    context.fillRect(detection.x1, labelY, labelWidth, labelHeight);
    context.fillStyle = "#0b1412";
    context.fillText(label, detection.x1 + 9, labelY + 5);
  });

  canvas.hidden = false;
  elements.emptyState.hidden = true;
}

function updateResults(detections, elapsedMs) {
  const count = detections.length;
  const topScore = count ? Math.max(...detections.map((item) => item.score)) : 0;
  elements.detectionCount.textContent = String(count);
  elements.topConfidence.textContent = count ? `${Math.round(topScore * 100)}%` : "—";
  elements.inferenceTime.textContent = `${Math.round(elapsedMs)} ms`;
  elements.resultMessage.textContent = count
    ? `${count} likely ${count === 1 ? "pothole" : "potholes"} marked in the frame.`
    : "No pothole passed the selected confidence threshold.";
}

async function detect(source, showOverlay = true) {
  if (!state.session || state.inferenceBusy) return;
  state.inferenceBusy = true;
  if (showOverlay) setWorking(true);

  try {
    const prepared = prepareInput(source);
    const startedAt = performance.now();
    const result = await state.session.run({ images: prepared.tensor });
    const elapsedMs = performance.now() - startedAt;
    const detections = parseDetections(result.output0, prepared.transform);
    drawResult(source, detections);
    updateResults(detections, elapsedMs);
  } catch (error) {
    console.error(error);
    elements.resultMessage.textContent = "This frame could not be processed. Try another image.";
  } finally {
    state.inferenceBusy = false;
    setWorking(false);
  }
}

function stopCamera() {
  state.cameraRunning = false;
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  elements.cameraVideo.srcObject = null;
  elements.cameraButton.hidden = false;
  elements.stopButton.hidden = true;
}

async function cameraLoop(timestamp) {
  if (!state.cameraRunning) return;
  if (
    elements.cameraVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
    && timestamp - state.lastCameraFrame >= CAMERA_INTERVAL_MS
    && !state.inferenceBusy
  ) {
    state.lastCameraFrame = timestamp;
    await detect(elements.cameraVideo, false);
  }
  requestAnimationFrame(cameraLoop);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    elements.resultMessage.textContent = "Camera access is not supported in this browser.";
    return;
  }

  try {
    stopCamera();
    state.currentImage = null;
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    elements.cameraVideo.srcObject = state.stream;
    await elements.cameraVideo.play();
    state.cameraRunning = true;
    state.lastCameraFrame = 0;
    elements.cameraButton.hidden = true;
    elements.stopButton.hidden = false;
    requestAnimationFrame(cameraLoop);
  } catch (error) {
    console.error(error);
    elements.resultMessage.textContent = "Camera access was unavailable. Upload an image instead.";
  }
}

function loadImageSource(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The image could not be loaded."));
    image.src = url;
  });
}

async function showImageUrl(url, revokeAfterLoad = false) {
  stopCamera();
  setWorking(true);
  try {
    const image = await loadImageSource(url);
    state.currentImage = image;
    await detect(image);
  } catch (error) {
    console.error(error);
    elements.resultMessage.textContent = "The image could not be opened.";
  } finally {
    if (revokeAfterLoad) URL.revokeObjectURL(url);
    setWorking(false);
  }
}

async function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    elements.resultMessage.textContent = "Choose a JPG, PNG, or WEBP image.";
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    elements.resultMessage.textContent = "Choose an image smaller than 20 MB.";
    return;
  }
  await showImageUrl(URL.createObjectURL(file), true);
}

elements.fileInput.addEventListener("change", async (event) => {
  await handleFile(event.target.files?.[0]);
  event.target.value = "";
});

elements.cameraButton.addEventListener("click", startCamera);
elements.stopButton.addEventListener("click", stopCamera);

elements.confidence.addEventListener("input", () => {
  elements.confidenceValue.textContent = `${elements.confidence.value}%`;
});

elements.confidence.addEventListener("change", async () => {
  if (state.currentImage) await detect(state.currentImage);
});

elements.sampleButtons.forEach((button) => {
  button.addEventListener("click", () => showImageUrl(button.dataset.sample));
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
});

elements.dropZone.addEventListener("drop", (event) => {
  handleFile(event.dataTransfer?.files?.[0]);
});

elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});

window.addEventListener("beforeunload", stopCamera);
loadModel();
