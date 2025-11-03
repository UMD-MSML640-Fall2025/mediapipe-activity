# Face Detection with MediaPipe — README

Detect faces in **images** or from a **live webcam** using the MediaPipe Tasks Face Detector.

---

## 1) Prereqs

* **Python** 3.8–3.12 (3.9+ recommended)
* A working webcam (for live mode)

### Install dependencies

Option A — from `requirements.txt` (recommended):

```bash
pip install -r requirements.txt
```

Option B — install manually:

```bash
pip install mediapipe opencv-python numpy
```

> Tip (Windows/macOS): use a virtual environment
>
> ```bash
> python -m venv .venv
> # Windows
> .venv\Scripts\activate
> # macOS/Linux
> source .venv/bin/activate
> pip install -r requirements.txt
> ```

---

## 2) Files

* `face_detector_script.py` — the runnable script (image + webcam modes)
* `requirements.txt` — dependencies (mediapipe, opencv-python, numpy)

The script automatically **downloads** the face detector model (`detector.tflite`) on first run (to the current folder) if it’s not present.

---

## 3) Usage

### A) Run on a single image

```bash
python face_detector_script.py --image path/to/image.jpg
```

* An annotated copy is displayed and also saved next to the input (e.g., `image_faces.jpg`).
* You can optionally specify where to save:

  ```bash
  python face_detector_script.py --image path/to/image.jpg --save results/annotated.jpg
  ```

### B) Run on a webcam

```bash
python face_detector_script.py --webcam 0
```

* Replace `0` with another index if you have multiple cameras.
* Press **q** (or **Esc**) to quit.

### Optional: custom model path

If you already have a face detector model:

```bash
python face_detector_script.py --webcam 0 --model path/to/detector.tflite
```

---

## 4) Notes about the script

* Uses **MediaPipe Tasks** Face Detector in **VIDEO** mode for webcam
  (`detect_for_video(...)` needs strictly increasing timestamps — handled internally).
* Converts frames BGR↔RGB as needed (OpenCV ↔ MediaPipe).
* Draws:

  * **red** bounding boxes around faces,
  * **green** keypoints (landmarks),
  * detection **score** label.

---

## 5) Troubleshooting

* **“Input timestamp must be monotonically increasing.”**
  You’re likely running a modified script. The provided version uses a strictly increasing timestamp per frame. If you changed it, ensure each call to `detect_for_video(...)` gets a *larger* ms timestamp than the previous one.

* **No camera / blank window**

  * Try a different index: `--webcam 1`, `--webcam 2`, …
  * Close other apps using the camera.
  * Check OS camera permissions (Windows: Privacy & Security → Camera; macOS: System Settings → Privacy & Security → Camera).

* **Import errors**

  * Make sure your venv is active.
  * Reinstall: `pip install -r requirements.txt`

* **Performance**

  * Smaller input images → faster.
  * CPU-only inference is expected (XNNPACK delegate messages are normal).

* **Headless / SSH**

  * Use **image mode** and rely on the saved output file (display may not work without a GUI).

---

## 6) Examples

```bash
# Image mode
python face_detector_script.py --image samples/group.jpg

# Webcam mode, alternate camera
python face_detector_script.py --webcam 1

# Image mode with explicit save path
python face_detector_script.py --image samples/face.jpg --save results/face_annotated.jpg
```

---

## 7) License / Credits

* Model: MediaPipe Face Detector (BlazeFace short-range).
* API: MediaPipe Tasks (Python).
* Visualization: OpenCV.
