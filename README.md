# Face Detection with MediaPipe — README

Detect faces in **images**, **videos**, or from a **live webcam** using the MediaPipe Tasks Face Detector.

---

## 1) Prereqs

* **Python** 3.8–3.12 (3.9+ recommended)
* A working webcam (for live mode)

### Install dependencies

**Option A — from `requirements.txt` (recommended):**

```bash
pip install -r requirements.txt
```

**Option B — install manually:**

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

* `face_detector_script.py` — the runnable script (image / video / webcam)
* `requirements.txt` — dependencies (mediapipe, opencv-python, numpy)

The script automatically **downloads** the face detector model (`detector.tflite`) on first run (to the current folder) if it’s not present.

---

## 3) Usage

> Choose exactly one of `--image`, `--video`, or `--webcam`.

### A) Run on a single image

```bash
python face_detector_script.py --image path/to/image.jpg
```

* An annotated copy is displayed and also saved next to the input (e.g., `image_faces.jpg`).
* You can optionally specify where to save:

```bash
python face_detector_script.py --image path/to/image.jpg --save annotated.jpg
```

### B) Run on a video file

```bash
python face_detector_script.py --video path/to/video.mp4
```

* Shows a **scaled preview** window (to avoid giant windows).
* Optionally save an **annotated MP4**:

```bash
python face_detector_script.py --video path/to/video.mp4 --save annotated.mp4
```

### C) Run on a webcam

```bash
python face_detector_script.py --webcam 0
```

* Replace `0` if you have multiple cameras (try `1`, `2`, …).
* Press **q** (or **Esc**) to quit.

### Optional: custom model path

If you already have a face detector model:

```bash
python face_detector_script.py --webcam 0 --model detector.tflite
```

---

## 4) Notes about the script

* Uses **MediaPipe Tasks** Face Detector:

  * **IMAGE** mode for `--image`
  * **VIDEO** mode for `--video` and `--webcam` (ensures strictly increasing timestamps)
* Converts frames BGR↔RGB as needed (OpenCV ↔ MediaPipe).
* Draws:

  * **red** bounding boxes around faces,
  * **green** keypoints (landmarks),
  * detection **score** label.
* **Video preview** is auto-scaled (defaults to max ~1280×720). Saved video keeps original resolution.

---

## 5) Troubleshooting

* **“Input timestamp must be monotonically increasing.”**
  Ensure you didn’t modify the internal timestamp logic; each frame must use a strictly increasing timestamp for `detect_for_video(...)`.

* **No camera / blank window**

  * Try another index: `--webcam 1`, `--webcam 2`, …
  * Close other apps using the camera.
  * Check OS camera permissions (Windows: Privacy & Security → Camera; macOS: System Settings → Privacy & Security → Camera).

* **Import errors**

  * Activate your venv.
  * Reinstall: `pip install -r requirements.txt`.

* **Performance**

  * Smaller inputs run faster.
  * CPU-only inference is expected (XNNPACK delegate messages are normal).

* **Headless / SSH**

  * Use **image** or **video** mode with `--save` and review the saved output.

---

## 6) Examples

```bash
# Image mode
python face_detector_script.py --image people.jpg

# Video mode, with saved annotated output
python face_detector_script.py --video people.mp4 --save people_annotated.mp4

# Webcam mode, alternate camera
python face_detector_script.py --webcam 1

# Image mode with explicit save path
python face_detector_script.py --image people.jpg --save face_annotated.jpg
```

---

## 7) License / Credits

* Model: MediaPipe Face Detector (BlazeFace short-range).
* API: MediaPipe Tasks (Python).
* Visualization: OpenCV.
