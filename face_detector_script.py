#!/usr/bin/env python3
"""
Face Detection (MediaPipe Tasks) — Image or Webcam
Usage:
  # Image:
  python face_detector_webcam.py --image path/to/image.jpg

  # Webcam (default camera index 0):
  python face_detector_webcam.py --webcam 0

  # Custom model path:
  python face_detector_webcam.py --webcam 0 --model detector.tflite
"""

from __future__ import annotations
import argparse, os, time, math
from typing import Optional, Tuple, Union, List
from urllib.request import urlretrieve

import cv2
import numpy as np
import mediapipe as mp

# Shortcuts (MediaPipe Tasks API)
BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision
FaceDetector = Vision.FaceDetector
FaceDetectorOptions = Vision.FaceDetectorOptions
RunningMode = Vision.RunningMode

# -----------------------------------------------------------------------------
# Model handling
# -----------------------------------------------------------------------------
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite")

def ensure_model(model_path: str) -> str:
    """Download model if missing; return local path."""
    if os.path.exists(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"[model] Downloading model to {model_path} ...")
    urlretrieve(MODEL_URL, model_path)
    print("[model] Done.")
    return model_path

# -----------------------------------------------------------------------------
# Drawing helpers (OpenCV)
# -----------------------------------------------------------------------------
MARGIN = 10        # px
ROW_SIZE = 10      # px
FONT_SIZE = 0.9
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 255)  # BGR (red)

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float,
    image_width: int, image_height: int
) -> Optional[Tuple[int, int]]:
    """Convert normalized (x,y) in [0,1] to integer pixel coords; clamp to bounds."""
    def valid(v: float) -> bool:
        return (v > 0 or math.isclose(v, 0.0)) and (v < 1 or math.isclose(v, 1.0))
    if not (valid(normalized_x) and valid(normalized_y)):
        return None
    x_px = min(int(normalized_x * image_width),  image_width - 1)
    y_px = min(int(normalized_y * image_height), image_height - 1)
    return (x_px, y_px)

def draw_detections_bgr(bgr_image: np.ndarray, detection_result) -> np.ndarray:
    """
    Draw boxes, keypoints, and scores onto a BGR image (in place).
    Returns the annotated image (same array).
    """
    h, w = bgr_image.shape[:2]
    for det in detection_result.detections:
        # Bounding box (already in pixel units)
        bbox = det.bounding_box
        x0, y0 = bbox.origin_x, bbox.origin_y
        x1, y1 = x0 + bbox.width, y0 + bbox.height
        cv2.rectangle(bgr_image, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red box

        # Keypoints (normalized → pixel)
        for kp in det.keypoints:
            pt = _normalized_to_pixel_coordinates(kp.x, kp.y, w, h)
            if pt is not None:
                # cv2.circle(image, center, radius, color, thickness)
                cv2.circle(bgr_image, pt, 2, (0, 255, 0), -1)  # small green dot

        # Label (if present) + score
        cat = det.categories[0] if det.categories else None
        name = "" if (cat is None or cat.category_name is None) else cat.category_name
        score = 0.0 if cat is None else float(cat.score)
        label = f"{name} ({score:.2f})" if name else f"{score:.2f}"
        cv2.putText(bgr_image, label, (x0 + MARGIN, y0 + MARGIN + ROW_SIZE),
                    cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return bgr_image

# -----------------------------------------------------------------------------
# Inference runners
# -----------------------------------------------------------------------------
def run_on_image(model_path: str, image_path: str, show: bool = True, save: Optional[str] = None) -> None:
    """Detect faces on a single image file and optionally display/save."""
    # Build detector (IMAGE mode is default when you use .detect)
    options = FaceDetectorOptions(base_options=BaseOptions(model_asset_path=model_path))
    with FaceDetector.create_from_options(options) as detector:
        # Load image as MediaPipe Image (expects RGB order)
        mp_image = mp.Image.create_from_file(image_path)
        result = detector.detect(mp_image)

        # Convert back to BGR to draw with OpenCV
        rgb = np.copy(mp_image.numpy_view())
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        draw_detections_bgr(bgr, result)

        if save:
            cv2.imwrite(save, bgr)
            print(f"[image] Saved: {save}")
        if show:
            cv2.imshow("Face Detection (image)", bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def run_on_webcam(model_path: str, cam_index: int = 0) -> None:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")

    # Get camera FPS; fall back to 30 if unknown
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:  # NaN or 0 or <1
        fps = 30.0
    ms_per_frame = int(round(1000.0 / fps))
    frame_idx = 0

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO
    )
    with FaceDetector.create_from_options(options) as detector:
        print("[webcam] Press 'q' to quit.")
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            ts_ms = frame_idx * ms_per_frame
            frame_idx += 1

            result = detector.detect_for_video(mp_image, ts_ms)
            draw_detections_bgr(frame_bgr, result)
            cv2.imshow("Face Detection (webcam)", frame_bgr)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="MediaPipe Face Detector — Image or Webcam")
    ap.add_argument("--model", type=str, default="detector.tflite",
                    help="Path to .tflite face detector model (auto-download if missing)")
    ap.add_argument("--image", type=str, default=None,
                    help="Path to an input image. If set, runs image mode.")
    ap.add_argument("--webcam", type=int, default=None,
                    help="Webcam index (e.g., 0). If set, runs webcam mode.")
    ap.add_argument("--save", type=str, default=None,
                    help="Output path for annotated image (image mode only).")
    return ap.parse_args()

def main():
    args = parse_args()
    model_path = ensure_model(args.model)

    if args.image and args.webcam is not None:
        print("Choose either --image or --webcam, not both.")
        return
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(args.image)
        out_path = args.save or os.path.splitext(args.image)[0] + "_faces.jpg"
        run_on_image(model_path, args.image, show=True, save=out_path)
    elif args.webcam is not None:
        run_on_webcam(model_path, args.webcam)
    else:
        print("Nothing to do. Provide --image <path> or --webcam <index>.")
        print("Example: python face_detector_webcam.py --webcam 0")

if __name__ == "__main__":
    main()
