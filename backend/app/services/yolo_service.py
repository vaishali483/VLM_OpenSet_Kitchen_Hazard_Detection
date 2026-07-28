"""YOLO object-detection service."""

from __future__ import annotations

import threading
import time
from io import BytesIO

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core.settings import (
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL_PATH,
    YOLO_RELEVANT_CLASSES,
)
from app.schemas.live_analysis import (
    NormalizedBoundingBox,
    YoloDetection,
)


_model: YOLO | None = None
_model_lock = threading.Lock()


def load_yolo_model() -> None:
    """Load the YOLO model once for all requests."""

    global _model

    if _model is not None:
        return

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}"
        )

    print(f"Loading YOLO model: {YOLO_MODEL_PATH.name}")

    _model = YOLO(str(YOLO_MODEL_PATH))

    print("YOLO model loaded.")


def warm_up_yolo() -> None:
    """
    Run one blank inference during backend startup.

    This prevents the first Unity request from paying the full
    model initialisation cost.
    """

    load_yolo_model()

    blank_image = np.zeros(
        (
            YOLO_IMAGE_SIZE,
            YOLO_IMAGE_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    with _model_lock:
        assert _model is not None

        _model.predict(
            source=blank_image,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            device=YOLO_DEVICE,
            verbose=False,
        )

    print("YOLO warm-up complete.")


def unload_yolo_model() -> None:
    """Release the global model reference during shutdown."""

    global _model
    _model = None


def detect_objects(
    image_bytes: bytes,
) -> tuple[list[YoloDetection], float]:
    """
    Run YOLO on one image.

    Returns:
        detections,
        inference time in milliseconds.
    """

    load_yolo_model()

    with Image.open(BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")
        image_array = np.asarray(rgb_image)

    image_height, image_width = image_array.shape[:2]

    start_time = time.perf_counter()

    with _model_lock:
        assert _model is not None

        results = _model.predict(
            source=image_array,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            device=YOLO_DEVICE,
            verbose=False,
        )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    if not results:
        return [], elapsed_ms

    result = results[0]

    if result.boxes is None:
        return [], elapsed_ms

    xyxy_values = result.boxes.xyxy.cpu().tolist()
    confidence_values = result.boxes.conf.cpu().tolist()
    class_values = result.boxes.cls.cpu().tolist()

    raw_detections: list[dict[str, object]] = []

    for bbox, confidence, class_index in zip(
        xyxy_values,
        confidence_values,
        class_values,
    ):
        class_name = result.names[int(class_index)]

        if class_name not in YOLO_RELEVANT_CLASSES:
            continue

        x_min, y_min, x_max, y_max = bbox

        raw_detections.append(
            {
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox_normalized": {
                    "x_min": max(
                        0.0,
                        min(1.0, x_min / image_width),
                    ),
                    "y_min": max(
                        0.0,
                        min(1.0, y_min / image_height),
                    ),
                    "x_max": max(
                        0.0,
                        min(1.0, x_max / image_width),
                    ),
                    "y_max": max(
                        0.0,
                        min(1.0, y_max / image_height),
                    ),
                },
            }
        )

    # Highest confidence first gives stable detection IDs.
    raw_detections.sort(
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )

    detections: list[YoloDetection] = []

    for index, item in enumerate(
        raw_detections,
        start=1,
    ):
        detections.append(
            YoloDetection(
                detection_id=f"D{index:03d}",
                class_name=str(item["class_name"]),
                confidence=float(item["confidence"]),
                bbox_normalized=NormalizedBoundingBox(
                    **item["bbox_normalized"]
                ),
            )
        )

    return detections, elapsed_ms