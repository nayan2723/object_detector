"""
Object detection module using Ultralytics YOLOv8.

This module exposes a `YoloDetector` class that loads a YOLOv8 model once
and performs batched or per-frame inference. Detections are returned in a
simple, model-agnostic format:

    (label, confidence, (x1, y1, x2, y2))

where coordinates are absolute pixel values in the original frame space.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


Detection = Tuple[str, float, Tuple[int, int, int, int]]


class YoloDetector:
    """
    Thin wrapper around Ultralytics YOLOv8 for real-time object detection.
    """

    def __init__(
        self,
        model_path: str | Path = "models/yolov8n.pt",
        conf_threshold: float = 0.4,
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        """
        Parameters
        ----------
        model_path : str or Path
            Path to the YOLOv8 model weights.
        conf_threshold : float
            Minimum confidence threshold for detections.
        device : str, optional
            Device to run inference on (e.g., "cpu", "cuda"). If None,
            Ultralytics will select automatically.
        imgsz : int
            Input image size for YOLO (smaller = faster). Default: 640.
        """
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz

        # Load model once. If weights are missing locally, Ultralytics can
        # fetch official weights (e.g., "yolov8n.pt") automatically.
        weights_path = Path(self.model_path)
        if weights_path.exists():
            self.model = YOLO(str(weights_path))
        else:
            # Allow end-to-end execution without a manual download step.
            # If the user passed the default "models/yolov8n.pt", fall back
            # to Ultralytics' built-in download-and-cache behavior.
            fallback = weights_path.name or "yolov8n.pt"
            if fallback.lower() != "yolov8n.pt":
                raise FileNotFoundError(
                    f"Model weights not found at '{self.model_path}'. "
                    "Place the .pt file at that path or pass --model to a valid weights file."
                )
            self.model = YOLO("yolov8n.pt")
        
        # Verify model has access to all classes
        self.num_classes = len(self.model.names)
        
    def get_all_classes(self) -> dict[int, str]:
        """
        Get all class names available in the model.
        
        Returns
        -------
        dict[int, str]
            Dictionary mapping class ID to class name.
        """
        return self.model.names.copy()
    
    def get_class_count(self) -> int:
        """
        Get the number of classes the model can detect.
        
        Returns
        -------
        int
            Number of classes (should be 80 for COCO dataset).
        """
        return self.num_classes

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array (H, W, 3).

        Returns
        -------
        List[Detection]
            List of detections as (label, confidence, (x1, y1, x2, y2)).
        """
        if frame is None or frame.size == 0:
            return []

        # Run inference with optimized settings for speed
        results = self.model.predict(
            source=frame,  # ndarray inference
            verbose=False,
            device=self.device,
            conf=self.conf_threshold,
            imgsz=self.imgsz,  # Smaller size = faster inference
            half=False,  # FP16 not always faster on CPU
        )

        detections: List[Detection] = []

        if not results:
            return detections

        # YOLO returns a list of Results; for single image we use the first.
        result = results[0]
        boxes = result.boxes

        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])

            # xyxy in absolute coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                (
                    label,
                    conf,
                    (int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections


__all__ = ["YoloDetector", "Detection"]


