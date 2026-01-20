"""
Entry point for the Real-Time Scene Description System.

This script wires together:
- Frame preprocessing
- YOLOv8 object detection
- Human–object interaction inference
- Caption generation
- Video I/O (webcam or file) and on-frame overlay
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2

from .preprocess import enhance_frame
from .detector import YoloDetector
from .interaction import infer_interactions
from .captioner import generate_caption
from .video_utils import (
    open_video_source,
    get_frame_size,
    create_video_writer,
    resize_frame,
    overlay_caption,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Scene Description using YOLOv8 and simple HOI heuristics.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. '0') or path to video file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs/output.mp4",
        help="Path to save the captioned output video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Output frame width. Lower = faster. Default: 320",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Output frame height. Lower = faster. Default: 240",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device for YOLO (e.g. 'cpu', 'cuda'). None lets YOLO choose.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.pt",
        help="Path to YOLOv8 model weights.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=5,
        help="Process every Nth frame for detection (higher = faster, less accurate). Default: 5",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        default=True,
        help="Skip expensive preprocessing for faster performance (default: True).",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable preprocessing (CLAHE) - slower but better for low-light.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="YOLO confidence threshold (lower = faster, more detections). Default: 0.3",
    )
    parser.add_argument(
        "--detection-size",
        type=int,
        default=320,
        help="YOLO detection input size (smaller = faster). Default: 320 (320x320)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target frame rate for output video (default: 30.0 fps).",
    )
    return parser.parse_args()


def _parse_source_arg(source_arg: str) -> int | str:
    """
    Interpret the source argument as an int (webcam index) if possible,
    otherwise as a file path.
    """
    try:
        return int(source_arg)
    except ValueError:
        return source_arg


def run(
    source: int | str,
    output_path: str,
    width: int,
    height: int,
    device: Optional[str],
    model_path: str,
    skip_frames: int = 5,
    no_preprocess: bool = True,
    conf_threshold: float = 0.3,
    detection_size: int = 320,
    target_fps: float = 30.0,
) -> None:
    """
    Execute the full real-time scene description pipeline.
    """
    cap = open_video_source(source)
    frame_size = (width, height)

    # Use the specified target FPS for output video
    # For webcam input, we'll process frames as fast as possible
    # but write them at the target FPS rate
    output_fps = target_fps

    writer = create_video_writer(output_path, fps=output_fps, frame_size=frame_size)
    detector = YoloDetector(
        model_path=model_path, 
        conf_threshold=conf_threshold, 
        device=device,
        imgsz=detection_size,
    )

    window_name = "Real-Time Scene Description (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Performance tracking
    frame_count = 0
    last_fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    # Cache for frame skipping
    cached_detections = []
    cached_caption = "Initializing..."
    cached_interactions = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            fps_counter += 1

            # Normalize frame size early
            frame = resize_frame(frame, frame_size)

            # Only run expensive detection every skip_frames frames
            should_detect = (frame_count % skip_frames == 0) or frame_count == 1

            if should_detect:
                # Preprocessing: skip by default for max speed
                if not no_preprocess:
                    # Only apply lightweight CLAHE if preprocessing is explicitly enabled
                    import numpy as np
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
                else:
                    enhanced = frame  # Skip preprocessing entirely for max speed

                # Run detection on enhanced frame
                detections = detector.detect(enhanced)
                cached_detections = detections

                # Infer human–object interactions
                interactions = infer_interactions(detections)
                cached_interactions = interactions

                # Generate caption
                caption = generate_caption(interactions)
                cached_caption = caption
            else:
                # Reuse cached results for skipped frames
                detections = cached_detections
                interactions = cached_interactions
                caption = cached_caption

            # Overlay caption on the *original* (resized) frame for display
            output_frame = overlay_caption(frame.copy(), caption)

            # Optionally draw bounding boxes for debugging / visualization
            for label, conf, (x1, y1, x2, y2) in detections:
                color = (0, 255, 255) if label.lower() == "person" else (255, 0, 0)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                txt = f"{label} {conf:.2f}"
                cv2.putText(
                    output_frame,
                    txt,
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # Calculate and display FPS
            now = time.time()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:  # Update FPS every second
                current_fps = fps_counter / elapsed
                fps_counter = 0
                last_fps_time = now

            # Draw FPS and performance info
            fps_text = f"FPS: {current_fps:.1f} | Target: {output_fps:.1f} | Process: {frame_count % skip_frames}/{skip_frames}"
            cv2.putText(
                output_frame,
                fps_text,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Show live output
            cv2.imshow(window_name, output_frame)

            # Write to video
            writer.write(output_frame)

            # Don't limit frame rate - let it run as fast as possible for max FPS
            # (Frame rate limiting removed for performance - video writer handles timing)

            # Clean exit on 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    source = _parse_source_arg(args.source)
    
    # Handle preprocessing flags: --preprocess overrides default --no-preprocess
    use_preprocess = args.preprocess if args.preprocess else not args.no_preprocess
    
    run(
        source=source,
        output_path=args.output,
        width=args.width,
        height=args.height,
        device=args.device,
        model_path=args.model,
        skip_frames=args.skip_frames,
        no_preprocess=not use_preprocess,
        conf_threshold=args.conf_threshold,
        detection_size=args.detection_size,
        target_fps=args.fps,
    )


if __name__ == "__main__":
    main()


