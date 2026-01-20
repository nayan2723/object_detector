"""
Video utilities for handling input capture and output writing.

This module provides helpers to:
- Open webcam or video file streams.
- Resize frames consistently.
- Overlay captions.
- Write processed frames to an output video file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def open_video_source(source: int | str = 0) -> cv2.VideoCapture:
    """
    Open a video capture from webcam index or file path.

    Parameters
    ----------
    source : int or str
        Webcam index (e.g., 0) or path to video file.

    Returns
    -------
    cv2.VideoCapture
        Opened video capture object.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return cap


def get_frame_size(
    cap: cv2.VideoCapture,
    target_size: Tuple[int, int] | None = (640, 480),
) -> Tuple[int, int]:
    """
    Determine frame size for processing and output.
    """
    if target_size is not None:
        return target_size

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        # Fallback to a safe default
        return 640, 480
    return width, height


def create_video_writer(
    output_path: str | Path,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    """
    Create a video writer for saving processed frames.

    Parameters
    ----------
    output_path : str or Path
        Target output file path (e.g., data/outputs/output.mp4).
    fps : float
        Frames per second for output video.
    frame_size : (int, int)
        Frame size as (width, height).

    Returns
    -------
    cv2.VideoWriter
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer at: {output_path}")
    return writer


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a BGR frame to the given size (width, height).
    """
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def overlay_caption(
    frame: np.ndarray,
    caption: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a caption string on top of a frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image.
    caption : str
        Text to overlay.
    position : (int, int)
        Top-left coordinate for the text.
    font_scale : float
        Scale of the font.
    color : (int, int, int)
        Text color in BGR.
    thickness : int
        Line thickness.
    """
    if not caption:
        return frame

    # Add a background rectangle for better readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(
        frame,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + baseline + 5),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(frame, caption, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


__all__ = [
    "open_video_source",
    "get_frame_size",
    "create_video_writer",
    "resize_frame",
    "overlay_caption",
]


