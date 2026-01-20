"""
Preprocessing utilities for enhancing video frames.

This module implements:
- Noise reduction using OpenCV's fastNlMeansDenoisingColored.
- Contrast enhancement using CLAHE in LAB color space.

The main entry point is `enhance_frame`, which takes a BGR frame
from OpenCV and returns an enhanced frame suitable for downstream
object detection in challenging lighting conditions.
"""

from __future__ import annotations

import cv2
import numpy as np


def denoise_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply fast non-local means denoising to a BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        Denoised BGR image.
    """
    # h and hColor control strength of luminance and color filtering.
    # Values are chosen to balance noise removal and detail preservation.
    return cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def enhance_contrast_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Enhance contrast using CLAHE in LAB color space.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Clip limit and tile grid chosen for robust enhancement without over-amplifying noise.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Enhance an input frame for robust downstream detection.

    The pipeline applies:
    1. Denoising
    2. CLAHE-based contrast enhancement

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        Enhanced BGR image.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty or None.")

    denoised = denoise_frame(frame)
    enhanced = enhance_contrast_clahe(denoised)
    return enhanced


__all__ = ["enhance_frame", "denoise_frame", "enhance_contrast_clahe"]


