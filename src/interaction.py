"""
Human–object interaction inference.

Given raw object detections, this module identifies persons and nearby /
overlapping objects and assigns simple interaction types such as
\"holding\" or \"using\".

The heuristic is intentionally lightweight for real-time performance and
does not require a separate HOI model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .detector import Detection


@dataclass
class Interaction:
    """
    Structured representation of a human–object interaction.
    """

    person_box: Tuple[int, int, int, int]
    object_box: Tuple[int, int, int, int]
    object_label: str
    interaction_type: str  # e.g. "holding", "using"


def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection-over-Union (IoU) between two boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _classify_interaction_type(object_label: str) -> str:
    """
    Heuristic mapping from object label to interaction type.
    """
    label_lower = object_label.lower()

    using_like = {
        "laptop",
        "keyboard",
        "mouse",
        "remote",
        "cell phone",
        "mobile phone",
        "tv",
        "tvmonitor",
        "monitor",
        "screen",
        "microwave",
        "oven",
        "toothbrush",
    }

    if any(token in label_lower for token in using_like):
        return "using"

    # Default to holding
    return "holding"


def infer_interactions(
    detections: List[Detection],
    iou_threshold: float = 0.1,
) -> List[Interaction]:
    """
    Infer human–object interactions from raw detections.

    Parameters
    ----------
    detections : List[Detection]
        List of detections (label, confidence, (x1, y1, x2, y2)).
    iou_threshold : float
        Minimum IoU between person and object boxes to consider an interaction.

    Returns
    -------
    List[Interaction]
        Structured list of interactions.
    """
    persons: List[Tuple[int, int, int, int]] = []
    objects: List[Tuple[str, Tuple[int, int, int, int]]] = []

    for label, _conf, box in detections:
        if label.lower() in {"person"}:
            persons.append(box)
        else:
            objects.append((label, box))

    interactions: List[Interaction] = []

    for person_box in persons:
        for object_label, object_box in objects:
            iou = _compute_iou(person_box, object_box)
            if iou >= iou_threshold:
                interaction_type = _classify_interaction_type(object_label)
                interactions.append(
                    Interaction(
                        person_box=person_box,
                        object_box=object_box,
                        object_label=object_label,
                        interaction_type=interaction_type,
                    )
                )

    return interactions


__all__ = ["Interaction", "infer_interactions"]


