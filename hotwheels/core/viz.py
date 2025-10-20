"""Visualization utilities for HotWheels detection and labeling.

This module provides drawing utilities for bounding boxes, labels, FPS display,
and other visual overlays for both detection and labeling interfaces.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    detections: List[Any],
    names: Dict[int, str],
    show_fps: bool = False,
    fps: Optional[float] = None,
    conf_threshold: float = 0.5,
    box_color: Tuple[int, int, int] = (0, 140, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes, labels, and optional FPS on the frame.

    Args:
        frame: BGR image array.
        detections: Ultralytics YOLO results per frame (first item expected).
        names: Mapping from class id to name.
        show_fps: Whether to overlay FPS.
        fps: Latest FPS estimate.
        conf_threshold: Minimum confidence for drawing.
        box_color: BGR color for bounding boxes.
        text_color: BGR color for text.
        font_scale: Font scale for labels.
        thickness: Line thickness for boxes and text.

    Returns:
        Annotated frame with detections drawn.
    """
    annotated = frame.copy()
    
    if detections is None or len(detections) == 0:
        if show_fps and fps is not None:
            _draw_fps(annotated, fps)
        return annotated

    # YOLOv8 returns a Results list; we take the first result per image
    result = detections[0]
    boxes = result.boxes if hasattr(result, "boxes") else None
    
    if boxes is None or len(boxes) == 0:
        # Optionally indicate no detections
        cv2.putText(
            annotated,
            "No cars detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            thickness,
        )
        if show_fps and fps is not None:
            _draw_fps(annotated, fps, y_offset=60)
        return annotated

    for box in boxes:
        # xyxy, confidence, class id
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        cls_id = int(box.cls[0]) if box.cls is not None else -1

        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, xyxy)
        label = names.get(cls_id, f"id_{cls_id}")
        text = f"{label} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
        
        # Draw label background for readability
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_h - baseline - 4)),
            (x1 + text_w + 4, y1),
            box_color,
            -1,
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            text,
            (x1 + 2, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
        )

    if show_fps and fps is not None:
        _draw_fps(annotated, fps)

    return annotated


def _draw_fps(frame: np.ndarray, fps: float, y_offset: int = 30) -> None:
    """Draw FPS text on frame.

    Args:
        frame: BGR image array to draw on.
        fps: FPS value to display.
        y_offset: Y position offset for text.
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )


def draw_bounding_box(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw a single bounding box on the frame.

    Args:
        frame: BGR image array.
        x1, y1, x2, y2: Bounding box coordinates.
        label: Optional label text.
        color: BGR color for the box.
        thickness: Line thickness.
        font_scale: Font scale for label.

    Returns:
        Frame with bounding box drawn.
    """
    annotated = frame.copy()
    
    # Draw rectangle
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_h - baseline - 4)),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )
    
    return annotated


def draw_crosshair(frame: np.ndarray, x: int, y: int, size: int = 20) -> np.ndarray:
    """Draw a crosshair at the specified position.

    Args:
        frame: BGR image array.
        x, y: Crosshair center coordinates.
        size: Crosshair size (half-length of each line).

    Returns:
        Frame with crosshair drawn.
    """
    annotated = frame.copy()
    color = (0, 255, 255)  # Yellow
    
    # Horizontal line
    cv2.line(annotated, (x - size, y), (x + size, y), color, 2)
    # Vertical line
    cv2.line(annotated, (x, y - size), (x, y + size), color, 2)
    
    return annotated


def draw_instructions(
    frame: np.ndarray,
    instructions: List[str],
    position: Tuple[int, int] = (10, 10),
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw instruction text on the frame.

    Args:
        frame: BGR image array.
        instructions: List of instruction strings.
        position: Starting position (x, y) for text.
        font_scale: Font scale for text.
        color: BGR color for text.
        thickness: Text thickness.

    Returns:
        Frame with instructions drawn.
    """
    annotated = frame.copy()
    x, y = position
    
    for instruction in instructions:
        cv2.putText(
            annotated,
            instruction,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )
        y += int(25 * font_scale)  # Line spacing
    
    return annotated


def create_overlay(
    frame: np.ndarray,
    alpha: float = 0.3,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Create a semi-transparent overlay on the frame.

    Args:
        frame: BGR image array.
        alpha: Transparency level (0.0 = transparent, 1.0 = opaque).
        color: BGR color for overlay.

    Returns:
        Frame with overlay applied.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
