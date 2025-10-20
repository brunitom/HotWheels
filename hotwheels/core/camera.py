"""Camera handling and FPS estimation for HotWheels detection.

This module provides camera capture functionality optimized for macOS with
AVFoundation backend support and graceful fallback handling.
"""

import cv2
from typing import Optional, Tuple


def make_video_capture(camera_index: int, backend: str) -> cv2.VideoCapture:
    """Create a VideoCapture with a macOS-friendly backend.

    On macOS, using CAP_AVFOUNDATION often improves camera access reliability.

    Args:
        camera_index: Camera device index (0 is usually built-in camera).
        backend: Video backend to use ('auto', 'avfoundation', or 'default').

    Returns:
        OpenCV VideoCapture object.

    Raises:
        RuntimeError: If camera cannot be opened with any backend.
    """
    if backend == "avfoundation":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    elif backend == "default":
        cap = cv2.VideoCapture(camera_index)
    else:  # auto
        # Try AVFoundation first, then fallback to default
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index} with backend '{backend}'.\n"
            "On macOS, grant camera permission to your Terminal/IDE.\n"
            "Try --backend avfoundation or a different --camera index."
        )
    
    return cap


def estimate_fps(prev_ticks: Optional[int], curr_ticks: int) -> Tuple[Optional[float], int]:
    """Estimate FPS using cv2.getTickCount/getTickFrequency.

    Args:
        prev_ticks: Previous tick count from cv2.getTickCount().
        curr_ticks: Current tick count from cv2.getTickCount().

    Returns:
        Tuple of (estimated_fps, current_ticks) for next iteration.
        FPS will be None if calculation is not possible.
    """
    if prev_ticks is None:
        return None, curr_ticks
    
    if curr_ticks <= prev_ticks:
        return None, curr_ticks
    
    dt = (curr_ticks - prev_ticks) / cv2.getTickFrequency()
    fps = 1.0 / dt if dt > 0 else None
    return fps, curr_ticks


def get_available_cameras(max_cameras: int = 5) -> list[int]:
    """Find available camera indices by testing each one.

    Args:
        max_cameras: Maximum number of cameras to test.

    Returns:
        List of working camera indices.
    """
    available = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    
    return available


def validate_camera_access(camera_index: int, backend: str = "auto") -> bool:
    """Validate that a camera can be accessed with the given backend.

    Args:
        camera_index: Camera device index to test.
        backend: Video backend to test.

    Returns:
        True if camera is accessible, False otherwise.
    """
    try:
        cap = make_video_capture(camera_index, backend)
        ret, _ = cap.read()
        cap.release()
        return ret
    except RuntimeError:
        return False
