"""YOLO model handling and inference for HotWheels detection.

This module provides YOLO model loading, device selection, and inference
functionality with proper error handling and device optimization.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

# Import lazily to provide a clearer error if ultralytics isn't installed
try:
    from ultralytics import YOLO
except ImportError as exc:
    print(
        "Ultralytics (YOLOv8) is required. Install with: pip install ultralytics\n"
        "If OpenCV is missing: pip install opencv-python\n"
        "If you are on Apple Silicon and want GPU (MPS) acceleration: pip install 'torch>=2.1' 'torchvision>=0.16' --index-url https://download.pytorch.org/whl/cu121\n",
        file=sys.stderr,
    )
    raise


def load_model(model_path: Union[str, Path]) -> YOLO:
    """Load a YOLO model from the specified path.

    Args:
        model_path: Path to the YOLO model file (.pt).

    Returns:
        Loaded YOLO model instance.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        RuntimeError: If model cannot be loaded.
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Train with ultralytics first or provide correct model path"
        )
    
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_path}': {e}") from e


def load_names(names_path: Optional[Union[str, Path]], model: YOLO) -> Dict[int, str]:
    """Load class names, prioritizing an external file if provided.

    Args:
        names_path: Optional path to a newline-delimited names file.
        model: Loaded YOLO model instance.

    Returns:
        Dictionary mapping class indices to names.

    Raises:
        FileNotFoundError: If names file is specified but doesn't exist.
    """
    if names_path:
        p = Path(names_path)
        if not p.exists():
            raise FileNotFoundError(f"Names file not found: {p}")
        
        names = {
            i: line.strip() 
            for i, line in enumerate(p.read_text().splitlines()) 
            if line.strip()
        }
        return names
    
    # Model has names attribute after loading weights
    return getattr(model, "names", {}) or {}


def get_device_string(device: str) -> Optional[str]:
    """Map CLI device option to ultralytics device argument.

    Args:
        device: Device string ('auto', 'cpu', 'mps', 'cuda').

    Returns:
        Device string for ultralytics, or None for auto-detection.
    """
    if device == "auto":
        return None
    return device


def predict(
    model: YOLO,
    source: np.ndarray,
    conf: float = 0.5,
    iou: float = 0.45,
    imgsz: int = 640,
    device: Optional[str] = None,
    verbose: bool = False,
) -> List[Any]:
    """Run YOLO inference on an image.

    Args:
        model: Loaded YOLO model instance.
        source: Input image as numpy array.
        conf: Confidence threshold for detections.
        iou: IoU threshold for NMS.
        imgsz: Inference image size.
        device: Device to run inference on.
        verbose: Whether to print verbose output.

    Returns:
        List of YOLO Results objects.

    Raises:
        RuntimeError: If inference fails.
    """
    try:
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=verbose,
        )
        return results
    except Exception as e:
        raise RuntimeError(f"Inference error: {e}") from e


def get_model_info(model: YOLO) -> Dict[str, Any]:
    """Get information about the loaded model.

    Args:
        model: Loaded YOLO model instance.

    Returns:
        Dictionary containing model information.
    """
    info = {
        "model_name": getattr(model, "model_name", "Unknown"),
        "names": getattr(model, "names", {}),
        "num_classes": len(getattr(model, "names", {})),
    }
    
    # Try to get model size if available
    try:
        if hasattr(model, "model"):
            info["model_size"] = sum(p.numel() for p in model.model.parameters())
    except Exception:
        info["model_size"] = "Unknown"
    
    return info


def validate_model_compatibility(model: YOLO, expected_classes: Optional[List[str]] = None) -> bool:
    """Validate that the model is compatible with expected classes.

    Args:
        model: Loaded YOLO model instance.
        expected_classes: List of expected class names.

    Returns:
        True if model is compatible, False otherwise.
    """
    if expected_classes is None:
        return True
    
    model_names = getattr(model, "names", {})
    model_classes = set(model_names.values())
    expected_set = set(expected_classes)
    
    return expected_set.issubset(model_classes)
