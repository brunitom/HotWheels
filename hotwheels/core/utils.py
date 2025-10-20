"""Utility functions for HotWheels detection and dataset management.

This module provides helper functions for file operations, path handling,
atomic writes, hashing, and other common utilities.
"""

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def safe_filename(filename: str, max_length: int = 255) -> str:
    """Create a safe filename by removing/replacing invalid characters.

    Args:
        filename: Original filename.
        max_length: Maximum filename length.

    Returns:
        Safe filename with invalid characters replaced.
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    safe_name = safe_name.strip('. ')
    
    # Truncate if too long
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:max_length - len(ext)] + ext
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed"
    
    return safe_name


def atomic_write(file_path: Union[str, Path], content: str, mode: str = "w") -> None:
    """Atomically write content to a file using a temporary file.

    Args:
        file_path: Target file path.
        content: Content to write.
        mode: File mode ('w' for text, 'wb' for binary).

    Raises:
        OSError: If atomic write fails.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode=mode,
        dir=file_path.parent,
        delete=False,
        prefix=f".{file_path.name}.",
        suffix=".tmp"
    ) as tmp_file:
        if mode == "w":
            tmp_file.write(content)
        else:
            tmp_file.write(content.encode() if isinstance(content, str) else content)
        tmp_path = tmp_file.name
    
    try:
        # Atomic move
        os.replace(tmp_path, file_path)
    except OSError:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """Calculate hash of a file.

    Args:
        file_path: Path to file.
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256').

    Returns:
        Hexadecimal hash string.

    Raises:
        ValueError: If algorithm is not supported.
        FileNotFoundError: If file doesn't exist.
    """
    if algorithm not in ["md5", "sha1", "sha256"]:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def calculate_array_hash(array: np.ndarray, algorithm: str = "md5") -> str:
    """Calculate hash of a numpy array.

    Args:
        array: Numpy array.
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256').

    Returns:
        Hexadecimal hash string.

    Raises:
        ValueError: If algorithm is not supported.
    """
    if algorithm not in ["md5", "sha1", "sha256"]:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(array.tobytes())
    return hash_obj.hexdigest()


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.

    Raises:
        OSError: If directory cannot be created.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_next_filename(base_path: Union[str, Path], prefix: str = "img", extension: str = ".jpg") -> Path:
    """Get next available filename with sequential numbering.

    Args:
        base_path: Directory to create file in.
        prefix: Filename prefix.
        extension: File extension.

    Returns:
        Path to next available filename.
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    counter = 1
    while True:
        filename = f"{prefix}_{counter:04d}{extension}"
        file_path = base_path / filename
        if not file_path.exists():
            return file_path
        counter += 1


def normalize_coordinates(
    x1: int, y1: int, x2: int, y2: int, 
    img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """Convert pixel coordinates to normalized YOLO format.

    Args:
        x1, y1, x2, y2: Bounding box coordinates in pixels.
        img_width, img_height: Image dimensions.

    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1].

    Raises:
        ValueError: If coordinates are invalid.
    """
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    
    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
    
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def denormalize_coordinates(
    x_center: float, y_center: float, width: float, height: float,
    img_width: int, img_height: int
) -> tuple[int, int, int, int]:
    """Convert normalized YOLO coordinates to pixel coordinates.

    Args:
        x_center, y_center, width, height: Normalized coordinates [0, 1].
        img_width, img_height: Image dimensions.

    Returns:
        Tuple of (x1, y1, x2, y2) in pixels.

    Raises:
        ValueError: If coordinates are not normalized.
    """
    if not all(0 <= coord <= 1 for coord in [x_center, y_center, width, height]):
        raise ValueError("Coordinates must be normalized [0, 1]")
    
    # Convert to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate corner coordinates
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    
    return x1, y1, x2, y2


def set_random_seed(seed: Optional[int] = None) -> int:
    """Set random seed for reproducible results.

    Args:
        seed: Random seed. If None, uses current time.

    Returns:
        The seed value used.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    random.seed(seed)
    np.random.seed(seed)
    
    return seed


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted size string (e.g., "1.5 MB").
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_image_info(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about an image file.

    Args:
        image_path: Path to image file.

    Returns:
        Dictionary with image information.

    Raises:
        FileNotFoundError: If image doesn't exist.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Get file info
    stat = image_path.stat()
    
    # Try to get image dimensions
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            height, width, channels = img.shape
        else:
            width = height = channels = None
    except ImportError:
        width = height = channels = None
    
    return {
        "path": str(image_path),
        "size_bytes": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "width": width,
        "height": height,
        "channels": channels,
        "modified": stat.st_mtime,
    }
