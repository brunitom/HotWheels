"""Dataset utilities for YOLO format I/O and validation.

This module provides functionality for reading/writing YOLO format datasets,
validation, and metadata management for HotWheels detection training.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml


class YOLODataset:
    """YOLO format dataset handler with validation and metadata support."""

    def __init__(self, dataset_path: Union[str, Path]):
        """Initialize dataset handler.

        Args:
            dataset_path: Path to dataset root directory.
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        self.classes_file = self.dataset_path / "classes.txt"
        self.data_yaml = self.dataset_path / "data.yaml"

    def create_structure(self, splits: List[str] = None) -> None:
        """Create dataset directory structure.

        Args:
            splits: List of dataset splits to create (default: ['train', 'val']).
        """
        if splits is None:
            splits = ["train", "val"]

        # Create main directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Create split directories
        for split in splits:
            (self.images_dir / split).mkdir(exist_ok=True)
            (self.labels_dir / split).mkdir(exist_ok=True)

    def save_image(
        self,
        image: np.ndarray,
        filename: str,
        split: str = "train",
        quality: int = 95,
    ) -> Path:
        """Save image to dataset.

        Args:
            image: Image array to save.
            filename: Filename (without extension).
            split: Dataset split ('train' or 'val').
            quality: JPEG quality (1-100).

        Returns:
            Path to saved image file.
        """
        split_dir = self.images_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = split_dir / f"{filename}.jpg"
        
        # Save with specified quality
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return image_path

    def save_labels(
        self,
        labels: List[Tuple[int, float, float, float, float]],
        filename: str,
        split: str = "train",
    ) -> Path:
        """Save YOLO format labels to dataset.

        Args:
            labels: List of (class_id, x_center, y_center, width, height) tuples.
            filename: Filename (without extension).
            split: Dataset split ('train' or 'val').

        Returns:
            Path to saved label file.

        Raises:
            ValueError: If labels are not properly normalized.
        """
        split_dir = self.labels_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        label_path = split_dir / f"{filename}.txt"
        
        # Validate labels
        for label in labels:
            if len(label) != 5:
                raise ValueError(f"Invalid label format: {label}")
            class_id, x, y, w, h = label
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                raise ValueError(f"Labels must be normalized [0,1]: {label}")
        
        # Write labels
        with open(label_path, "w") as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
        
        return label_path

    def load_labels(self, filename: str, split: str = "train") -> List[Tuple[int, float, float, float, float]]:
        """Load YOLO format labels from dataset.

        Args:
            filename: Filename (without extension).
            split: Dataset split ('train' or 'val').

        Returns:
            List of (class_id, x_center, y_center, width, height) tuples.

        Raises:
            FileNotFoundError: If label file doesn't exist.
        """
        label_path = self.labels_dir / split / f"{filename}.txt"
        
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                labels.append((class_id, *coords))
        
        return labels

    def save_classes(self, classes: List[str]) -> Path:
        """Save class names to classes.txt.

        Args:
            classes: List of class names.

        Returns:
            Path to classes.txt file.
        """
        with open(self.classes_file, "w") as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        
        return self.classes_file

    def load_classes(self) -> List[str]:
        """Load class names from classes.txt.

        Returns:
            List of class names.

        Raises:
            FileNotFoundError: If classes.txt doesn't exist.
        """
        if not self.classes_file.exists():
            raise FileNotFoundError(f"Classes file not found: {self.classes_file}")
        
        with open(self.classes_file, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        
        return classes

    def save_data_yaml(self, classes: List[str], train_split: str = "train", val_split: str = "val") -> Path:
        """Save data.yaml configuration file.

        Args:
            classes: List of class names.
            train_split: Training split name.
            val_split: Validation split name.

        Returns:
            Path to data.yaml file.
        """
        data = {
            "path": str(self.dataset_path.absolute()),
            "train": f"images/{train_split}",
            "val": f"images/{val_split}",
            "names": {i: name for i, name in enumerate(classes)},
        }
        
        with open(self.data_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        
        return self.data_yaml

    def save_metadata(
        self,
        filename: str,
        split: str,
        metadata: Dict[str, Any],
    ) -> Path:
        """Save metadata for an image/label pair.

        Args:
            filename: Base filename (without extension).
            split: Dataset split.
            metadata: Metadata dictionary to save.

        Returns:
            Path to metadata file.
        """
        metadata_dir = self.dataset_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = metadata_dir / f"{filename}.json"
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return metadata_path

    def load_metadata(self, filename: str, split: str) -> Optional[Dict[str, Any]]:
        """Load metadata for an image/label pair.

        Args:
            filename: Base filename (without extension).
            split: Dataset split.

        Returns:
            Metadata dictionary, or None if not found.
        """
        metadata_path = self.dataset_path / "metadata" / f"{filename}.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, "r") as f:
            return json.load(f)

    def validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset integrity and return validation report.

        Returns:
            Dictionary containing validation results and statistics.
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }
        
        # Check directory structure
        if not self.images_dir.exists():
            report["errors"].append("Images directory missing")
            report["valid"] = False
        
        if not self.labels_dir.exists():
            report["errors"].append("Labels directory missing")
            report["valid"] = False
        
        if not self.classes_file.exists():
            report["warnings"].append("Classes file missing")
        
        # Count files per split
        splits = ["train", "val"]
        for split in splits:
            images_split = self.images_dir / split
            labels_split = self.labels_dir / split
            
            if images_split.exists():
                image_files = list(images_split.glob("*.jpg"))
                report["stats"][f"{split}_images"] = len(image_files)
                
                # Check for missing labels
                if labels_split.exists():
                    label_files = list(labels_split.glob("*.txt"))
                    report["stats"][f"{split}_labels"] = len(label_files)
                    
                    missing_labels = []
                    for img_file in image_files:
                        label_file = labels_split / f"{img_file.stem}.txt"
                        if not label_file.exists():
                            missing_labels.append(img_file.name)
                    
                    if missing_labels:
                        report["warnings"].append(f"Missing labels for {len(missing_labels)} images in {split}")
        
        return report

    def get_image_hash(self, image: np.ndarray) -> str:
        """Calculate MD5 hash of image for duplicate detection.

        Args:
            image: Image array.

        Returns:
            MD5 hash string.
        """
        return hashlib.md5(image.tobytes()).hexdigest()

    def create_metadata(
        self,
        image: np.ndarray,
        labels: List[Tuple[int, float, float, float, float]],
        split: str,
        device_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create metadata dictionary for an image/label pair.

        Args:
            image: Image array.
            labels: YOLO format labels.
            split: Dataset split.
            device_info: Optional device information.

        Returns:
            Metadata dictionary.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "split": split,
            "image_shape": image.shape,
            "num_objects": len(labels),
            "classes": [label[0] for label in labels],
            "image_hash": self.get_image_hash(image),
            "device": device_info,
        }
