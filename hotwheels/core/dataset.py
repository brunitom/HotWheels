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

    def validate_dataset(self, check_duplicates: bool = False) -> Dict[str, Any]:
        """Validate dataset integrity and return comprehensive validation report.

        Args:
            check_duplicates: Whether to check for duplicate images (slower).

        Returns:
            Dictionary containing validation results and statistics.
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "per_class": {},
                "per_split": {},
            },
        }

        # Check directory structure
        if not self.images_dir.exists():
            report["errors"].append("Images directory missing")
            report["valid"] = False
            return report

        if not self.labels_dir.exists():
            report["errors"].append("Labels directory missing")
            report["valid"] = False
            return report

        # Load classes
        classes = []
        if self.classes_file.exists():
            try:
                classes = self.load_classes()
                report["stats"]["num_classes"] = len(classes)
            except Exception as e:
                report["warnings"].append(f"Failed to load classes: {e}")
        else:
            report["warnings"].append("Classes file missing")

        # Initialize per-class counters
        class_counts = {i: {"train": 0, "val": 0} for i in range(len(classes))}

        # Track hashes for duplicate detection
        image_hashes = {} if check_duplicates else None

        # Count files per split
        splits = ["train", "val"]
        total_images = 0
        total_labels = 0
        total_objects = 0

        for split in splits:
            images_split = self.images_dir / split
            labels_split = self.labels_dir / split

            split_stats = {
                "images": 0,
                "labels": 0,
                "objects": 0,
                "missing_labels": [],
                "malformed_labels": [],
                "empty_labels": [],
            }

            if not images_split.exists():
                report["warnings"].append(f"Split directory missing: {split}/images")
                continue

            if not labels_split.exists():
                report["warnings"].append(f"Split directory missing: {split}/labels")
                continue

            image_files = list(images_split.glob("*.jpg")) + list(images_split.glob("*.png"))
            split_stats["images"] = len(image_files)
            total_images += len(image_files)

            # Check each image and its label
            for img_file in image_files:
                label_file = labels_split / f"{img_file.stem}.txt"

                if not label_file.exists():
                    split_stats["missing_labels"].append(img_file.name)
                    continue

                split_stats["labels"] += 1
                total_labels += 1

                # Validate label file
                try:
                    with open(label_file, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]

                    if len(lines) == 0:
                        split_stats["empty_labels"].append(img_file.name)
                        continue

                    # Parse and validate each label
                    for line_num, line in enumerate(lines, 1):
                        parts = line.split()
                        if len(parts) != 5:
                            split_stats["malformed_labels"].append(
                                f"{img_file.name}:{line_num} (expected 5 values, got {len(parts)})"
                            )
                            continue

                        try:
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]

                            # Validate class ID
                            if class_id < 0 or (classes and class_id >= len(classes)):
                                split_stats["malformed_labels"].append(
                                    f"{img_file.name}:{line_num} (invalid class_id: {class_id})"
                                )
                                continue

                            # Validate coordinates
                            if not all(0 <= coord <= 1 for coord in coords):
                                split_stats["malformed_labels"].append(
                                    f"{img_file.name}:{line_num} (coordinates out of bounds)"
                                )
                                continue

                            # Count objects per class
                            if class_id in class_counts:
                                class_counts[class_id][split] += 1
                            split_stats["objects"] += 1
                            total_objects += 1

                        except (ValueError, IndexError) as e:
                            split_stats["malformed_labels"].append(
                                f"{img_file.name}:{line_num} (parse error: {e})"
                            )

                except Exception as e:
                    split_stats["malformed_labels"].append(f"{img_file.name} (read error: {e})")

                # Check for duplicates if requested
                if check_duplicates and image_hashes is not None:
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img_hash = self.get_image_hash(img)
                            if img_hash in image_hashes:
                                report["warnings"].append(
                                    f"Duplicate image: {img_file.name} matches {image_hashes[img_hash]}"
                                )
                            else:
                                image_hashes[img_hash] = img_file.name
                    except Exception:
                        pass  # Skip duplicate check on error

            # Add split-level warnings
            if split_stats["missing_labels"]:
                report["warnings"].append(
                    f"{split}: {len(split_stats['missing_labels'])} images without labels"
                )
            if split_stats["malformed_labels"]:
                report["errors"].extend(
                    [f"{split}: {msg}" for msg in split_stats["malformed_labels"][:10]]
                )
                if len(split_stats["malformed_labels"]) > 10:
                    report["errors"].append(
                        f"{split}: ... and {len(split_stats['malformed_labels']) - 10} more malformed labels"
                    )
                report["valid"] = False
            if split_stats["empty_labels"]:
                report["warnings"].append(
                    f"{split}: {len(split_stats['empty_labels'])} empty label files"
                )

            report["stats"]["per_split"][split] = split_stats

        # Overall statistics
        report["stats"]["total_images"] = total_images
        report["stats"]["total_labels"] = total_labels
        report["stats"]["total_objects"] = total_objects

        # Per-class statistics
        for class_id, counts in class_counts.items():
            class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
            total_class = counts["train"] + counts["val"]
            if total_class > 0:
                report["stats"]["per_class"][class_name] = {
                    "train": counts["train"],
                    "val": counts["val"],
                    "total": total_class,
                }

        # Check for class imbalance
        if classes:
            class_totals = [counts["train"] + counts["val"] for counts in class_counts.values()]
            if class_totals:
                max_count = max(class_totals)
                min_count = min(class_totals)
                if min_count == 0:
                    missing_classes = [
                        classes[i]
                        for i, total in enumerate(class_totals)
                        if total == 0 and i < len(classes)
                    ]
                    report["warnings"].append(f"Classes with no samples: {missing_classes}")
                elif max_count > 10 * min_count:
                    report["warnings"].append(
                        f"Significant class imbalance detected (max: {max_count}, min: {min_count})"
                    )

        # Check split ratio
        if total_images > 0:
            train_ratio = report["stats"]["per_split"].get("train", {}).get("images", 0) / total_images
            val_ratio = report["stats"]["per_split"].get("val", {}).get("images", 0) / total_images
            report["stats"]["split_ratio"] = {"train": train_ratio, "val": val_ratio}

            if val_ratio < 0.1:
                report["warnings"].append("Validation set is very small (< 10% of data)")
            elif val_ratio > 0.5:
                report["warnings"].append("Validation set is very large (> 50% of data)")

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
