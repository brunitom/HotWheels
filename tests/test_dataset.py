"""Unit tests for dataset utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hotwheels.core.dataset import YOLODataset
from hotwheels.core.utils import normalize_coordinates, denormalize_coordinates


class TestYOLODataset:
    """Test cases for YOLODataset class."""

    def test_create_structure(self):
        """Test dataset directory structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)
            dataset.create_structure(["train", "val"])
            
            assert dataset.images_dir.exists()
            assert dataset.labels_dir.exists()
            assert (dataset.images_dir / "train").exists()
            assert (dataset.images_dir / "val").exists()
            assert (dataset.labels_dir / "train").exists()
            assert (dataset.labels_dir / "val").exists()

    def test_save_and_load_classes(self):
        """Test saving and loading class names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)
            classes = ["car1", "car2", "car3"]
            
            # Save classes
            dataset.save_classes(classes)
            assert dataset.classes_file.exists()
            
            # Load classes
            loaded_classes = dataset.load_classes()
            assert loaded_classes == classes

    def test_save_and_load_labels(self):
        """Test saving and loading YOLO format labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)
            dataset.create_structure()
            
            # Test labels
            labels = [
                (0, 0.5, 0.5, 0.2, 0.1),  # class 0, center (0.5, 0.5), size (0.2, 0.1)
                (1, 0.3, 0.7, 0.15, 0.08),  # class 1, center (0.3, 0.7), size (0.15, 0.08)
            ]
            
            # Save labels
            label_path = dataset.save_labels(labels, "test", "train")
            assert label_path.exists()
            
            # Load labels
            loaded_labels = dataset.load_labels("test", "train")
            assert len(loaded_labels) == 2
            assert loaded_labels[0] == labels[0]
            assert loaded_labels[1] == labels[1]

    def test_save_labels_validation(self):
        """Test label validation for normalized coordinates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)
            dataset.create_structure()
            
            # Test invalid labels (not normalized)
            invalid_labels = [(0, 1.5, 0.5, 0.2, 0.1)]  # x_center > 1
            
            with pytest.raises(ValueError, match="Labels must be normalized"):
                dataset.save_labels(invalid_labels, "test", "train")

    def test_save_data_yaml(self):
        """Test data.yaml creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)
            classes = ["car1", "car2"]
            
            yaml_path = dataset.save_data_yaml(classes)
            assert yaml_path.exists()
            
            # Read and verify content
            import yaml
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            
            assert data["names"][0] == "car1"
            assert data["names"][1] == "car2"
            assert "train" in data
            assert "val" in data

    def test_validate_dataset(self):
        """Test dataset validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)
            
            # Test empty dataset
            report = dataset.validate_dataset()
            assert not report["valid"]
            assert "Images directory missing" in report["errors"]
            assert "Labels directory missing" in report["errors"]
            
            # Create valid structure
            dataset.create_structure()
            dataset.save_classes(["car1", "car2"])
            
            report = dataset.validate_dataset()
            assert report["valid"]
            assert len(report["errors"]) == 0

    def test_get_image_hash(self):
        """Test image hash calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)
            
            # Create test image
            img1 = np.zeros((100, 100, 3), dtype=np.uint8)
            img2 = np.ones((100, 100, 3), dtype=np.uint8)
            
            hash1 = dataset.get_image_hash(img1)
            hash2 = dataset.get_image_hash(img2)
            
            assert hash1 != hash2
            assert len(hash1) == 32  # MD5 hash length
            assert len(hash2) == 32

    def test_create_metadata(self):
        """Test metadata creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)
            
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            labels = [(0, 0.5, 0.5, 0.2, 0.1)]
            
            metadata = dataset.create_metadata(img, labels, "train", "test_device")
            
            assert metadata["split"] == "train"
            assert metadata["image_shape"] == (100, 100, 3)
            assert metadata["num_objects"] == 1
            assert metadata["classes"] == [0]
            assert metadata["device"] == "test_device"
            assert "timestamp" in metadata
            assert "image_hash" in metadata


class TestUtils:
    """Test cases for utility functions."""

    def test_normalize_coordinates(self):
        """Test coordinate normalization."""
        # Test normal case
        x1, y1, x2, y2 = 100, 50, 200, 150
        img_width, img_height = 400, 300
        
        x_center, y_center, width, height = normalize_coordinates(
            x1, y1, x2, y2, img_width, img_height
        )
        
        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert 0 <= width <= 1
        assert 0 <= height <= 1
        
        # Expected values
        expected_x_center = 150 / 400  # 0.375
        expected_y_center = 100 / 300  # 0.333...
        expected_width = 100 / 400     # 0.25
        expected_height = 100 / 300    # 0.333...
        
        assert abs(x_center - expected_x_center) < 1e-6
        assert abs(y_center - expected_y_center) < 1e-6
        assert abs(width - expected_width) < 1e-6
        assert abs(height - expected_height) < 1e-6

    def test_denormalize_coordinates(self):
        """Test coordinate denormalization."""
        # Test normal case
        x_center, y_center, width, height = 0.375, 0.333333, 0.25, 0.333333
        img_width, img_height = 400, 300
        
        x1, y1, x2, y2 = denormalize_coordinates(
            x_center, y_center, width, height, img_width, img_height
        )
        
        # Expected values
        expected_x1 = 100  # 150 - 50
        expected_y1 = 50   # 100 - 50
        expected_x2 = 200  # 150 + 50
        expected_y2 = 150  # 100 + 50
        
        assert x1 == expected_x1
        assert y1 == expected_y1
        assert x2 == expected_x2
        assert y2 == expected_y2

    def test_coordinate_roundtrip(self):
        """Test coordinate normalization and denormalization roundtrip."""
        original_coords = (100, 50, 200, 150)
        img_width, img_height = 400, 300
        
        # Normalize
        x_center, y_center, width, height = normalize_coordinates(
            *original_coords, img_width, img_height
        )
        
        # Denormalize
        x1, y1, x2, y2 = denormalize_coordinates(
            x_center, y_center, width, height, img_width, img_height
        )
        
        # Should be close to original (within 1 pixel due to rounding)
        assert abs(x1 - original_coords[0]) <= 1
        assert abs(y1 - original_coords[1]) <= 1
        assert abs(x2 - original_coords[2]) <= 1
        assert abs(y2 - original_coords[3]) <= 1

    def test_invalid_coordinates(self):
        """Test handling of invalid coordinates."""
        # Test invalid bounding box (x2 <= x1)
        with pytest.raises(ValueError, match="Invalid bounding box"):
            normalize_coordinates(200, 50, 100, 150, 400, 300)
        
        # Test invalid normalized coordinates
        with pytest.raises(ValueError, match="Coordinates must be normalized"):
            denormalize_coordinates(1.5, 0.5, 0.2, 0.1, 400, 300)
