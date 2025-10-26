"""Unit tests for capture CLI utilities."""

import tempfile
from pathlib import Path

import pytest

from hotwheels.cli.capture import find_similar_classes
from hotwheels.core.dataset import YOLODataset


class TestSimilarityDetection:
    """Test cases for class name similarity detection."""

    def test_exact_match(self):
        """Test exact match detection (case-insensitive)."""
        existing = ["Mustang_GT_Blue", "Camaro_SS_Red"]
        similar = find_similar_classes("mustang_gt_blue", existing, threshold=0.7)

        assert len(similar) == 1
        assert similar[0][0] == "Mustang_GT_Blue"
        assert similar[0][1] > 0.9  # Very high similarity

    def test_typo_detection(self):
        """Test detection of similar names with typos."""
        existing = ["Subaru_Impreza_WRX", "Ferrari_488_Red"]
        similar = find_similar_classes("Subaru_Imprezza_WRX", existing, threshold=0.7)

        assert len(similar) >= 1
        assert "Subaru_Impreza_WRX" in [s[0] for s in similar]

    def test_no_similar_classes(self):
        """Test when no similar classes exist."""
        existing = ["Mustang_GT_Blue", "Camaro_SS_Red"]
        similar = find_similar_classes("Tesla_Cybertruck_Silver", existing, threshold=0.7)

        assert len(similar) == 0

    def test_multiple_similar_classes(self):
        """Test detection of multiple similar classes."""
        existing = ["Mustang_GT_Blue", "Mustang_GT_Red", "Mustang_Shelby_Blue"]
        similar = find_similar_classes("Mustang_GT_Black", existing, threshold=0.7)

        assert len(similar) >= 2
        # Should be sorted by similarity (highest first)
        assert similar[0][1] >= similar[1][1]

    def test_threshold_filtering(self):
        """Test that threshold properly filters results."""
        existing = ["Mustang_GT_Blue", "Tesla_Model_S"]

        # Low threshold - might find some matches
        similar_low = find_similar_classes("Mustang_Shelby", existing, threshold=0.3)

        # High threshold - fewer matches
        similar_high = find_similar_classes("Mustang_Shelby", existing, threshold=0.9)

        assert len(similar_low) >= len(similar_high)

    def test_case_insensitive(self):
        """Test case-insensitive comparison."""
        existing = ["MUSTANG_GT_BLUE"]
        similar = find_similar_classes("mustang_gt_blue", existing, threshold=0.7)

        assert len(similar) == 1
        assert similar[0][1] == 1.0  # Perfect match despite case difference

    def test_empty_existing_classes(self):
        """Test with empty existing classes list."""
        similar = find_similar_classes("NewClass", [], threshold=0.7)
        assert len(similar) == 0


class TestClassPersistence:
    """Test cases for class persistence across sessions."""

    def test_classes_persist_after_save(self):
        """Test that classes are saved and can be loaded in new session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)

            # Session 1: Create and save classes
            classes_session1 = ["Mustang_GT_Blue", "Camaro_SS_Red"]
            dataset.save_classes(classes_session1)

            # Session 2: Create new dataset instance and load
            dataset2 = YOLODataset(temp_dir)
            loaded_classes = dataset2.load_classes()

            assert loaded_classes == classes_session1

    def test_classes_preserve_order(self):
        """Test that class order is preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)

            classes = ["Class_A", "Class_B", "Class_C", "Class_D"]
            dataset.save_classes(classes)

            loaded = dataset.load_classes()
            assert loaded == classes

    def test_new_class_appended(self):
        """Test that new classes are appended to existing list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)

            # Initial classes
            initial = ["Class_A", "Class_B"]
            dataset.save_classes(initial)

            # Add new class
            updated = initial + ["Class_C"]
            dataset.save_classes(updated)

            loaded = dataset.load_classes()
            assert loaded == updated
            assert loaded[0] == "Class_A"  # Original order preserved
            assert loaded[-1] == "Class_C"  # New class at end

    def test_classes_file_not_overwritten(self):
        """Test that existing classes file is not overwritten on startup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = YOLODataset(temp_dir)

            # Create classes file
            original_classes = ["Mustang_GT_Blue", "Subaru_Impreza_WRX"]
            dataset.save_classes(original_classes)

            # Simulate new session - check file exists
            assert dataset.classes_file.exists()

            # Load and verify - should NOT create default classes
            loaded = dataset.load_classes()
            assert loaded == original_classes
            assert "Subaru_Impreza_WRX" in loaded


class TestClassValidation:
    """Test cases for class name validation."""

    def test_valid_class_names(self):
        """Test that valid class names are accepted."""
        valid_names = [
            "Mustang_GT_Blue",
            "Ferrari_488_Red",
            "BMW_M3_Silver",
            "Class123",
            "Car_1",
        ]

        for name in valid_names:
            # Only alphanumeric and underscores
            assert all(c.isalnum() or c == '_' for c in name)

    def test_invalid_class_names(self):
        """Test detection of invalid class names."""
        invalid_names = [
            "Mustang GT Blue",  # Spaces
            "Ferrari-488",      # Hyphens
            "BMW/M3",           # Slashes
            "Car #1",           # Special chars
            "",                 # Empty
        ]

        for name in invalid_names:
            # Should NOT pass validation
            if name:  # Skip empty string
                assert not all(c.isalnum() or c == '_' for c in name)
