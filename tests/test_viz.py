"""Unit tests for visualization utilities."""

import numpy as np
import pytest

from hotwheels.core.viz import (
    draw_bounding_box,
    draw_crosshair,
    draw_detections,
    draw_instructions,
    create_overlay,
)


class TestVisualizationUtils:
    """Test cases for visualization utility functions."""

    def test_draw_bounding_box(self):
        """Test drawing a single bounding box."""
        # Create test image
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Draw box
        annotated = draw_bounding_box(
            frame, 10, 10, 50, 50, "test_label", (0, 255, 0), 2
        )
        
        # Check that image was modified
        assert not np.array_equal(frame, annotated)
        assert annotated.shape == frame.shape

    def test_draw_bounding_box_no_label(self):
        """Test drawing a bounding box without label."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        annotated = draw_bounding_box(frame, 10, 10, 50, 50)
        
        assert not np.array_equal(frame, annotated)
        assert annotated.shape == frame.shape

    def test_draw_crosshair(self):
        """Test drawing a crosshair."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        annotated = draw_crosshair(frame, 50, 50, 20)
        
        assert not np.array_equal(frame, annotated)
        assert annotated.shape == frame.shape

    def test_draw_instructions(self):
        """Test drawing instruction text."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        instructions = ["Press 'q' to quit", "Press 's' to save"]
        
        annotated = draw_instructions(frame, instructions)
        
        assert not np.array_equal(frame, annotated)
        assert annotated.shape == frame.shape

    def test_draw_instructions_empty(self):
        """Test drawing empty instructions."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        annotated = draw_instructions(frame, [])
        
        # Should return original frame if no instructions
        assert np.array_equal(frame, annotated)

    def test_create_overlay(self):
        """Test creating semi-transparent overlay."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
        
        overlay = create_overlay(frame, alpha=0.5, color=(0, 0, 0))
        
        assert not np.array_equal(frame, overlay)
        assert overlay.shape == frame.shape
        
        # Overlay should be darker than original
        assert np.mean(overlay) < np.mean(frame)

    def test_draw_detections_empty(self):
        """Test drawing detections with empty results."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        names = {0: "car", 1: "truck"}
        
        annotated = draw_detections(frame, [], names, show_fps=False)
        
        # Should return original frame if no detections
        assert np.array_equal(frame, annotated)

    def test_draw_detections_none(self):
        """Test drawing detections with None results."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        names = {0: "car", 1: "truck"}
        
        annotated = draw_detections(frame, None, names, show_fps=False)
        
        # Should return original frame if no detections
        assert np.array_equal(frame, annotated)

    def test_draw_detections_with_fps(self):
        """Test drawing detections with FPS display."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        names = {0: "car", 1: "truck"}
        
        annotated = draw_detections(frame, [], names, show_fps=True, fps=30.0)
        
        # Should be different from original due to FPS text
        assert not np.array_equal(frame, annotated)

    def test_draw_detections_no_boxes(self):
        """Test drawing detections when result has no boxes."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        names = {0: "car", 1: "truck"}
        
        # Mock a result object with empty boxes
        class MockResult:
            def __init__(self):
                self.boxes = None
        
        class MockDetection:
            def __init__(self):
                self.boxes = None
        
        detections = [MockDetection()]
        
        annotated = draw_detections(frame, detections, names, show_fps=False)
        
        # Should show "No cars detected" message
        assert not np.array_equal(frame, annotated)

    def test_draw_detections_confidence_filtering(self):
        """Test that detections below confidence threshold are filtered."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        names = {0: "car"}
        
        # Mock a result object with low confidence box
        class MockBox:
            def __init__(self, conf, cls):
                self.conf = [conf]
                self.cls = [cls]
                self.xyxy = [np.array([10, 10, 50, 50])]
        
        class MockResult:
            def __init__(self):
                self.boxes = [MockBox(0.3, 0)]  # Low confidence
        
        class MockDetection:
            def __init__(self):
                self.boxes = [MockBox(0.3, 0)]
        
        detections = [MockDetection()]

        # High confidence threshold should filter out the box
        annotated = draw_detections(
            frame, detections, names, show_fps=False, conf_threshold=0.5
        )

        # Box is filtered out, so frame should remain unchanged (no boxes drawn)
        # The function only shows "No cars detected" when boxes list is empty, not when filtered
        assert np.array_equal(frame, annotated)

    def test_draw_detections_custom_colors(self):
        """Test drawing detections with custom colors."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        names = {0: "car"}
        
        # Mock a result object
        class MockBox:
            def __init__(self):
                self.conf = [0.8]
                self.cls = [0]
                self.xyxy = [np.array([10, 10, 50, 50])]
        
        class MockResult:
            def __init__(self):
                self.boxes = [MockBox()]
        
        class MockDetection:
            def __init__(self):
                self.boxes = [MockBox()]
        
        detections = [MockDetection()]
        
        annotated = draw_detections(
            frame,
            detections,
            names,
            show_fps=False,
            box_color=(255, 0, 0),  # Red
            text_color=(255, 255, 255),  # White
        )
        
        assert not np.array_equal(frame, annotated)
        assert annotated.shape == frame.shape
