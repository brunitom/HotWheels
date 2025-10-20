"""Unit tests for camera utilities."""

import pytest

from hotwheels.core.camera import estimate_fps, get_available_cameras, validate_camera_access


class TestCameraUtils:
    """Test cases for camera utility functions."""

    def test_estimate_fps_first_frame(self):
        """Test FPS estimation on first frame."""
        curr_ticks = 1000
        fps, new_ticks = estimate_fps(None, curr_ticks)
        
        assert fps is None
        assert new_ticks == curr_ticks

    def test_estimate_fps_valid(self):
        """Test FPS estimation with valid frame difference."""
        prev_ticks = 1000
        curr_ticks = 2000  # 1000 ticks difference
        
        # Mock cv2.getTickFrequency to return 1000 (1 tick = 1ms)
        import cv2
        original_freq = cv2.getTickFrequency
        cv2.getTickFrequency = lambda: 1000.0
        
        try:
            fps, new_ticks = estimate_fps(prev_ticks, curr_ticks)
            assert fps == 1.0  # 1000ms = 1 second = 1 FPS
            assert new_ticks == curr_ticks
        finally:
            cv2.getTickFrequency = original_freq

    def test_estimate_fps_invalid_timing(self):
        """Test FPS estimation with invalid timing."""
        prev_ticks = 2000
        curr_ticks = 1000  # curr < prev (invalid)
        
        fps, new_ticks = estimate_fps(prev_ticks, curr_ticks)
        
        assert fps is None
        assert new_ticks == curr_ticks

    def test_estimate_fps_zero_delta(self):
        """Test FPS estimation with zero time delta."""
        prev_ticks = 1000
        curr_ticks = 1000  # Same time
        
        fps, new_ticks = estimate_fps(prev_ticks, curr_ticks)
        
        assert fps is None
        assert new_ticks == curr_ticks

    @pytest.mark.skip(reason="Requires camera hardware")
    def test_get_available_cameras(self):
        """Test finding available cameras."""
        cameras = get_available_cameras(max_cameras=3)
        
        # Should return a list of integers
        assert isinstance(cameras, list)
        assert all(isinstance(cam, int) for cam in cameras)
        assert all(cam >= 0 for cam in cameras)

    @pytest.mark.skip(reason="Requires camera hardware")
    def test_validate_camera_access(self):
        """Test camera access validation."""
        # Test with camera 0 (most common)
        has_camera_0 = validate_camera_access(0)
        assert isinstance(has_camera_0, bool)
        
        # Test with non-existent camera
        has_camera_999 = validate_camera_access(999)
        assert has_camera_999 is False

    def test_validate_camera_access_invalid_backend(self):
        """Test camera access validation with invalid backend."""
        # This should not raise an exception, just return False
        result = validate_camera_access(0, "invalid_backend")
        assert result is False
