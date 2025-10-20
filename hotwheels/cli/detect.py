"""Real-time HotWheels detection CLI.

This module provides a command-line interface for real-time HotWheels car detection
using YOLOv8 with camera capture and visualization.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from hotwheels.core.camera import estimate_fps, make_video_capture
from hotwheels.core.yolo import get_device_string, load_model, load_names, predict
from hotwheels.core.viz import draw_detections


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the detector script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "HotWheels real-time detector using YOLOv8. Press 'q' to quit.\n"
            "Tip: macOS camera permissions: System Settings > Privacy & Security > Camera > allow Terminal/IDE."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path("runs/detect/train/weights/best.pt")),
        help=(
            "Path to trained YOLOv8 model .pt file. Default tries 'runs/detect/train/weights/best.pt' "
            "(created by ultralytics training)."
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help=(
            "Camera index. 0 is usually the built-in camera on macOS. If it fails, try 1 or 2."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "avfoundation", "default"],
        help=(
            "Video backend. 'avfoundation' can be more reliable on macOS."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (pixels).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (0-1).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (0-1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help=(
            "Inference device: auto|cpu|mps|cuda. On Apple Silicon, 'mps' may speed up inference."
        ),
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay frames-per-second text on the video.",
    )
    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help=(
            "Optional path to a text file with class names, one per line, overriding model names."
        ),
    )
    parser.add_argument(
        "--window",
        type=str,
        default="HotWheels Detector",
        help="Window title.",
    )
    return parser.parse_args()


def main() -> None:
    """Main detection function."""
    args = parse_args()

    # Load YOLO model
    try:
        model = load_model(args.model)
        print(f"Loaded model: {args.model}")
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load class names
    try:
        names = load_names(args.names, model)
        print(f"Loaded {len(names)} classes: {list(names.values())}")
    except Exception as e:
        print(f"Failed to load names: {e}", file=sys.stderr)
        sys.exit(1)

    # Open camera
    try:
        cap = make_video_capture(args.camera, args.backend)
        print(f"Opened camera {args.camera} with backend '{args.backend}'")
    except RuntimeError as e:
        print(f"Camera error: {e}", file=sys.stderr)
        sys.exit(2)

    # Create window
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    # FPS measurement
    prev_ticks: Optional[int] = None
    last_fps: Optional[float] = None

    # Device configuration
    device_arg = get_device_string(args.device)
    if device_arg:
        print(f"Using device: {device_arg}")

    try:
        print("Starting detection loop. Press 'q' to quit...")
        
        while True:
            # Read frame
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera frame read failed; attempting to continue...", file=sys.stderr)
                continue

            # Run inference
            try:
                results = predict(
                    model=model,
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=device_arg,
                    verbose=False,
                )
            except Exception as infer_err:
                print(f"Inference error: {infer_err}", file=sys.stderr)
                results = []

            # Update FPS
            curr_ticks = cv2.getTickCount()
            last_fps, prev_ticks = estimate_fps(prev_ticks, curr_ticks)

            # Draw detections
            annotated = draw_detections(
                frame=frame,
                detections=results,
                names=names,
                show_fps=args.show_fps,
                fps=last_fps,
                conf_threshold=args.conf,
            )

            # Display frame
            cv2.imshow(args.window, annotated)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f") and args.show_fps:
                # Toggle FPS display
                args.show_fps = not args.show_fps
                print(f"FPS display: {'ON' if args.show_fps else 'OFF'}")

    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")


if __name__ == "__main__":
    main()
