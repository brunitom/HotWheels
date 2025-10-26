"""HotWheels data collection and labeling CLI.

This module provides a command-line interface for capturing images and manually
labeling HotWheels cars with mouse-driven bounding box creation and editing.
"""

import argparse
import difflib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from hotwheels.core.camera import make_video_capture
from hotwheels.core.dataset import YOLODataset
from hotwheels.core.utils import (
    assess_image_quality,
    format_validation_report,
    get_next_filename,
    normalize_coordinates,
    safe_filename,
)
from hotwheels.core.viz import (
    draw_bounding_box,
    draw_crosshair,
    draw_instructions,
    draw_quality_indicator,
)
from hotwheels.core.yolo import get_device_string, load_model, load_names, predict


class LabelingState:
    """State management for the labeling interface."""

    def __init__(self, classes: List[str], current_class: int):
        self.classes = classes
        self.current_class = current_class
        self.box: Optional[Tuple[int, int, int, int]] = None  # Single box: (x1, y1, x2, y2)
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.current_box: Optional[Tuple[int, int, int, int]] = None
    
    def start_drawing(self, x: int, y: int) -> None:
        """Start drawing a new bounding box."""
        self.drawing = True
        self.start_point = (x, y)
        self.current_box = (x, y, x, y)

    def update_drawing(self, x: int, y: int) -> None:
        """Update current bounding box while drawing."""
        if self.drawing and self.start_point:
            x1 = min(self.start_point[0], x)
            y1 = min(self.start_point[1], y)
            x2 = max(self.start_point[0], x)
            y2 = max(self.start_point[1], y)
            self.current_box = (x1, y1, x2, y2)

    def finish_drawing(self, x: int, y: int) -> None:
        """Finish drawing and save bounding box."""
        if self.drawing and self.start_point:
            x1 = min(self.start_point[0], x)
            y1 = min(self.start_point[1], y)
            x2 = max(self.start_point[0], x)
            y2 = max(self.start_point[1], y)

            # Only save if box is large enough
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.box = (x1, y1, x2, y2)

            self.drawing = False
            self.start_point = None
            self.current_box = None

    def clear(self) -> None:
        """Clear box and reset state."""
        self.box = None
        self.drawing = False
        self.start_point = None
        self.current_box = None

    def has_box(self) -> bool:
        """Check if a box has been drawn."""
        return self.box is not None


def find_similar_classes(new_name: str, existing_classes: List[str], threshold: float = 0.7) -> List[Tuple[str, float]]:
    """Find similar class names using string similarity.

    Args:
        new_name: New class name to check.
        existing_classes: List of existing class names.
        threshold: Similarity threshold (0.0-1.0).

    Returns:
        List of (class_name, similarity_score) tuples for matches above threshold.
    """
    similar = []
    new_lower = new_name.lower()

    for existing in existing_classes:
        existing_lower = existing.lower()

        # Calculate similarity ratio
        ratio = difflib.SequenceMatcher(None, new_lower, existing_lower).ratio()

        if ratio >= threshold:
            similar.append((existing, ratio))

    # Sort by similarity (highest first)
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """Mouse callback for labeling interface."""
    state = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing new box
        state.start_drawing(x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drawing:
            state.update_drawing(x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if state.drawing:
            state.finish_drawing(x, y)


def draw_labeling_interface(
    frame: np.ndarray,
    state: LabelingState,
    instructions: List[str],
    is_frozen: bool = False,
    quality_info: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Draw the labeling interface on the frame."""
    annotated = frame.copy()

    # Draw saved box
    if state.box:
        x1, y1, x2, y2 = state.box
        label = f"{state.classes[state.current_class]}"
        annotated = draw_bounding_box(annotated, x1, y1, x2, y2, label, (0, 255, 0))

    # Draw current box being drawn
    if state.current_box:
        x1, y1, x2, y2 = state.current_box
        annotated = draw_bounding_box(annotated, x1, y1, x2, y2, "", (255, 0, 0), 1)

    # Draw crosshair
    if state.drawing and state.start_point:
        annotated = draw_crosshair(annotated, state.start_point[0], state.start_point[1])

    # Draw instructions
    annotated = draw_instructions(annotated, instructions)

    # Draw class info
    class_text = f"Class: {state.classes[state.current_class]}"
    cv2.putText(annotated, class_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw mode indicator
    mode_text = "FROZEN - Draw box" if is_frozen else "PREVIEW - Press 'c' to capture"
    cv2.putText(annotated, mode_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw quality indicator if provided (position it below mode text)
    if quality_info is not None:
        annotated = draw_quality_indicator(annotated, quality_info, position=(10, 85))

    return annotated


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the capture script."""
    parser = argparse.ArgumentParser(
        description=(
            "HotWheels data collection and labeling tool. "
            "Use mouse to draw boxes, keyboard for controls."
        )
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "avfoundation", "default"],
        help="Video backend. 'avfoundation' can be more reliable on macOS.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. 0 is usually the built-in camera on macOS.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset",
        help="Output root for labeled data. Images and labels saved under images/{split} and labels/{split}.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Path to classes.txt file. If not provided, will be created interactively.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="HotWheels Labeler",
        help="Window title.",
    )
    parser.add_argument(
        "--prelabel",
        action="store_true",
        help="Enable AI-assisted labeling using a trained model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model for prelabeling (required if --prelabel is used).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for prelabeling (0-1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device for prelabeling inference.",
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Enable real-time image quality analysis and warnings.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=str,
        default="fair",
        choices=["good", "fair", "poor"],
        help="Minimum quality level to allow saving (good/fair/poor). Only applies with --quality-check.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset integrity and print report (does not start capture interface).",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Check for duplicate images during validation (slower). Only applies with --validate.",
    )
    return parser.parse_args()


def main() -> None:
    """Main capture and labeling function."""
    args = parse_args()

    # Initialize dataset
    dataset = YOLODataset(args.out_dir)

    # Validation mode - validate and exit
    if args.validate:
        print(f"Validating dataset: {args.out_dir}")
        print("This may take a while for large datasets...")
        print()

        report = dataset.validate_dataset(check_duplicates=args.check_duplicates)
        formatted_report = format_validation_report(report)
        print(formatted_report)

        # Exit with appropriate code
        sys.exit(0 if report["valid"] else 1)

    # Normal capture mode
    dataset.create_structure()

    # Load or create classes
    # First check if classes.txt exists in the dataset
    if dataset.classes_file.exists():
        classes = dataset.load_classes()
        print(f"Loaded {len(classes)} classes from {dataset.classes_file}")
    elif args.classes and Path(args.classes).exists():
        # Load from custom classes file
        classes = dataset.load_classes()
        print(f"Loaded {len(classes)} classes from {args.classes}")
    else:
        # Create default HotWheels classes
        classes = [
            "Mustang_GT_Blue",
            "Camaro_SS_Red",
            "Civic_TypeR_White",
            "Corvette_Stingray_Yellow",
            "Porsche_911_Black",
            "BMW_M3_Silver",
            "Audi_R8_Orange",
            "Lamborghini_Huracan_Green",
            "Ferrari_488_Red",
            "McLaren_720S_Blue"
        ]
        dataset.save_classes(classes)
        print(f"Created default classes: {classes}")

    # Ask user to select class for this session
    print("\nAvailable classes:")
    for i, cls in enumerate(classes):
        print(f"  {i}: {cls}")

    while True:
        try:
            class_input = input(f"\nWhich car class? (0-{len(classes)-1} or type new name): ").strip()

            # Try to parse as integer first
            try:
                selected_class = int(class_input)
                if 0 <= selected_class < len(classes):
                    break
                else:
                    print(f"Please enter a number between 0 and {len(classes)-1}")
            except ValueError:
                # Not a number, treat as new class name
                if not class_input:
                    print("Please enter a class number or name")
                    continue

                # Validate class name (alphanumeric and underscores only)
                if not all(c.isalnum() or c == '_' for c in class_input):
                    print("Class name must contain only letters, numbers, and underscores")
                    continue

                # Check for exact duplicate (case-insensitive)
                if class_input.lower() in [cls.lower() for cls in classes]:
                    print(f"❌ Class '{class_input}' already exists (case-insensitive match)")
                    continue

                # Check for similar existing classes
                similar = find_similar_classes(class_input, classes, threshold=0.7)
                if similar:
                    print(f"\n⚠️  Similar classes found:")
                    for cls, score in similar:
                        print(f"    {cls} (similarity: {score:.0%})")

                    confirm = input(f"\nAdd '{class_input}' anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        print("Cancelled. Please choose an existing class or enter a different name.")
                        continue

                # Add new class
                classes.append(class_input)
                selected_class = len(classes) - 1
                dataset.save_classes(classes)
                print(f"✅ Added new class: {class_input} (class_id: {selected_class})")
                break

        except KeyboardInterrupt:
            print("\nCapture cancelled")
            sys.exit(0)

    print(f"Selected: {classes[selected_class]}")

    # Ask user to select split for this session
    print("\nDataset splits:")
    print("  0: train (training data)")
    print("  1: val (validation data)")
    print("  2: test (test data)")

    split_map = {0: "train", 1: "val", 2: "test"}
    while True:
        try:
            split_input = input("\nWhich split for this session? (0=train, 1=val, 2=test): ").strip()
            split_choice = int(split_input)
            if split_choice in split_map:
                selected_split = split_map[split_choice]
                break
            else:
                print("Please enter 0, 1, or 2")
        except (ValueError, KeyboardInterrupt):
            print("\nCapture cancelled")
            sys.exit(0)

    print(f"Selected split: {selected_split}")
    print(f"\nCapturing {classes[selected_class]} for {selected_split} split")
    
    # Load prelabeling model if requested
    prelabel_model = None
    prelabel_names = {}
    if args.prelabel:
        if not args.model:
            print("Error: --model is required when using --prelabel", file=sys.stderr)
            sys.exit(1)

        try:
            prelabel_model = load_model(args.model)
            prelabel_names = load_names(None, prelabel_model)
            print(f"Loaded prelabeling model: {args.model}")
        except Exception as e:
            print(f"Failed to load prelabeling model: {e}", file=sys.stderr)
            sys.exit(1)

    # Open camera
    try:
        cap = make_video_capture(args.camera, args.backend)
        print(f"Opened camera {args.camera} with backend '{args.backend}'")
    except RuntimeError as e:
        print(f"Camera error: {e}", file=sys.stderr)
        sys.exit(2)

    # Create window and set mouse callback
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    # Initialize labeling state with selected class
    state = LabelingState(classes, selected_class)

    # Instructions
    instructions = [
        "PREVIEW: Press 'c' to capture frame",
        "FROZEN: Draw box, 'r'=redo, ENTER=save, 'n'=skip",
        "q=quit"
    ]

    # Set mouse callback
    cv2.setMouseCallback(args.window, mouse_callback, state)

    frame_count = 0
    frozen_frame = None
    frozen_frame_quality = None

    try:
        print("\nStarting capture interface...")
        print(f"Press 'c' to capture a frame, then draw a box around the {classes[selected_class]}")
        if args.quality_check:
            print(f"Quality checking enabled (minimum: {args.quality_threshold})")

        while True:
            # Read frame
            if frozen_frame is None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Camera frame read failed; attempting to continue...", file=sys.stderr)
                    continue
            else:
                frame = frozen_frame.copy()

            # Assess quality if enabled
            quality_info = None
            if args.quality_check and frozen_frame is not None:
                quality_info = frozen_frame_quality
            elif args.quality_check and frozen_frame is None:
                # Real-time quality assessment for live view
                quality_info = assess_image_quality(frame)

            # Run prelabeling if enabled and frame is frozen
            if prelabel_model and frozen_frame is not None and not state.has_box():
                try:
                    device_arg = get_device_string(args.device)
                    results = predict(
                        model=prelabel_model,
                        source=frozen_frame,
                        conf=args.conf,
                        device=device_arg,
                        verbose=False,
                    )

                    # Convert prelabeling results to box (take first matching detection)
                    if results and len(results) > 0:
                        result = results[0]
                        boxes = result.boxes if hasattr(result, "boxes") else None
                        if boxes and len(boxes) > 0:
                            for box in boxes:
                                xyxy = box.xyxy[0].tolist()
                                conf = float(box.conf[0]) if box.conf is not None else 0.0

                                if conf >= args.conf:
                                    x1, y1, x2, y2 = map(int, xyxy)
                                    state.box = (x1, y1, x2, y2)
                                    print(f"Prelabeling suggestion added (conf: {conf:.2f}). Adjust if needed or press ENTER to save.")
                                    break  # Only take first detection
                except Exception as e:
                    print(f"Prelabeling error: {e}", file=sys.stderr)

            # Draw interface
            annotated = draw_labeling_interface(frame, state, instructions, frozen_frame is not None, quality_info)

            # Display frame
            cv2.imshow(args.window, annotated)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):  # c - capture/freeze frame
                if frozen_frame is None:
                    frozen_frame = frame.copy()
                    state.clear()  # Clear any existing box

                    # Assess quality when capturing
                    if args.quality_check:
                        frozen_frame_quality = assess_image_quality(frozen_frame)
                        print(f"Frame captured. Quality: {frozen_frame_quality['overall_quality']}")
                        if frozen_frame_quality['warnings']:
                            for warning in frozen_frame_quality['warnings']:
                                print(f"  Warning: {warning}")
                    else:
                        frozen_frame_quality = None
                        print("Frame captured. Draw bounding box and press ENTER to save.")
            elif key == ord("\r") or key == ord("\n"):  # ENTER - save
                if frozen_frame is not None and state.has_box():
                    # Check quality threshold if enabled
                    quality_threshold_map = {"good": 3, "fair": 2, "poor": 1}
                    quality_level_map = {"good": 3, "fair": 2, "poor": 1}

                    if args.quality_check and frozen_frame_quality:
                        current_quality = quality_level_map.get(frozen_frame_quality["overall_quality"], 1)
                        threshold = quality_threshold_map.get(args.quality_threshold, 2)

                        if current_quality < threshold:
                            print(f"Image quality ({frozen_frame_quality['overall_quality']}) is below threshold ({args.quality_threshold}). Not saved.")
                            print(f"  Recommendation: {frozen_frame_quality['recommendation']}")
                            continue

                    # Generate filename
                    filename = f"img_{frame_count:04d}"
                    frame_count += 1

                    # Save image
                    image_path = dataset.save_image(frozen_frame, filename, selected_split)

                    # Convert box to YOLO format
                    img_height, img_width = frozen_frame.shape[:2]
                    x1, y1, x2, y2 = state.box
                    x_center, y_center, width, height = normalize_coordinates(
                        x1, y1, x2, y2, img_width, img_height
                    )
                    yolo_labels = [(selected_class, x_center, y_center, width, height)]

                    # Save labels
                    label_path = dataset.save_labels(yolo_labels, filename, selected_split)

                    # Save metadata (include quality info if available)
                    metadata = dataset.create_metadata(
                        frozen_frame, yolo_labels, selected_split
                    )
                    if frozen_frame_quality:
                        metadata["quality"] = frozen_frame_quality
                    dataset.save_metadata(filename, selected_split, metadata)

                    print(f"Saved: {image_path} + {label_path} ({classes[selected_class]})")
                    frozen_frame = None
                    frozen_frame_quality = None
                    state.clear()
                elif frozen_frame is not None and not state.has_box():
                    print("No box drawn. Draw a box or press 'n' to skip.")
                else:
                    print("No frame captured. Press 'c' to capture a frame.")
            elif key == ord("n"):  # n - skip current frame
                if frozen_frame is not None:
                    frozen_frame = None
                    frozen_frame_quality = None
                    state.clear()
                    print("Skipped frame")
            elif key == ord("r"):  # r - redo/clear box
                if frozen_frame is not None:
                    state.clear()
                    print("Box cleared. Draw again.")

    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()

        # Save data.yaml
        dataset.save_data_yaml(classes)
        print(f"Saved data.yaml with {len(classes)} classes")
        print("Capture stopped")


if __name__ == "__main__":
    main()
