"""HotWheels data collection and labeling CLI.

This module provides a command-line interface for capturing images and manually
labeling HotWheels cars with mouse-driven bounding box creation and editing.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from hotwheels.core.camera import make_video_capture
from hotwheels.core.dataset import YOLODataset
from hotwheels.core.utils import get_next_filename, normalize_coordinates, safe_filename
from hotwheels.core.viz import draw_bounding_box, draw_instructions, draw_crosshair
from hotwheels.core.yolo import get_device_string, load_model, load_names, predict


class LabelingState:
    """State management for the labeling interface."""
    
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.current_class = 0
        self.boxes: List[Tuple[int, int, int, int, int]] = []  # (x1, y1, x2, y2, class_id)
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.current_box: Optional[Tuple[int, int, int, int]] = None
        self.selected_box: Optional[int] = None
        self.undo_stack: List[List[Tuple[int, int, int, int, int]]] = []
        self.redo_stack: List[List[Tuple[int, int, int, int, int]]] = []
    
    def start_drawing(self, x: int, y: int) -> None:
        """Start drawing a new bounding box."""
        self.drawing = True
        self.start_point = (x, y)
        self.current_box = (x, y, x, y)
        self.selected_box = None
    
    def update_drawing(self, x: int, y: int) -> None:
        """Update current bounding box while drawing."""
        if self.drawing and self.start_point:
            x1 = min(self.start_point[0], x)
            y1 = min(self.start_point[1], y)
            x2 = max(self.start_point[0], x)
            y2 = max(self.start_point[1], y)
            self.current_box = (x1, y1, x2, y2)
    
    def finish_drawing(self, x: int, y: int) -> None:
        """Finish drawing and add bounding box."""
        if self.drawing and self.start_point:
            x1 = min(self.start_point[0], x)
            y1 = min(self.start_point[1], y)
            x2 = max(self.start_point[0], x)
            y2 = max(self.start_point[1], y)
            
            # Only add if box is large enough
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.save_state()
                self.boxes.append((x1, y1, x2, y2, self.current_class))
                self.redo_stack.clear()  # Clear redo when new action is performed
            
            self.drawing = False
            self.start_point = None
            self.current_box = None
    
    def select_box(self, x: int, y: int) -> Optional[int]:
        """Select a box at the given coordinates."""
        for i, (x1, y1, x2, y2, _) in enumerate(self.boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_box = i
                return i
        self.selected_box = None
        return None
    
    def delete_selected_box(self) -> bool:
        """Delete the currently selected box."""
        if self.selected_box is not None:
            self.save_state()
            del self.boxes[self.selected_box]
            self.selected_box = None
            self.redo_stack.clear()
            return True
        return False
    
    def set_class(self, class_id: int) -> None:
        """Set the current class for new boxes."""
        if 0 <= class_id < len(self.classes):
            self.current_class = class_id
    
    def undo(self) -> bool:
        """Undo the last action."""
        if self.undo_stack:
            self.redo_stack.append(self.boxes.copy())
            self.boxes = self.undo_stack.pop()
            self.selected_box = None
            return True
        return False
    
    def redo(self) -> bool:
        """Redo the last undone action."""
        if self.redo_stack:
            self.undo_stack.append(self.boxes.copy())
            self.boxes = self.redo_stack.pop()
            self.selected_box = None
            return True
        return False
    
    def save_state(self) -> None:
        """Save current state for undo/redo."""
        self.undo_stack.append(self.boxes.copy())
        if len(self.undo_stack) > 50:  # Limit undo history
            self.undo_stack.pop(0)
    
    def clear(self) -> None:
        """Clear all boxes and reset state."""
        self.save_state()
        self.boxes.clear()
        self.selected_box = None
        self.drawing = False
        self.start_point = None
        self.current_box = None


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """Mouse callback for labeling interface."""
    state, frame = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            # Ctrl+click to select box
            state.select_box(x, y)
        else:
            # Start drawing new box
            state.start_drawing(x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drawing:
            state.update_drawing(x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        if state.drawing:
            state.finish_drawing(x, y)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right-click to delete selected box
        state.delete_selected_box()


def draw_labeling_interface(
    frame: np.ndarray, 
    state: LabelingState, 
    instructions: List[str]
) -> np.ndarray:
    """Draw the labeling interface on the frame."""
    annotated = frame.copy()
    
    # Draw existing boxes
    for i, (x1, y1, x2, y2, class_id) in enumerate(state.boxes):
        color = (0, 255, 0) if i == state.selected_box else (0, 140, 255)
        label = f"{state.classes[class_id]}"
        annotated = draw_bounding_box(annotated, x1, y1, x2, y2, label, color)
    
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
    class_text = f"Class: {state.classes[state.current_class]} ({state.current_class})"
    cv2.putText(annotated, class_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw box count
    count_text = f"Boxes: {len(state.boxes)}"
    cv2.putText(annotated, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to save labeled data to (train or val).",
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
    return parser.parse_args()


def main() -> None:
    """Main capture and labeling function."""
    args = parse_args()
    
    # Initialize dataset
    dataset = YOLODataset(args.out_dir)
    dataset.create_structure()
    
    # Load or create classes
    if args.classes and Path(args.classes).exists():
        classes = dataset.load_classes()
        print(f"Loaded {len(classes)} classes from {args.classes}")
    else:
        # Default HotWheels classes
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
    
    # Initialize labeling state
    state = LabelingState(classes)
    
    # Instructions
    instructions = [
        "Mouse: Draw boxes, Ctrl+click to select",
        "Keys: 1-9,0=class, u=undo, r=redo, d=delete",
        "SPACE=capture, ENTER=save, n=next, q=quit"
    ]
    
    # Set mouse callback
    cv2.setMouseCallback(args.window, mouse_callback, (state, None))
    
    frame_count = 0
    frozen_frame = None
    
    try:
        print("Starting labeling interface. Press 'q' to quit...")
        
        while True:
            # Read frame
            if frozen_frame is None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Camera frame read failed; attempting to continue...", file=sys.stderr)
                    continue
            else:
                frame = frozen_frame.copy()
            
            # Run prelabeling if enabled
            if prelabel_model and frozen_frame is None:
                try:
                    device_arg = get_device_string(args.device)
                    results = predict(
                        model=prelabel_model,
                        source=frame,
                        conf=args.conf,
                        device=device_arg,
                        verbose=False,
                    )
                    
                    # Convert prelabeling results to boxes
                    if results and len(results) > 0:
                        result = results[0]
                        boxes = result.boxes if hasattr(result, "boxes") else None
                        if boxes and len(boxes) > 0:
                            for box in boxes:
                                xyxy = box.xyxy[0].tolist()
                                conf = float(box.conf[0]) if box.conf is not None else 0.0
                                cls_id = int(box.cls[0]) if box.cls is not None else -1
                                
                                if conf >= args.conf:
                                    x1, y1, x2, y2 = map(int, xyxy)
                                    # Map model class to our classes (simplified)
                                    mapped_class = min(cls_id, len(classes) - 1)
                                    state.boxes.append((x1, y1, x2, y2, mapped_class))
                except Exception as e:
                    print(f"Prelabeling error: {e}", file=sys.stderr)
            
            # Draw interface
            annotated = draw_labeling_interface(frame, state, instructions)
            
            # Display frame
            cv2.imshow(args.window, annotated)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
            elif key == ord(" "):  # SPACE - capture/freeze frame
                frozen_frame = frame.copy()
                state.clear()  # Clear any existing boxes
                print("Frame captured. Draw bounding boxes and press ENTER to save.")
            elif key == ord("\r") or key == ord("\n"):  # ENTER - save
                if frozen_frame is not None and len(state.boxes) > 0:
                    # Generate filename
                    filename = f"img_{frame_count:04d}"
                    frame_count += 1
                    
                    # Save image
                    image_path = dataset.save_image(frozen_frame, filename, args.split)
                    
                    # Convert boxes to YOLO format
                    img_height, img_width = frozen_frame.shape[:2]
                    yolo_labels = []
                    for x1, y1, x2, y2, class_id in state.boxes:
                        x_center, y_center, width, height = normalize_coordinates(
                            x1, y1, x2, y2, img_width, img_height
                        )
                        yolo_labels.append((class_id, x_center, y_center, width, height))
                    
                    # Save labels
                    label_path = dataset.save_labels(yolo_labels, filename, args.split)
                    
                    # Save metadata
                    metadata = dataset.create_metadata(
                        frozen_frame, yolo_labels, args.split
                    )
                    dataset.save_metadata(filename, args.split, metadata)
                    
                    print(f"Saved: {image_path} + {label_path}")
                    frozen_frame = None
                    state.clear()
                else:
                    print("No frame captured or no boxes drawn")
            elif key == ord("n"):  # n - next frame without saving
                frozen_frame = None
                state.clear()
                print("Skipped frame")
            elif key == ord("u"):  # u - undo
                if state.undo():
                    print("Undone")
            elif key == ord("r"):  # r - redo
                if state.redo():
                    print("Redone")
            elif key == ord("d"):  # d - delete selected box
                if state.delete_selected_box():
                    print("Deleted selected box")
            elif ord("1") <= key <= ord("9"):  # 1-9 - select class
                class_id = key - ord("1")
                state.set_class(class_id)
                print(f"Selected class: {classes[class_id]} ({class_id})")
            elif key == ord("0"):  # 0 - select 10th class
                if len(classes) > 9:
                    state.set_class(9)
                    print(f"Selected class: {classes[9]} (9)")
    
    except KeyboardInterrupt:
        print("\nLabeling interrupted by user")
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Save data.yaml
        dataset.save_data_yaml(classes)
        print(f"Saved data.yaml with {len(classes)} classes")
        print("Labeling stopped")


if __name__ == "__main__":
    main()
