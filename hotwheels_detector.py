import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Import lazily to provide a clearer error if ultralytics isn't installed
try:
	from ultralytics import YOLO
except Exception as exc:  # noqa: BLE001
	print(
		"Ultralytics (YOLOv8) is required. Install with: pip install ultralytics\n"
		"If OpenCV is missing: pip install opencv-python\n"
		"If you are on Apple Silicon and want GPU (MPS) acceleration: pip install 'torch>=2.1' 'torchvision>=0.16' --index-url https://download.pytorch.org/whl/cu121\n",
		file=sys.stderr,
	)
	raise


# -----------------------------
# Utility functions
# -----------------------------

def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for the detector script.

	Returns:
		argparse.Namespace: Parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description=(
			"HotWheels real-time detector and labeler using YOLOv8. Press 'q' to quit.\n"
			"Tip: macOS camera permissions: System Settings > Privacy & Security > Camera > allow Terminal/IDE."
		)
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="detect",
		choices=["detect", "label"],
		help=(
			"Run mode: 'detect' for real-time inference, 'label' to capture and annotate snapshots."
		),
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
	parser.add_argument(
		"--out-dir",
		type=str,
		default="dataset",
		help=(
			"Output root for labeling mode. Images and labels saved under images/{split} and labels/{split}."
		),
	)
	parser.add_argument(
		"--split",
		type=str,
		default="train",
		choices=["train", "val"],
		help="Dataset split to save labeled data to (train or val).",
	)
	return parser.parse_args()


def make_video_capture(camera_index: int, backend: str) -> cv2.VideoCapture:
	"""Create a VideoCapture with a macOS-friendly backend.

	On macOS, using CAP_AVFOUNDATION often improves camera access reliability.
	"""
	if backend == "avfoundation":
		cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
	elif backend == "default":
		cap = cv2.VideoCapture(camera_index)
	else:
		# Try AVFoundation first, then fallback to default
		cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
		if not cap.isOpened():
			cap.release()
			cap = cv2.VideoCapture(camera_index)
	return cap


def load_names(names_path: Optional[str], model: YOLO) -> dict:
	"""Load class names, prioritizing an external file if provided.

	Args:
		names_path: Optional path to a newline-delimited names file.
		model: Loaded YOLO model instance.

	Returns:
		Dictionary mapping class indices to names.
	"""
	if names_path:
		p = Path(names_path)
		if not p.exists():
			raise FileNotFoundError(f"Names file not found: {p}")
		names = {i: line.strip() for i, line in enumerate(p.read_text().splitlines()) if line.strip()}
		return names
	# Model has names attribute after loading weights
	return getattr(model, "names", {}) or {}


def draw_detections(
	frame: np.ndarray,
	detections,
	names: dict,
	show_fps: bool,
	fps: Optional[float],
	conf_threshold: float,
) -> np.ndarray:
	"""Draw bounding boxes, labels, and optional FPS on the frame.

	Args:
		frame: BGR image.
		detections: Ultralytics YOLO results per frame (first item expected).
		names: Mapping from class id to name.
		show_fps: Whether to overlay FPS.
		fps: Latest FPS estimate.
		conf_threshold: Minimum confidence for drawing.
	"""
	annotated = frame.copy()
	if detections is None or len(detections) == 0:
		if show_fps and fps is not None:
			cv2.putText(
				annotated,
				f"FPS: {fps:.1f}",
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 255, 0),
				2,
			)
		return annotated

	# YOLOv8 returns a Results list; we take the first result per image
	result = detections[0]
	boxes = result.boxes if hasattr(result, "boxes") else None
	if boxes is None or len(boxes) == 0:
		# Optionally indicate no detections
		cv2.putText(
			annotated,
			"No cars detected",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(0, 255, 255),
			2,
		)
		if show_fps and fps is not None:
			cv2.putText(
				annotated,
				f"FPS: {fps:.1f}",
				(10, 60),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 255, 0),
				2,
			)
		return annotated

	for box in boxes:
		# xyxy, confidence, class id
		xyxy = box.xyxy[0].tolist()
		conf = float(box.conf[0]) if box.conf is not None else 0.0
		cls_id = int(box.cls[0]) if box.cls is not None else -1

		if conf < conf_threshold:
			continue

		x1, y1, x2, y2 = map(int, xyxy)
		label = names.get(cls_id, f"id_{cls_id}")
		text = f"{label} {conf:.2f}"

		# Box
		cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)
		# Text background for readability
		(text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
		cv2.rectangle(
			annotated,
			(x1, max(0, y1 - text_h - baseline - 4)),
			(x1 + text_w + 4, y1),
			(0, 140, 255),
			-1,
		)
		# Text
		cv2.putText(
			annotated,
			text,
			(x1 + 2, max(12, y1 - 6)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 0, 0),
			2,
		)

	if show_fps and fps is not None:
		cv2.putText(
			annotated,
			f"FPS: {fps:.1f}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(255, 255, 0),
			2,
		)

	return annotated


def get_device_string(device: str) -> Optional[str]:
	"""Map CLI device option to ultralytics device argument."""
	if device == "auto":
		return None
	return device


def estimate_fps(prev_ticks: Optional[int], curr_ticks: int) -> Tuple[Optional[float], int]:
	"""Estimate FPS using cv2.getTickCount/getTickFrequency."""
	if prev_ticks is None:
		return None, curr_ticks
		dt = (curr_ticks - prev_ticks) / cv2.getTickFrequency()
	if curr_ticks <= prev_ticks:
		return None, curr_ticks
		dt = (curr_ticks - prev_ticks) / cv2.getTickFrequency()
	fps = 1.0 / dt if dt > 0 else None
	return fps, curr_ticks


def main() -> None:
	args = parse_args()

	model_path = Path(args.model)
	if not model_path.exists():
		print(
			f"Model not found at: {model_path}\n"
			"Train with ultralytics first or provide --model /path/to/best.pt",
			file=sys.stderr,
		)
		# Continue anyway to let YOLO try to resolve built-in models if a name is passed

	# Load YOLO model
	try:
		model = YOLO(str(model_path))
	except Exception as e:  # noqa: BLE001
		print(f"Failed to load model '{model_path}': {e}", file=sys.stderr)
		sys.exit(1)

	names = load_names(args.names, model)

	# Open camera
	cap = make_video_capture(args.camera, args.backend)
	if not cap.isOpened():
		print(
			"Could not open camera.\n"
			"- On macOS, grant camera permission to your Terminal/IDE.\n"
			"- Try --backend avfoundation or a different --camera index.",
			file=sys.stderr,
		)
		sys.exit(2)

	cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

	# FPS measure
	prev_ticks: Optional[int] = None
	last_fps: Optional[float] = None

	device_arg = get_device_string(args.device)

	try:
		while True:
			ok, frame = cap.read()
			if not ok or frame is None:
				print("Camera frame read failed; attempting to continue...", file=sys.stderr)
				continue

			# Inference: returns a list of Results
			try:
				results = model.predict(
					source=frame,
					conf=args.conf,
					iou=args.iou,
					imgsz=args.imgsz,
					device=device_arg,
					verbose=False,
				)
			except Exception as infer_err:  # noqa: BLE001
				print(f"Inference error: {infer_err}", file=sys.stderr)
				results = []

			# FPS update
			curr_ticks = cv2.getTickCount()
			last_fps, prev_ticks = estimate_fps(prev_ticks, curr_ticks)

			annotated = draw_detections(
				frame,
				results,
				names,
				args.show_fps,
				last_fps,
				args.conf,
			)

			cv2.imshow(args.window, annotated)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
		finally:
			# Ensure resources are released even on exception
			cap.release()
			cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
