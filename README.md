# HotWheels Detection & Labeling System

A Python application for real-time HotWheels car detection and dataset creation using YOLOv8 and OpenCV, optimized for macOS.

## Features

- **Real-time Detection**: Live camera feed with YOLOv8 inference and bounding box overlays
- **Data Collection**: Camera-based image capture with manual labeling interface
- **YOLO Dataset Management**: Automatic YOLO format generation with validation
- **macOS Optimized**: AVFoundation camera backend with proper permission handling
- **Prelabeling**: AI-assisted labeling using trained models
- **Quality Controls**: Sharpness detection and exposure analysis

## Quick Start

### Prerequisites

- Python 3.9+ (3.11+ recommended)
- macOS with camera access
- Terminal/IDE with camera permissions

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bruno.molinari12/HotWheels.git
   cd HotWheels
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install ultralytics opencv-python numpy pyyaml rich
   ```

4. **Install in development mode**:
   ```bash
   pip install -e .
   ```

### Camera Permissions (macOS)

Before running, ensure camera access is granted:
1. Open **System Settings** → **Privacy & Security** → **Camera**
2. Enable access for your Terminal/IDE (Terminal, iTerm, Cursor, PyCharm, etc.)
3. If issues persist, try `--backend avfoundation` flag

## Usage

### Real-time Detection

```bash
# Basic detection (requires trained model)
hotwheels-detect --model runs/detect/train/weights/best.pt --backend avfoundation

# With MPS acceleration on Apple Silicon
hotwheels-detect --model runs/detect/train/weights/best.pt --device mps --show-fps

# Custom confidence threshold
hotwheels-detect --model best.pt --conf 0.7 --backend avfoundation
```

### Data Collection & Labeling

```bash
# Manual labeling mode
hotwheels-capture --backend avfoundation --split train --out-dir dataset

# With prelabeling assistance
hotwheels-capture --backend avfoundation --split train --out-dir dataset \
  --prelabel --model runs/detect/train/weights/best.pt --device mps

# Validation mode
hotwheels-capture --validate dataset/
```

### Training a Model

1. **Prepare your dataset**:
   - Collect images using `hotwheels-capture`
   - Ensure proper train/val split
   - Validate with `hotwheels-capture --validate dataset/`

2. **Create data.yaml**:
   ```yaml
   path: ./dataset
   train: images/train
   val: images/val
   names:
     0: Mustang_GT_Blue
     1: Camaro_SS_Red
     2: Civic_TypeR_White
   ```

3. **Train with YOLOv8**:
   ```bash
   yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=100 device=auto
   ```

## Project Structure

```
HotWheels/
├── cli/
│   ├── detect.py      # Real-time detection CLI
│   └── capture.py     # Data collection & labeling CLI
├── core/
│   ├── camera.py      # Camera handling, FPS estimation
│   ├── yolo.py        # Model loading, inference, names
│   ├── viz.py         # Drawing utilities, overlays
│   ├── dataset.py     # YOLO I/O, validation, manifests
│   └── utils.py       # Path helpers, atomic writes
├── tests/
│   └── test_dataset.py
├── hotwheels_detector.py  # Legacy single script
├── IMPLEMENTATION_PLAN.md
└── README.md
```

## Dataset Format

The system generates YOLO-compatible datasets:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_0001.jpg
│   │   └── img_0002.jpg
│   └── val/
│       └── img_1001.jpg
├── labels/
│   ├── train/
│   │   ├── img_0001.txt
│   │   └── img_0002.txt
│   └── val/
│       └── img_1001.txt
├── data.yaml
└── classes.txt
```

Each label file contains normalized coordinates:
```
0 0.512 0.438 0.200 0.120
1 0.300 0.600 0.150 0.100
```

## Command Reference

### Detection CLI (`hotwheels-detect`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to YOLO model (.pt) | `runs/detect/train/weights/best.pt` |
| `--backend` | Camera backend | `auto` |
| `--camera` | Camera index | `0` |
| `--device` | Inference device | `auto` |
| `--conf` | Confidence threshold | `0.5` |
| `--show-fps` | Display FPS overlay | `False` |

### Capture CLI (`hotwheels-capture`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--out-dir` | Dataset output directory | `dataset` |
| `--split` | Dataset split (train/val) | `train` |
| `--classes` | Class names file | Auto-generated |
| `--prelabel` | Enable AI-assisted labeling | `False` |
| `--validate` | Validate dataset integrity | `False` |

### Keyboard Shortcuts (Capture Mode)

| Key | Action |
|-----|--------|
| `SPACE` | Capture/freeze frame |
| `1-9,0` | Select class (0 = 10th) |
| `u` | Undo last action |
| `r` | Redo |
| `ESC` | Cancel current annotations |
| `ENTER` | Save image + labels |
| `n` | Next frame (skip save) |
| `q` | Quit application |

## Troubleshooting

### Camera Issues
- **Permission denied**: Check System Settings → Privacy & Security → Camera
- **Camera not found**: Try different `--camera` index (0, 1, 2)
- **Poor performance**: Use `--backend avfoundation` on macOS

### Model Issues
- **Model not found**: Train a model first or provide correct `--model` path
- **Low accuracy**: Increase training data, adjust confidence threshold
- **Slow inference**: Use `--device mps` on Apple Silicon

### Dataset Issues
- **Invalid labels**: Run `hotwheels-capture --validate dataset/`
- **Missing files**: Check file permissions and disk space
- **Class mismatch**: Ensure `classes.txt` is consistent

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
```bash
# Format code
black hotwheels/
isort hotwheels/

# Type checking
mypy hotwheels/
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for computer vision
- Apple for macOS camera integration

## Support

For issues and questions:
- Create an issue on GitHub
- Email: bruno.molinari12@gmail.com
