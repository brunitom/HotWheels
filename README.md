# 🏎️ HotWheels Detection & Labeling System

A Python application for real-time HotWheels car detection and dataset creation using YOLOv8 and OpenCV, optimized for macOS.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)](tests/)

## ✨ Features

- **🎥 Real-time Detection**: Live camera feed with YOLOv8 inference and bounding box overlays
- **📸 Smart Data Collection**: Interactive labeling with duplicate detection and class management
- **🎯 Dynamic Class Management**: Add new car classes on-the-fly with intelligent duplicate detection
- **🔄 Persistent Class Storage**: Classes automatically saved and available across sessions
- **🤖 Prelabeling**: AI-assisted labeling using trained models to speed up annotation
- **✅ Quality Controls**: Real-time sharpness detection and exposure analysis with quality thresholds
- **📊 Dataset Validation**: Comprehensive validation with per-class statistics and duplicate detection
- **🗃️ Metadata Tracking**: Automatic metadata generation with quality metrics
- **🍎 macOS Optimized**: AVFoundation camera backend with proper permission handling

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

#### Getting Started with Capture

```bash
# Basic capture with quality checks
hotwheels-capture --backend avfoundation --out-dir dataset --quality-check
```

**Interactive Workflow:**
1. **Choose a class**: Type a number (0-9) to select existing class OR type a new class name
   ```
   Available classes:
     0: Mustang_GT_Blue
     1: Camaro_SS_Red
     ...

   Which car class? (0-9 or type new name): Tesla_Cybertruck_Silver
   ✅ Added new class: Tesla_Cybertruck_Silver (class_id: 10)
   ```

2. **Smart duplicate detection**: System warns if class name is similar to existing ones
   ```
   ⚠️  Similar classes found:
       Subaru_Impreza_WRX (similarity: 93%)

   Add 'Subaru_Imprezza_WRX' anyway? (y/n):
   ```

3. **Select split**: Choose train/val/test for this session

4. **Capture images**: Press 'c' to freeze, draw boxes, press ENTER to save

#### Advanced Capture Options

```bash
# With prelabeling assistance (faster annotation)
hotwheels-capture --backend avfoundation --out-dir dataset \
  --prelabel --model runs/detect/train/weights/best.pt --device mps

# With strict quality thresholds
hotwheels-capture --backend avfoundation --out-dir dataset \
  --quality-check --quality-threshold good

# Validation mode (check dataset integrity)
hotwheels-capture --validate --out-dir dataset/

# Validation with duplicate image detection
hotwheels-capture --validate --check-duplicates --out-dir dataset/
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

## 📂 Project Structure

```
HotWheels/
├── hotwheels/
│   ├── cli/
│   │   ├── detect.py      # Real-time detection CLI
│   │   └── capture.py     # Data collection & labeling CLI (with duplicate detection)
│   └── core/
│       ├── camera.py      # Camera handling, FPS estimation
│       ├── yolo.py        # Model loading, inference, names
│       ├── viz.py         # Drawing utilities, overlays
│       ├── dataset.py     # YOLO I/O, validation, class management
│       └── utils.py       # Path helpers, atomic writes, quality assessment
├── tests/
│   ├── test_capture.py    # New: Similarity detection & class persistence tests
│   ├── test_dataset.py    # Dataset I/O and validation tests
│   ├── test_viz.py        # Visualization tests
│   └── test_camera.py     # Camera handling tests
├── dataset/               # Your captured data (git-ignored)
│   └── classes.txt        # Persistent class list
├── pyproject.toml         # Project config & dependencies
├── CLAUDE.md             # AI assistant instructions
├── IMPLEMENTATION_PLAN.md # Development roadmap
└── README.md
```

## 📁 Dataset Format

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
├── data.yaml        # YOLO training config
└── classes.txt      # Canonical class list (persistent)
```

Each label file contains normalized coordinates:
```
0 0.512 0.438 0.200 0.120
1 0.300 0.600 0.150 0.100
```

### Managing Classes

**`classes.txt` is the source of truth** for your car collection. The file persists across sessions:

```txt
Mustang_GT_Blue
Camaro_SS_Red
Civic_TypeR_White
Tesla_Cybertruck_Silver
```

**Adding Classes:**
- **During capture**: Just type the new name when prompted
- **Manual editing**: Add new lines to `classes.txt` (one class per line)

**Class Naming Rules:**
- ✅ Use underscores: `Ford_F150_Red`
- ✅ Be descriptive: Include model AND color
- ✅ Alphanumeric + underscores only
- ❌ No spaces: `Ford F150` → `Ford_F150`
- ❌ No special chars: `Car#1` → `Car_1`
- ⚠️ **Never reorder or delete** existing classes (breaks label consistency)

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
| `--model` | Model path for prelabeling | None |
| `--conf` | Confidence threshold for prelabeling | `0.5` |
| `--quality-check` | Enable real-time quality analysis | `False` |
| `--quality-threshold` | Min quality (good/fair/poor) | `fair` |
| `--validate` | Validate dataset integrity | `False` |
| `--check-duplicates` | Check for duplicate images | `False` |

### Keyboard Shortcuts (Capture Mode)

| Key | Action |
|-----|--------|
| `c` | Capture/freeze frame |
| `r` | Redo (when drawing box) |
| `ENTER` | Save image + labels |
| `n` | Next frame (skip save) |
| `q` | Quit application |

**Mouse Controls:**
- Click and drag to draw bounding box around car
- Release to finalize box

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
- **Classes disappearing**: Fixed! Classes are now automatically loaded from `dataset/classes.txt` on startup
- **Duplicate class names**: System will warn and ask for confirmation before adding similar names

## 🧪 Development

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hotwheels --cov-report=html

# Skip camera tests (require hardware)
pytest tests/ -v -m "not camera"

# Run specific test file
pytest tests/test_capture.py -v
```

**Test Coverage:** 44 tests covering:
- ✅ Similarity detection for duplicate class prevention
- ✅ Class persistence across sessions
- ✅ Dataset I/O and validation
- ✅ Visualization utilities
- ✅ Camera handling

### Code Quality
```bash
# Format code
black hotwheels/
isort hotwheels/

# Type checking
mypy hotwheels/

# Run all checks together
black hotwheels/ && isort hotwheels/ && mypy hotwheels/ && pytest tests/ -v
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
