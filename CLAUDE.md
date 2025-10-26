# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HotWheels Detection & Labeling System - A Python application for real-time YOLOv8 object detection and dataset creation, optimized for macOS. The system provides two CLI tools: one for real-time inference and one for data collection with manual labeling.

**Key Technologies**: YOLOv8 (ultralytics), OpenCV, Python 3.9+

## Development Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dependencies
pip install --upgrade pip
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

## Essential Commands

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hotwheels --cov-report=html

# Skip camera tests (require hardware)
pytest tests/ -v -m "not camera"

# Run specific test file
pytest tests/test_dataset.py -v
```

### Code Quality
```bash
# Format code
black hotwheels/
isort hotwheels/

# Type checking
mypy hotwheels/

# Run all quality checks together
black hotwheels/ && isort hotwheels/ && mypy hotwheels/ && pytest tests/ -v
```

### Using the CLIs
```bash
# Real-time detection (requires trained model)
hotwheels-detect --model runs/detect/train/weights/best.pt --backend avfoundation

# Data collection with manual labeling
hotwheels-capture --backend avfoundation --split train --out-dir dataset

# With prelabeling assistance
hotwheels-capture --backend avfoundation --prelabel --model best.pt --device mps

# Dataset validation
hotwheels-capture --validate --out-dir dataset/
```

## Architecture

The codebase is organized as a modular Python package with strict separation between CLI interfaces and reusable core functionality:

### Package Structure
```
hotwheels/
├── cli/           # Command-line interfaces (user-facing)
│   ├── detect.py  # Real-time detection CLI
│   └── capture.py # Data collection & labeling CLI
└── core/          # Reusable core modules
    ├── camera.py  # Camera handling, FPS estimation
    ├── yolo.py    # Model loading, inference
    ├── viz.py     # Drawing utilities, overlays
    ├── dataset.py # YOLO I/O, validation
    └── utils.py   # Path helpers, quality assessment
```

### Core Design Principles

1. **Separation of Concerns**: CLI code handles argument parsing and user interaction; core modules provide reusable functionality
2. **macOS Optimization**: Uses `CAP_AVFOUNDATION` backend for reliable camera access with graceful fallback
3. **YOLO Dataset Format**: All I/O uses normalized coordinates [0,1] with strict validation
4. **Quality First**: Built-in sharpness detection and exposure analysis to maintain dataset quality
5. **Atomic Operations**: File writes use temporary files + atomic rename to prevent data corruption

### Key Modules Explained

**camera.py**: Abstracts camera capture with backend selection (AVFoundation for macOS). The `make_video_capture()` function tries AVFoundation first, then falls back to default backend. Always provides clear error messages for permission issues.

**yolo.py**: Handles YOLO model lifecycle - loading from `.pt` files, device selection (auto/cpu/mps/cuda), and inference. The `load_names()` function prioritizes external class files over model-embedded names for flexibility.

**dataset.py**: The `YOLODataset` class manages complete dataset lifecycle - directory structure creation, image/label I/O with validation, and comprehensive validation with per-class statistics. All coordinates are validated to be in [0,1] range.

**viz.py**: Provides drawing primitives for both detection (bounding boxes with labels) and labeling UI (interactive boxes, crosshairs, quality indicators). Handles all OpenCV drawing operations consistently.

**utils.py**: Contains cross-cutting utilities - filename sanitization, atomic file writes, image quality assessment (sharpness via Laplacian, exposure via histograms), and validation report formatting.

## YOLO Dataset Format

The system generates YOLO-compatible datasets with this structure:

```
dataset/
├── images/{split}/    # Raw images (.jpg)
├── labels/{split}/    # YOLO labels (.txt, one per image)
├── metadata/          # JSON metadata (optional)
├── data.yaml          # YOLO config (paths + class names)
└── classes.txt        # Canonical class order
```

**Label Format**: One line per object: `class_id x_center y_center width height` (all coordinates normalized to [0,1])

**Important**: Class order in `classes.txt` is canonical and must remain consistent. Adding new classes appends to the end.

## Camera Handling on macOS

Camera access requires explicit permissions:
1. **System Settings** → **Privacy & Security** → **Camera**
2. Enable for Terminal/IDE (Terminal, iTerm, Cursor, PyCharm, etc.)
3. Application restart may be required after granting permission

**Troubleshooting**: If camera fails to open, try `--backend avfoundation` explicitly or test different camera indices with `--camera 1` or `--camera 2`.

## Testing Philosophy

- Tests use pytest with markers: `@pytest.mark.camera` for hardware-dependent tests
- Camera tests are skipped in CI (`-m "not camera"`)
- Dataset I/O tests verify round-trip correctness (write → read → compare)
- Validation tests use synthetic datasets to cover edge cases
- Mock camera captures to avoid hardware dependency in unit tests

## Important Implementation Notes

### Device Selection
The codebase supports `auto`, `cpu`, `mps` (Apple Silicon), and `cuda`. Auto-detection happens in ultralytics, but explicit device selection is available for performance tuning.

### Quality Controls
The capture CLI includes real-time quality assessment:
- **Sharpness**: Laplacian variance (blurry if < 100)
- **Exposure**: Histogram analysis (dark if mean brightness < 50)
- Use `--quality-check` flag to enable warnings
- Use `--quality-threshold good|fair|poor` to enforce minimums

### Prelabeling Workflow
When `--prelabel` is enabled, the system runs YOLO inference on each captured frame and overlays predictions as editable boxes. This accelerates labeling by 2-3x but requires a trained model.

### Dataset Validation
The validation mode (`--validate`) performs comprehensive checks:
- Directory structure integrity
- Label file format and coordinate bounds
- Per-class object counts and split ratios
- Missing labels, empty labels, malformed labels
- Optional duplicate detection (`--check-duplicates`)

## Configuration Files

**pyproject.toml**: Defines project metadata, dependencies, console scripts (`hotwheels-detect`, `hotwheels-capture`), and tool configurations (black, isort, mypy, pytest).

**data.yaml**: Generated by capture CLI, required for YOLO training. Contains dataset path, split paths, and class name mapping.

## Common Development Tasks

### Adding a New Core Module
1. Create file in `hotwheels/core/`
2. Add comprehensive docstrings and type hints
3. Create corresponding test file in `tests/`
4. Import in relevant CLI files

### Adding a New CLI Command
1. Create new file in `hotwheels/cli/` or extend existing
2. Add console script entry in `pyproject.toml` under `[project.scripts]`
3. Reinstall package: `pip install -e .`
4. Follow argparse patterns from existing CLIs

### Modifying Dataset Format
1. Update `YOLODataset` class in `core/dataset.py`
2. Update validation logic to match new format
3. Add tests for new format in `tests/test_dataset.py`
4. Update README.md dataset structure section

### Training a Model
```bash
# Ensure data.yaml exists with correct paths
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=100 device=auto

# For Apple Silicon with MPS
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=100 device=mps
```

Trained weights will be at `runs/detect/train/weights/best.pt`.

## Project Status

This is a mature, working system. All 5 implementation phases are complete (see IMPLEMENTATION_PLAN.md). The codebase includes:
- ✅ Real-time detection CLI
- ✅ Data collection & manual labeling CLI
- ✅ Prelabeling integration
- ✅ Quality controls
- ✅ Dataset validation
- ✅ Comprehensive test suite (29 tests passing)

Optional future enhancements: review mode for editing existing labels, advanced split management, performance profiling.
