# HotWheels Detection & Labeling System - Implementation Plan

## Overview
Split the monolithic `hotwheels_detector.py` into two focused CLIs with shared core modules for better maintainability, testability, and user experience.

## Current State
- Single script: `hotwheels_detector.py` (364 lines)
- Real-time YOLOv8 detection with OpenCV camera capture
- macOS-optimized with AVFoundation backend
- Command-line arguments for model, camera, device, confidence, etc.

## Target Architecture

```
hotwheels/
├── cli/
│   ├── detect.py      # Real-time inference CLI
│   └── capture.py     # Data collection & labeling CLI
├── core/
│   ├── camera.py      # Camera handling, FPS estimation
│   ├── yolo.py        # Model loading, inference, names
│   ├── viz.py         # Drawing utilities, overlays
│   ├── dataset.py     # YOLO I/O, validation, manifests
│   └── utils.py       # Path helpers, atomic writes, hashing
├── tests/
│   └── test_dataset.py
├── pyproject.toml
└── README.md
```

## Progress Summary

**Overall Progress**: 75% Complete (3 of 4 major phases completed)

### ✅ **COMPLETED PHASES**
- **Phase 1**: Foundation & Core Modules (100%)
- **Phase 2**: Detection CLI (100%) 
- **Phase 3**: Capture CLI (95% - missing quality controls)

### 🔄 **IN PROGRESS**
- **Phase 4**: Advanced Features (25% - prelabeling implemented)

### ❌ **PENDING**
- **Phase 5**: Packaging & Documentation (0%)

### 🎯 **NEXT PRIORITIES**
1. Complete Phase 4 quality controls (sharpness detection, exposure analysis)
2. Add dataset validation mode (`--validate`)
3. Update comprehensive README with usage examples
4. Performance optimization and advanced testing

---

## Implementation Phases

### Phase 1: Foundation & Core Modules ✅ **COMPLETED**
**Goal**: Extract shared functionality into reusable core modules

**Tasks**:
1. ✅ Create package structure (`hotwheels/cli/`, `hotwheels/core/`)
2. ✅ Extract camera handling from current script:
   - `core/camera.py`: Camera open/close, AVFoundation fallback, FPS estimation
   - Support for multiple backends and camera indices
   - Graceful error handling with clear macOS permission guidance
3. ✅ Extract YOLO functionality:
   - `core/yolo.py`: Model loading, device selection (auto/cpu/mps/cuda), names handling
   - Predict wrapper with error handling
   - Support for custom names files
4. ✅ Extract visualization utilities:
   - `core/viz.py`: Box drawing, label overlays, FPS display
   - Mouse interaction helpers for labeling UI
5. ✅ Create dataset utilities:
   - `core/dataset.py`: YOLO format read/write, normalization, validation
   - `data.yaml` and `classes.txt` maintenance
   - Metadata tracking (capture time, device, hash)
6. ✅ Create utility functions:
   - `core/utils.py`: Safe filenames, atomic writes, hashing, RNG seeding

**Acceptance Criteria**:
- ✅ All imports compile without errors
- ✅ No linter warnings
- ✅ Core modules have comprehensive docstrings and type hints
- ✅ Unit tests for critical functions (normalization, validation)

### Phase 2: Detection CLI ✅ **COMPLETED**
**Goal**: Create focused detection CLI with current functionality

**Tasks**:
1. ✅ Implement `cli/detect.py`:
   - Arguments: `--model`, `--backend`, `--camera`, `--device`, `--imgsz`, `--conf`, `--iou`, `--show-fps`, `--names`, `--window`
   - Camera capture loop using `core/camera`
   - Model inference using `core/yolo`
   - Visualization using `core/viz`
   - Clean exit on 'q' keypress
2. ✅ Maintain feature parity with current script
3. ✅ Add error handling for common failure modes

**Acceptance Criteria**:
- ✅ Identical behavior to current `hotwheels_detector.py`
- ✅ Works reliably on macOS with `--backend avfoundation`
- ✅ Clear error messages for camera/model failures
- ✅ Performance equivalent to current implementation

### Phase 3: Capture CLI (Manual Labeling) ✅ **COMPLETED**
**Goal**: Create data collection and manual labeling interface

**Tasks**:
1. ✅ Implement `cli/capture.py`:
   - Arguments: `--backend`, `--camera`, `--out-dir`, `--split`, `--classes`, `--window`
   - Camera capture with freeze-frame capability
   - Mouse-driven bounding box creation/editing:
     - Draw boxes with mouse drag
     - Move/resize existing boxes
     - Delete boxes with right-click or 'd' key
   - Keyboard controls:
     - `1-9,0`: Select class (0 = 10th class)
     - `u/r`: Undo/redo
     - `ESC`: Cancel current annotations
     - `ENTER`: Save image + labels
     - `n`: Next frame without saving
     - `SPACE`: Capture/freeze frame
2. ✅ Implement YOLO format saving:
   - Normalized coordinates [0,1]
   - One `.txt` file per image
   - Validation of box bounds and class IDs
3. ✅ Implement dataset management:
   - Auto-create `data.yaml` and `classes.txt`
   - Maintain canonical class order
   - Track metadata in sidecar JSON files
4. ❌ Add quality controls:
   - Sharpness detection (Laplacian variance)
   - Exposure histogram analysis
   - Warnings for blurry/underexposed images

**Acceptance Criteria**:
- ✅ Intuitive mouse/keyboard interface
- ✅ Robust box editing (move, resize, delete)
- ✅ Correct YOLO format output
- ❌ Quality warnings for poor captures
- ✅ Atomic file operations (no partial writes)

**BONUS FEATURES IMPLEMENTED**:
- ✅ Prelabeling integration (`--prelabel` flag with AI-assisted labeling)
- ✅ Enhanced error handling and user feedback

### Phase 4: Advanced Features
**Goal**: Add productivity and quality features

**Tasks**:
1. Prelabeling integration:
   - `--prelabel` flag to enable YOLO-assisted labeling
   - Arguments: `--model`, `--conf`, `--imgsz`, `--device`
   - Overlay predicted boxes as editable annotations
   - Performance throttling (every N frames or on-demand)
2. Dataset validation:
   - `--validate` mode to check dataset integrity
   - Detect missing/malformed labels
   - Report per-class counts and split ratios
   - Identify potential data leakage
3. Review mode:
   - Browse existing labeled images
   - Edit existing annotations
   - Batch operations (rename classes, delete images)
4. Split management:
   - Deterministic train/val splitting
   - Near-duplicate detection and prevention
   - Scene-based grouping to avoid leakage

**Acceptance Criteria**:
- Prelabeling improves labeling speed by 2-3x
- Validation catches common dataset errors
- Review mode allows efficient annotation editing
- Split tooling prevents data leakage

### Phase 5: Packaging & Documentation
**Goal**: Make system production-ready and user-friendly

**Tasks**:
1. Create `pyproject.toml`:
   - Dependencies: `ultralytics`, `opencv-python`, `numpy`, `pyyaml`
   - Optional: `rich` for enhanced CLI output
   - Console scripts for both CLIs
2. Write comprehensive README:
   - Installation instructions (venv, macOS permissions)
   - Usage examples for both CLIs
   - Dataset structure documentation
   - Troubleshooting guide
3. Add basic test suite:
   - Dataset I/O round-trip tests
   - Validation logic tests
   - Camera handling tests (mocked)
4. Performance optimization:
   - Profile and optimize hot paths
   - Memory usage optimization
   - FPS improvements for detection

**Acceptance Criteria**:
- `pip install -e .` works correctly
- Both CLIs accessible from anywhere
- README enables new users to succeed
- Tests provide confidence in core functionality

## Technical Specifications

### Camera Handling
- **Primary Backend**: `cv2.CAP_AVFOUNDATION` (macOS optimized)
- **Fallback**: Default OpenCV backend
- **Error Handling**: Clear permission guidance, camera index suggestions
- **Performance**: FPS estimation, optional throttling

### Dataset Format
- **Structure**: YOLO format with `images/{split}/` and `labels/{split}/`
- **Coordinates**: Normalized to [0,1] range
- **Validation**: Bounds checking, class ID validation, file integrity
- **Metadata**: JSON sidecar files with capture details

### Model Support
- **Primary**: YOLOv8 (ultralytics)
- **Devices**: Auto-detect, CPU, MPS (Apple Silicon), CUDA
- **Formats**: `.pt` weights files
- **Names**: Model-inferred or custom `classes.txt`

### User Interface
- **Detection**: Real-time overlay with configurable confidence
- **Labeling**: Mouse-driven box editing, keyboard shortcuts
- **Feedback**: Visual indicators, quality warnings, progress tracking

## Risk Assessment & Mitigation

### High Risk
1. **Camera Permission Issues on macOS**
   - *Risk*: Users can't access camera
   - *Mitigation*: Clear documentation, permission check, fallback backends

2. **Performance Degradation with Prelabeling**
   - *Risk*: UI becomes unresponsive
   - *Mitigation*: Throttling, optional feature, device optimization

### Medium Risk
1. **Dataset Corruption During Labeling**
   - *Risk*: Lost work due to crashes
   - *Mitigation*: Atomic writes, autosave, recovery mechanisms

2. **Class Order Drift**
   - *Risk*: Inconsistent class IDs across sessions
   - *Mitigation*: Canonical `classes.txt`, validation warnings

### Low Risk
1. **Cross-platform Compatibility**
   - *Risk*: Windows/Linux issues
   - *Mitigation*: Focus on macOS first, test on other platforms later

## Success Metrics
- **Functionality**: Feature parity with current script + labeling capabilities
- **Performance**: Detection FPS ≥ current implementation
- **Usability**: New users can label 100+ images in <2 hours
- **Reliability**: <1% data corruption rate during labeling
- **Maintainability**: Clear separation of concerns, comprehensive tests

## Timeline Estimate
- **Phase 1**: 2-3 days (foundation)
- **Phase 2**: 1-2 days (detection CLI)
- **Phase 3**: 3-4 days (capture CLI)
- **Phase 4**: 2-3 days (advanced features)
- **Phase 5**: 1-2 days (packaging/docs)
- **Total**: 9-14 days

## Future Enhancements
- Video annotation support
- Multi-class batch operations
- Export to other formats (COCO, Pascal VOC)
- Cloud storage integration
- Collaborative labeling features
- Advanced augmentation during capture
