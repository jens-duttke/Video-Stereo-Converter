# Video Stereo Converter

Convert 2D video to 3D stereoscopic side-by-side format using AI-powered depth estimation with Apple's [DepthPro](https://huggingface.co/apple/DepthPro) model (optimized as ONNX for DirectML). Developed and tested on Windows.

## Features

- Workflow-based processing with centralized configuration
- GPU-accelerated batch processing with resumable workflows
- Flexible pipeline where each step runs independently
- Interactive settings tester with config save functionality
- Built-in disk space management

## Installation

### Prerequisites

- [FFmpeg](https://www.ffmpeg.org) - for video frame extraction and encoding
- [Python 3.12](https://www.python.org) - recommended for ROCm compatibility

### Setup

```bash
# Create and activate virtual environment
python -m venv .
Scripts\activate

# Install required packages
pip install onnxruntime-directml numpy opencv-python tqdm kornia
```

### Optional: AMD ROCm Support for Windows

For enhanced performance on AMD GPUs on Windows, install the [ROCm PyTorch version](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html):

```bash
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_devel-0.1.dev0-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm-0.1.dev0.tar.gz

pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torch-2.9.0+rocmsdk20251116-cp312-cp312-win_amd64.whl
```

## Workflow

All scripts use a centralized workflow directory containing a `config.json` file with processing parameters. This eliminates the need to specify paths and settings for each command.

### Step 1: Initialize Workflow

Create a new workflow directory with default configuration:

```bash
# Creates workflow/ subdirectory next to input video
python workflow_init.py --input-video "C:/Videos/input.mkv"

# Or specify a custom workflow directory
python workflow_init.py --input-video "C:/Videos/input.mkv" --workflow-dir "D:/Video-Processing/workflow"
```

This creates the following structure:

```
D:/Video-Processing/workflow/
    config.json
    frames/
    depth_maps/
    sbs/
    chunks/
```

### Step 2: Extract Frames

Extract video frames as PNG images:

```bash
python frame_extractor.py "D:/Video-Processing/workflow"
```

### Step 3: Generate Depth Maps

Process frames to create depth maps using the DepthPro model:

```bash
python depth_map_generator.py "D:/Video-Processing/workflow"
```

Optional parameters:
- `--cpu` - Force CPU inference
- `--batch-size N` - Override batch size from config
- `--start-frame N` - First frame to process
- `--end-frame N` - Last frame to process

### Step 4: Test Settings (Optional)

Find optimal 3D depth settings interactively. Best used with a 3D display.

```bash
python sbs_tester.py "D:/Video-Processing/workflow"
```

Features:
- Frame navigation with Prev/Next buttons
- Real-time preview of stereo output
- Save button updates stereo parameters in `config.json`
- Fullscreen mode for 3D monitors (press F)
- Switch between monitors (press M)

### Step 5: Generate Side-by-Side Images

Create stereoscopic image pairs using settings from config:

```bash
python sbs_generator.py "D:/Video-Processing/workflow"
```

Optional parameters:
- `--cpu` - Force CPU processing

### Step 6: Create Video Chunks

Encode SBS images into video segments:

```bash
python chunk_generator.py "D:/Video-Processing/workflow"
```

Optional parameters:
- `--end-frame N` - Process frames up to this number

Creates files like `sbs_000001_002000.mkv` in the `chunks/` directory. The last frame of each chunk is preserved as the first frame of the next chunk to ensure seamless concatenation.

### Step 7: Concatenate with Audio

Merge all chunks and add the original audio track:

```bash
python video_concatenator.py "D:/Video-Processing/workflow"
```

Automatically detects overlapping frames between chunks and skips duplicate first frames during concatenation.

Output path is defined in `config.json` (default: same directory as input video, named `{input_stem}_stereo.mkv`).

## Configuration

### config.json Structure

The workflow configuration file contains all processing parameters:

```json
{
    "input_video": "C:/Videos/input.mkv",
    "output_video": "C:/Videos/input_stereo.mkv",
    "directories": {
        "frames": "frames",
        "depth_maps": "depth_maps",
        "sbs": "sbs",
        "chunks": "chunks"
    },
    "stereo": {
        "max_disparity": 50.0,
        "convergence": 0.0,
        "super_sampling": 3.0,
        "edge_softness": 20.0,
        "smoothing_strength": 1.0,
        "depth_gamma": 0.2,
        "sharpen": 14.0
    },
    "depth": {
        "save_16bit": false,
        "batch_size": 1
    },
    "encoding": {
        "crf": 19,
        "preset": "slow"
    },
    "free_space": {
        "sbs_generator": "frame",
        "chunk_generator": "sbs"
    }
}
```

### Configuration Parameters

**Paths:**
- `input_video` - Absolute path to the source video file
- `output_video` - Output path (relative to workflow directory or absolute)
- `directories` - Subdirectory names within the workflow

**Stereo Parameters:**
- `max_disparity` - Maximum pixel displacement (recommended: 14-60)
- `convergence` - Focal plane shift (positive = pop-out, negative = sink-in)
- `super_sampling` - Internal upscale factor (recommended: 1.0-3.0)
- `edge_softness` - Depth edge softening strength (recommended: 0-20)
- `smoothing_strength` - Warping artifact smoothing (recommended: 0-5)
- `depth_gamma` - Depth gamma correction (recommended: 0.2-0.5)
- `sharpen` - Unsharp mask strength (recommended: 0.5-16.0)

**Depth Generation:**
- `save_16bit` - Save depth maps as 16-bit TIFF (slightly reduces artifacts, about 10 times larger files)
- `batch_size` - Inference batch size (higher = more VRAM usage and maybe faster but often slower)

**Video Encoding:**
- `crf` - Constant Rate Factor for libx265 (lower = higher quality)
- `preset` - FFmpeg encoding preset (slower = better compression)

**Disk Space Management:**
- `free_space.sbs_generator` - Delete source images after successful SBS image generation:
  - `"frame"` - Delete only frame files (default)
  - `"depth"` - Delete only depth map files
  - `"all"` - Delete both frame and depth files
  - `"none"` - Keep all files
- `free_space.chunk_generator` - Delete source imaegs after successful video chunk encoding:
  - `"sbs"` - Delete SBS files after a chuck is encoded (default)
  - `"none"` - Keep all files

### Depth Map File Preference

When looking for depth maps, scripts prefer `.tif` files over `.png` files. If both exist for a frame, the `.tif` version is used.

### Config Validation

The configuration is strictly validated on load. Errors are thrown for:
- Missing required keys
- Unknown extra keys
- Wrong value types

Error messages show which keys are problematic with expected types and example values.

### CLI Parameter Override

Some parameters can be overridden via command line. When both config and CLI specify a value, CLI wins. However, missing required config keys always cause errors regardless of CLI arguments.

## Additional Information

### Disk Space

The conversion pipeline creates large amounts of intermediate data (~5 GB per 1000 frames at 1080p). Plan accordingly and consider processing in chunks.

### Resumable Processing

All scripts automatically skip already-processed files, making it safe to interrupt and resume processing at any time.

### Script Reference

| Script | Description |
|--------|-------------|
| [`workflow_init.py`](./workflow_init.py) | Create new workflow with default config |
| [`frame_extractor.py`](./frame_extractor.py) | Extract frames from input video |
| [`depth_map_generator.py`](./depth_map_generator.py) | Generate depth maps from frames |
| [`sbs_tester.py`](./sbs_tester.py) | Interactive parameter testing |
| [`sbs_generator.py`](./sbs_generator.py) | Create side-by-side stereo images |
| [`chunk_generator.py`](./chunk_generator.py) | Encode SBS images to video chunks |
| [`video_concatenator.py`](./video_concatenator.py) | Concatenate chunks with audio |

Run any script with `-h` to see all available options.

### Linux/macOS Compatibility

Most scripts use cross-platform libraries (Python, FFmpeg, OpenCV), but some components have Windows-specific dependencies:

| Component | Windows | Linux/macOS |
|-----------|---------|-------------|
| Core pipeline (frame extraction, encoding) | Yes | Should work |
| Depth map generation | Yes | Yes (install appropriate onnxruntime) |
| Interactive tester (`sbs_tester.py`) | Yes | Requires code changes |

**Setup for Linux/macOS:**
- Install `onnxruntime-gpu` (CUDA) or `onnxruntime` (CPU) instead of `onnxruntime-directml`
- The depth map generator auto-detects the best available execution provider
- The `sbs_tester.py` script uses Windows APIs (`ctypes.windll`, `winsound`) for monitor detection and audio feedback
