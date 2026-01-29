1# Video Stereo Converter

Convert 2D video to 3D stereoscopic side-by-side format using a [performance-optimized ONNX version](https://huggingface.co/Jens-Duttke/DepthPro-ONNX-HighPerf) of Apple's [DepthPro](https://huggingface.co/apple/DepthPro) depth estimation model (optimized as ONNX for DirectML). Developed and tested on Windows.

## Features

- Batch processing of multiple videos via workflow orchestrator
- GPU-accelerated depth estimation with resumable workflows
- Flexible pipeline where each step runs independently
- Interactive settings tester with config save functionality
- Built-in disk space management

![Workflow Orchestrator in action](https://raw.githubusercontent.com/jens-duttke/Video-Stereo-Converter/main/assets/screenshot1.png)

![SBS Tester in action](https://raw.githubusercontent.com/jens-duttke/Video-Stereo-Converter/main/assets/screenshot2.jpg)

![Converted example video](https://raw.githubusercontent.com/jens-duttke/Video-Stereo-Converter/main/assets/majestic-aerial-view-of-a-historic-castle_stereo.mp4)

*Example clip from [Pexels](https://pexels.com) by _ MARROS _ ([Historic Castle](https://www.pexels.com/video/majestic-aerial-view-of-a-historic-castle-33250829/))*

Watch your converted videos with the free [SBS Video Player](https://www.duttke.de/sbs-video-player/) - a browser-based 3D media player for glasses-free displays. No installation required, fully local processing, complete privacy.

## Quick Start

For processing multiple videos, use the workflow orchestrator:

```bash
# 1. Initialize workflows for each video
python workflow_init.py --input-video "C:/Videos/video1.mkv"
python workflow_init.py --input-video "C:/Videos/video2.mkv"

# 2. Create a workflows.yaml file with one workflow path per line (add ":" suffix)
C:/Videos/video1/workflow:
C:/Videos/video2/workflow:

# 3. Run the orchestrator
python workflow_orchestrator.py workflows.yaml
```

The orchestrator automatically manages all processing steps, runs tasks in parallel where possible, and handles failures gracefully.

## Installation

### Prerequisites

- [FFmpeg](https://www.ffmpeg.org) - for video frame extraction and encoding
- [Python 3.12](https://www.python.org) - recommended for ROCm compatibility

### Setup

```bash
# Clone the repository
git clone https://github.com/jens-duttke/Video-Stereo-Converter.git
cd Video-Stereo-Converter

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install ONNX Runtime for your hardware
pip install onnxruntime-directml  # AMD/Intel GPU (Windows)
pip install onnxruntime-gpu       # NVIDIA GPU (CUDA)
pip install onnxruntime           # CPU only
```

### Optional: AMD ROCm Support for Windows

For enhanced SBS generation performance on AMD GPUs on Windows, install the [ROCm PyTorch version](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html):

```bash
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz

pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl 
```

## Workflow

All scripts use a centralized workflow directory containing a `config.json` file with processing parameters. This eliminates the need to specify paths and settings for each command.

Two approaches are available:

1. **Manual Workflow** - Process a single video step-by-step. Recommended for initial setup to find optimal stereo parameters using the interactive tester.
2. **Batch Processing** - Automate multiple videos using the workflow orchestrator. Use this once you have established settings that work well for your content.

## Batch Processing

The workflow orchestrator automates multi-video processing by managing all pipeline steps.

### Step 1: Initialize Workflows

Create a workflow directory for each video:

```bash
python workflow_init.py --input-video "C:/Videos/video1.mkv"
python workflow_init.py --input-video "C:/Videos/video2.mkv"
python workflow_init.py --input-video "C:/Videos/video3.mkv"
```

### Step 2: Create Workflows File

Create a `workflows.yaml` file listing all workflow directories:

```yaml
C:/Videos/video1/workflow:
C:/Videos/video2/workflow:
C:/Videos/video3/workflow:
```

Each workflow tracks five processing steps:
- `frame_extractor` - Extract frames from video
- `depth_map_generator` - Generate AI depth maps
- `sbs_generator` - Create stereoscopic images
- `chunk_generator` - Encode video segments
- `video_concatenator` - Merge chunks with audio

Step status values: `PENDING`, `RUNNING`, `DONE`, `FAILED`, `ERROR`

### Step 3: Run the Orchestrator

```bash
python workflow_orchestrator.py workflows.yaml
```

Optional parameters:
- `--validate-only` - Validate workflows without running

The orchestrator provides a live dashboard showing:
- Active processes and their progress
- Completed and pending workflows
- Estimated time remaining

Key orchestrator features:
- Runs depth map and SBS generation in parallel when resources allow
- Automatically resumes from where it left off
- Handles disk space management between steps
- Graceful shutdown on Ctrl+C (saves state for resume)

## Manual Workflow

For processing a single video manually, run each step individually:

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
├── config.json
├── frames/
├── depth_maps/
├── sbs/
└── chunks/
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
- `--start-frame N` - First frame to process
- `--end-frame N` - Last frame to process
- `--no-interactive` - Exit on error instead of waiting for user input (used by orchestrator)

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
- `--no-interactive` - Exit on error instead of waiting for user input (used by orchestrator)

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
        "convergence": -10.0,
        "super_sampling": 3.0,
        "edge_softness": 20.0,
        "artifact_smoothing": 1.0,
        "depth_gamma": 0.2,
        "sharpen": 14.0
    },
    "depth": {
        "save_16bit": false
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
- `directories.frames` - Subdirectory for extracted video frames
- `directories.depth_maps` - Subdirectory for generated depth maps
- `directories.sbs` - Subdirectory for side-by-side stereo images
- `directories.chunks` - Subdirectory for encoded video chunks

**Stereo Parameters:**
- `max_disparity` - Maximum pixel displacement (recommended: 14-60)
- `convergence` - Focal plane shift (positive = pop-out, negative = sink-in)
- `super_sampling` - Internal upscale factor (recommended: 1.0-3.0)
- `edge_softness` - Depth edge softening strength (recommended: 0-30)
- `artifact_smoothing` - Warping artifact smoothing (recommended: 0-5)
- `depth_gamma` - Depth gamma correction (recommended: 0.2-0.5)
- `sharpen` - Unsharp mask strength (recommended: 0.5-16.0)

**Depth Generation:**
- `save_16bit` - Save depth maps as 16-bit TIFF (slightly reduces artifacts, about 10 times larger files)

**Video Encoding:**
- `crf` - Constant Rate Factor for libx265 (lower = higher quality, default: 19)
- `preset` - FFmpeg encoding preset (slower = better compression, default: slow)

**Disk Space Management:**
- `free_space.sbs_generator` - Delete source images after successful SBS image generation:
  - `"frame"` - Delete only frame files (default)
  - `"depth"` - Delete only depth map files
  - `"all"` - Delete both frame and depth files
  - `"none"` - Keep all files
- `free_space.chunk_generator` - Delete source images after successful video chunk encoding:
  - `"sbs"` - Delete SBS files after a chunk is encoded (default)
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

### Performance

Processing speed depends heavily on GPU capabilities. Help improve this documentation by sharing your benchmark results:

- Depth map generation speed (images/second)
- SBS generation speed (images/second)
- System specs: CPU, RAM, GPU model, VRAM, OS, video resolution

Send your results to github@duttke.de

### Disk Space

The conversion pipeline creates large amounts of intermediate data (~5 GB per 1000 frames at 1080p). Plan accordingly and consider processing in chunks. The workflow orchestrator handles disk space management automatically between steps.

### Resumable Processing

All scripts automatically skip already-processed files, making it safe to interrupt and resume processing at any time.

### Script Reference

| Script | Description |
|--------|-------------|
| [`workflow_init.py`](./workflow_init.py) | Create new workflow with default config |
| [`workflow_orchestrator.py`](./workflow_orchestrator.py) | Automate multi-video batch processing |
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

## License Notice

The depth-estimation model used by this software is licensed under the Apple Machine Learning Research Model License.  
Use of the model is restricted to non-commercial scientific research and academic purposes only.  
Using the model in applications for video conversion, entertainment, or other non-research purposes may violate the license.  
Users are responsible for complying with these terms.
