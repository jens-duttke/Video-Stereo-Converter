#!/usr/bin/env python3
"""
Frame Extractor
===============

Extracts frames from a video file using FFmpeg.
Reads input video path from workflow config.json.

Usage:
    python frame_extractor.py "D:/Video-Processing/workflow"
"""

from __future__ import annotations

import helper.utf8_console  # noqa: F401 # pyright: ignore[reportUnusedImport]
import helper.terminal_title  # noqa: F401 # pyright: ignore[reportUnusedImport]

import argparse
import re
import subprocess
import sys
import threading
from pathlib import Path

from tqdm import tqdm

from helper.config_manager import ConfigError, get_path, load_config
from helper.ffmpeg_utils import estimate_frame_count


def extract_frames(workflow_path: Path, config: dict) -> bool:
    """
    Extract frames from input video using FFmpeg.

    Parameters
    ----------
    workflow_path : Path
        Path to the workflow directory.
    config : dict
        Workflow configuration.

    Returns
    -------
    bool
        True on success, False on error.
    """
    input_video = get_path(workflow_path, config, 'input_video')
    frames_dir = get_path(workflow_path, config, 'frames')

    if not input_video.is_file():
        print(f'ERROR: Input video not found: {input_video}')
        return False

    # Check if frames already exist
    existing_frames = list(frames_dir.glob('frame_*.png'))
    if existing_frames:
        print(f'WARNING: {len(existing_frames)} frames already exist in {frames_dir}')
        response = input('Continue and overwrite? [y/N]: ').strip().lower()
        if response != 'y':
            print('Aborted.')
            return False

    # Try to get frame count for progress bar
    print(f'Analyzing video: {input_video.name}')

    # Use estimation (fast) - tqdm will adjust if actual count differs
    frame_count = estimate_frame_count(input_video)
    if frame_count:
        print(f'Estimated frames: {frame_count}')
    else:
        print('Could not determine frame count, progress will be estimated.')
        frame_count = 0

    # Build FFmpeg command
    output_pattern = frames_dir / 'frame_%06d.png'
    cmd = [
        'ffmpeg',
        '-y',
        '-i', str(input_video),
        '-an',
        '-progress', 'pipe:1',
        '-nostats',
        str(output_pattern)
    ]

    print()
    print(f'Extracting frames to: {frames_dir}')
    print()

    # Run FFmpeg with progress tracking
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    # Read stderr in background thread to avoid deadlock (buffer full while reading stdout)
    stderr_output: list[str] = []
    stderr_thread = threading.Thread(target=lambda: stderr_output.append(process.stderr.read()), daemon=True)
    stderr_thread.start()

    frame_pattern = re.compile(r'^frame=(\d+)')

    with tqdm(total=frame_count if frame_count > 0 else None, unit='frame', bar_format='{l_bar}{bar}| {n_fmt}/~{total_fmt}{postfix} [{elapsed}<{remaining}, {rate_noinv_fmt}]', mininterval=0.5) as pbar:
        current_frame = 0

        for line in process.stdout:
            match = frame_pattern.match(line.strip())
            if match:
                new_frame = int(match.group(1))
                if new_frame > current_frame:
                    # Dynamically extend progress bar if actual count exceeds estimate
                    if pbar.total and new_frame > pbar.total:
                        pbar.total = new_frame
                        pbar.refresh()
                    pbar.update(new_frame - current_frame)
                    current_frame = new_frame
                    pbar.set_postfix_str(f'Frame: {current_frame}')

    process.wait()
    stderr_thread.join()

    if process.returncode != 0:
        stderr = stderr_output[0] if stderr_output else ''
        print(f'ERROR: FFmpeg failed!')
        print(f'stderr: {stderr[-1000:]}')
        return False

    # Count extracted frames
    extracted = list(frames_dir.glob('frame_*.png'))
    print()
    print(f'Extracted {len(extracted)} frames successfully.')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from video using FFmpeg',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python frame_extractor.py "D:/Video-Processing/workflow"\n'
            '\n'
            'Reads the input video path from config.json and extracts frames to the frames/ directory.\n'
        )
    )
    parser.add_argument('workflow_path', type=Path, help='Path to the workflow directory containing config.json')

    args = parser.parse_args()

    # Validate workflow directory
    if not args.workflow_path.is_dir():
        print(f'ERROR: Workflow directory does not exist: {args.workflow_path}')
        sys.exit(1)

    # Load config
    try:
        config = load_config(args.workflow_path)
    except ConfigError as e:
        print(f'ERROR: {e}')
        sys.exit(1)

    # Extract frames
    success = extract_frames(args.workflow_path, config)

    if not success:
        sys.exit(1)

    print()
    print('Done!')


if __name__ == '__main__':
    main()
