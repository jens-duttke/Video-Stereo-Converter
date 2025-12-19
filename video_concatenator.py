#!/usr/bin/env python3
"""
Video Concatenator
==================

Concatenates sequential SBS video chunks and adds audio from input video.
Reads paths and settings from workflow config.json.

Usage:
    python video_concatenator.py "D:/Video-Processing/workflow"
"""

import re
import subprocess
import sys
import tempfile
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from helper.config_manager import ConfigError, get_path, load_config
from helper.ffmpeg_utils import get_video_duration, get_video_framerate, parse_framerate


def _find_video_chunks(input_path: Path) -> List[Tuple[int, int, Path]]:
    """
    Find all sbs_XXXXXX_YYYYYY.mkv files and return sorted list.

    Parameters
    ----------
    input_path : Path
        Directory containing video chunks.

    Returns
    -------
    list of tuple
        List of (start_frame, end_frame, path) tuples, sorted by start_frame.
    """
    pattern = re.compile(r'^sbs_(\d+)_(\d+)\.mkv$')
    videos = []

    for file in input_path.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                start_frame = int(match.group(1))
                end_frame = int(match.group(2))
                videos.append((start_frame, end_frame, file))

    return sorted(videos, key=lambda x: x[0])


def _validate_sequence(videos: List[Tuple[int, int, Path]]) -> Tuple[bool, bool]:
    """
    Validate that videos form a continuous sequence.

    Parameters
    ----------
    videos : list
        List of video chunk tuples.

    Returns
    -------
    tuple of bool
        (is_valid, is_overlapping)
    """
    if len(videos) < 2:
        return True, False

    is_overlapping = None

    for i in range(len(videos) - 1):
        current_end = videos[i][1]
        next_start = videos[i + 1][0]

        if current_end == next_start:
            current_overlapping = True
        elif current_end + 1 == next_start:
            current_overlapping = False
        else:
            print('ERROR: Gap or invalid sequence detected!')
            print(f'  {videos[i][2].name} ends at frame {current_end}')
            print(f'  {videos[i + 1][2].name} starts at frame {next_start}')
            return False, False

        if is_overlapping is None:
            is_overlapping = current_overlapping
        elif is_overlapping != current_overlapping:
            print('ERROR: Mixed overlapping/non-overlapping sequences detected!')
            print(f'  Between {videos[i][2].name} and {videos[i + 1][2].name}')
            return False, False

    return True, is_overlapping if is_overlapping is not None else False


def _validate_framerates(videos: List[Tuple[int, int, Path]]) -> Tuple[bool, str | None]:
    """
    Validate that all videos have the same framerate.

    Parameters
    ----------
    videos : list
        List of video chunk tuples.

    Returns
    -------
    tuple
        (is_valid, framerate_string)
    """
    if not videos:
        return False, None

    first_framerate = get_video_framerate(videos[0][2])
    if not first_framerate:
        print(f'ERROR: Could not detect framerate of {videos[0][2].name}')
        return False, None

    for start, end, path in videos[1:]:
        framerate = get_video_framerate(path)
        if framerate != first_framerate:
            print('ERROR: Framerate mismatch!')
            print(f'  {videos[0][2].name}: {first_framerate}')
            print(f'  {path.name}: {framerate}')
            return False, None

    return True, first_framerate


def _concatenate_videos(
    videos: List[Tuple[int, int, Path]],
    output_path: Path,
    is_overlapping: bool,
    framerate_str: str,
    audio_source: Path | None = None
) -> bool:
    """
    Concatenate videos using ffmpeg concat demuxer.

    Parameters
    ----------
    videos : list
        List of video chunk tuples.
    output_path : Path
        Output video file path.
    is_overlapping : bool
        Whether videos have overlapping frames.
    framerate_str : str
        Framerate string for calculating frame duration.
    audio_source : Path or None
        Audio source file path.

    Returns
    -------
    bool
        True on success, False on error.
    """
    if not videos:
        print('ERROR: No videos to concatenate!')
        return False

    # Calculate frame duration for accurate frame skipping
    if is_overlapping:
        framerate = parse_framerate(framerate_str)
        frame_duration = 1.0 / framerate if framerate else 0.001
    else:
        frame_duration = 0.001

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create concat file with proper inpoints for overlapping videos
        concat_file = temp_path / 'concat.txt'
        with open(concat_file, 'w', encoding='utf-8') as f:
            for i, (start, end, path) in enumerate(videos):
                escaped_path = str(path.absolute()).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

                if is_overlapping and i > 0:
                    f.write(f'inpoint {frame_duration:.6f}\n')

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file)
        ]

        if audio_source:
            cmd.extend(['-i', str(audio_source)])

        cmd.extend(['-map', '0:v'])

        if audio_source:
            cmd.extend(['-map', '1:a?'])
            cmd.extend(['-c:a', 'copy'])
            cmd.extend(['-shortest'])

        cmd.extend(['-c:v', 'copy'])
        cmd.append(str(output_path))

        print(f'Concatenating {len(videos)} video(s)...')
        if is_overlapping:
            print('  Mode: Overlapping (skipping duplicate frames)')
        else:
            print('  Mode: Non-overlapping')

        if audio_source:
            print(f'  Audio source: {audio_source.name}')

        print(f'  Output: {output_path.name}')
        print()

        # Calculate total duration for progress bar
        total_duration = sum(get_video_duration(path) or 0.0 for _, _, path in videos)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        stderr_output = []
        current_time = 0.0

        with tqdm(
            total=total_duration,
            unit='s',
            bar_format='  {l_bar}{bar}| {n:.1f}s/{total:.1f}s [{elapsed}<{remaining}]'
        ) as pbar:
            for line in process.stderr:
                stderr_output.append(line)
                time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
                if time_match:
                    hours = int(time_match.group(1))
                    minutes = int(time_match.group(2))
                    seconds = float(time_match.group(3))
                    new_time = hours * 3600 + minutes * 60 + seconds
                    if new_time > current_time:
                        pbar.update(new_time - current_time)
                        current_time = new_time

        process.wait()

        if process.returncode != 0:
            print('ERROR: ffmpeg failed!')
            error_text = ''.join(stderr_output[-20:])
            print(f'stderr: {error_text[-1000:]}')
            return False

        if not output_path.exists() or output_path.stat().st_size == 0:
            print('ERROR: Output file was not created or is empty!')
            return False

        print(f'Video created: {output_path.stat().st_size / (1024*1024):.1f} MB')
        return True


def main():
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='Video Concatenator - Concatenate SBS video chunks',
        epilog=(
            'Example:\n'
            '  python video_concatenator.py "D:/Video-Processing/workflow"\n'
        )
    )

    parser.add_argument('workflow_path', type=Path, help='Path to workflow directory containing config.json')

    args = parser.parse_args()

    # Validate workflow directory
    if not args.workflow_path.is_dir():
        print(f'ERROR: Workflow directory not found: {args.workflow_path}')
        sys.exit(1)

    # Load config
    try:
        config = load_config(args.workflow_path)
    except ConfigError as e:
        print(f'ERROR: {e}')
        sys.exit(1)

    # Get paths from config
    chunks_dir = get_path(args.workflow_path, config, 'chunks')
    output_path = get_path(args.workflow_path, config, 'output_video')
    input_video = get_path(args.workflow_path, config, 'input_video')

    if not chunks_dir.is_dir():
        print(f'ERROR: Chunks directory does not exist: {chunks_dir}')
        sys.exit(1)

    # Use input video as audio source
    audio_source = input_video if input_video.is_file() else None
    if not audio_source:
        print(f'WARNING: Input video not found, output will have no audio: {input_video}')

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find video chunks
    videos = _find_video_chunks(chunks_dir)

    if not videos:
        print(f'ERROR: No sbs_XXXXXX_YYYYYY.mkv files found in {chunks_dir}')
        sys.exit(1)

    print(f'Found {len(videos)} video chunk(s):')
    for start, end, path in videos:
        print(f'  - {path.name} (frames {start} to {end})')
    print()

    # Validate sequence
    is_valid, is_overlapping = _validate_sequence(videos)
    if not is_valid:
        sys.exit(1)

    # Validate framerates
    is_valid_fps, framerate = _validate_framerates(videos)
    if not is_valid_fps:
        sys.exit(1)

    print(f'Framerate: {framerate}')
    print()

    # Concatenate
    success = _concatenate_videos(videos, output_path, is_overlapping, framerate, audio_source)

    if not success:
        sys.exit(1)

    print()
    print(f'Done! Output: {output_path}')


if __name__ == '__main__':
    main()
