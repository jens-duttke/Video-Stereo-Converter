#!/usr/bin/env python3
"""
Chunk Generator
===============

Creates video chunks from SBS PNG sequences.
Reads encoding settings and input video for framerate from workflow config.json.

NOTE: Resuming is not supported here. If interrupted, the whole chunk must be re-encoded again.
Use the --end-frame parameter to process in smaller chunks if needed.

Usage:
    python chunk_generator.py "D:/Video-Processing/workflow"
    python chunk_generator.py "D:/Video-Processing/workflow" --end-frame 5000
"""

from __future__ import annotations

import helper.utf8_console  # noqa: F401 # pyright: ignore[reportUnusedImport]
import helper.terminal_title  # noqa: F401 # pyright: ignore[reportUnusedImport]

import re
import subprocess
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from helper.config_manager import ConfigError, get_path, load_config
from helper.ffmpeg_utils import get_video_framerate


def _find_sbs_files(input_path: Path) -> List[Tuple[int, Path]]:
    """
    Find all sbs_*.png files and extract frame numbers.

    Parameters
    ----------
    input_path : Path
        Directory to search for SBS files.

    Returns
    -------
    list of tuple
        List of (frame_number, filepath) tuples, sorted by frame_number.
    """
    pattern = re.compile(r'sbs_(\d+)\.png$')
    files = []

    for file in input_path.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                frame_num = int(match.group(1))
                files.append((frame_num, file))

    files.sort(key=lambda x: x[0])
    return files


def _find_existing_videos(input_path: Path) -> List[Tuple[int, int, Path]]:
    """
    Find all existing sbs_XXXXXX_YYYYYY.mkv files and extract frame ranges.

    Parameters
    ----------
    input_path : Path
        Directory to search for video files.

    Returns
    -------
    list of tuple
        List of (start_frame, end_frame, filepath) tuples, sorted by start_frame.
    """
    pattern = re.compile(r'sbs_(\d+)_(\d+)\.mkv$')
    videos = []

    for file in input_path.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                start_frame = int(match.group(1))
                end_frame = int(match.group(2))
                videos.append((start_frame, end_frame, file))

    videos.sort(key=lambda x: x[0])
    return videos


def _get_processed_frame_range(videos: List[Tuple[int, int, Path]]) -> int | None:
    """
    Determine the highest end frame from existing videos.

    Parameters
    ----------
    videos : list
        List of existing video chunks.

    Returns
    -------
    int or None
        The highest end frame number, or None if no videos exist.
    """
    if not videos:
        return None
    return max(v[1] for v in videos)


def _filter_unprocessed_frames(frames: List[Tuple[int, Path]], last_processed_frame: int | None, end_frame: int | None = None) -> List[Tuple[int, Path]]:
    """
    Filter out frames that are already processed.

    Parameters
    ----------
    frames : list
        All available frames.
    last_processed_frame : int or None
        The last frame number that was already processed.
    end_frame : int or None
        Maximum frame number to process (inclusive).

    Returns
    -------
    list
        List of frames starting from last_processed_frame (inclusive) up to end_frame.
    """
    if last_processed_frame is None:
        result = frames
    else:
        result = [(num, path) for num, path in frames if num >= last_processed_frame]

    if end_frame is not None:
        result = [(num, path) for num, path in result if num <= end_frame]

    return result


def _validate_frame_sequence(frames: List[Tuple[int, Path]]) -> bool:
    """
    Check if all frames are consecutively numbered without gaps.

    Parameters
    ----------
    frames : list
        List of frames to validate.

    Returns
    -------
    bool
        True if valid, False if gaps exist.
    """
    if not frames:
        print('ERROR: No sbs_*.png files found!')
        return False

    frame_numbers = [f[0] for f in frames]
    expected_start = frame_numbers[0]
    expected_end = frame_numbers[-1]
    expected_count = expected_end - expected_start + 1

    if len(frame_numbers) != expected_count:
        print('ERROR: Frame sequence has gaps!')
        print(f'  Expected: {expected_count} frames ({expected_start} to {expected_end})')
        print(f'  Found: {len(frame_numbers)} frames')

        frame_set = set(frame_numbers)
        missing = [i for i in range(expected_start, expected_end + 1) if i not in frame_set]

        if missing:
            print(f'  Missing frames: {missing[:10]}' +
                  (f' ... and {len(missing)-10} more' if len(missing) > 10 else ''))

        return False

    print(f'Frame sequence validated: {len(frames)} frames ({expected_start} to {expected_end})')
    return True


def _parse_ffmpeg_frame(line: str) -> int | None:
    """Extract current frame number from ffmpeg output line."""
    match = re.search(r'frame=\s*(\d+)', line)
    if match:
        return int(match.group(1))
    return None


def _cleanup_partial_video(output_path: Path) -> None:
    """
    Delete partial/incomplete video file if it exists.

    Parameters
    ----------
    output_path : Path
        Path to the video file to delete.
    """
    if output_path.exists():
        try:
            output_path.unlink()
            print(f'\nCleaned up partial video file: {output_path.name}')
        except OSError as e:
            print(f'\nWarning: Could not delete partial video file: {e}')


def _create_video_clip(frames: List[Tuple[int, Path]], output_path: Path, framerate: str, crf: int, preset: str) -> bool:
    """
    Create a video from a frame sequence with progress bar.

    Uses atomic write pattern: writes to .tmp file first, then renames on success.
    This ensures incomplete chunks are never mistaken for complete ones.

    Parameters
    ----------
    frames : list
        List of frames to encode.
    output_path : Path
        Output video file path.
    framerate : str
        Video framerate string.
    crf : int
        CRF value for libx265.
    preset : str
        FFmpeg preset.

    Returns
    -------
    bool
        True on success, False on error.
    """
    if not frames:
        return False

    total_frames = len(frames)
    start_frame_num = frames[0][0]
    input_dir = frames[0][1].parent

    # Use temporary file for atomic write
    temp_path = output_path.with_suffix('.mkv.tmp')

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(framerate),
        '-start_number', str(start_frame_num),
        '-i', str(input_dir / 'sbs_%06d.png'),
        '-frames:v', str(total_frames),
        '-c:v', 'libx265',
        '-preset', preset,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p10le',
        '-f', 'matroska',  # Explicit format for .tmp extension
        str(temp_path)  # Write to temp file
    ]

    print(f'  Creating video: {output_path.name}')
    print(f'  Frames: {total_frames}, Framerate: {framerate}, CRF: {crf}, Preset: {preset}')

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        stderr_output = []
        current_frame = 0

        with tqdm(total=total_frames, unit='frame', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]', mininterval=0.5) as pbar:
            for line in process.stderr:
                stderr_output.append(line)
                frame = _parse_ffmpeg_frame(line)
                if frame is not None and frame != current_frame:
                    pbar.update(frame - current_frame)
                    current_frame = frame

        process.wait()

        if process.returncode != 0:
            print('ERROR: ffmpeg failed!')
            error_text = ''.join(stderr_output[-20:])
            print(f'stderr: {error_text[-500:]}')
            _cleanup_partial_video(temp_path)
            return False

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            print('ERROR: Video file was not created or is empty!')
            _cleanup_partial_video(temp_path)
            return False

        # Atomic rename: temp file -> final file
        temp_path.rename(output_path)

        print(f'  Video created: {output_path.stat().st_size / (1024*1024):.1f} MB')
        return True

    except KeyboardInterrupt:
        print('\n\nEncoding interrupted by user!')
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        _cleanup_partial_video(temp_path)
        raise  # Re-raise to allow main() to handle it

    except Exception as e:
        print(f'\nERROR: Unexpected error during encoding: {e}')
        if process:
            process.terminate()
        _cleanup_partial_video(temp_path)
        return False


def main():
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='Chunk Generator - Create video chunks from SBS images',
        epilog=(
            'Note: If interrupted, the whole chunk must be re-encode again.\n'
            'Use --end-frame to process in smaller chunks if needed.\n'
            '\n'
            'Example:\n'
            '  python chunk_generator.py "D:/Video-Processing/workflow"\n'
            '  python chunk_generator.py "D:/Video-Processing/workflow" --end-frame 5000\n'
        )
    )

    parser.add_argument('workflow_path', type=Path, help='Path to workflow directory containing config.json')
    parser.add_argument('--end-frame', type=int, default=None, help='Process frames up to this frame number (inclusive)')

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
    sbs_dir = get_path(args.workflow_path, config, 'sbs')
    chunks_dir = get_path(args.workflow_path, config, 'chunks')
    input_video = get_path(args.workflow_path, config, 'input_video')

    if not sbs_dir.exists():
        print(f'ERROR: SBS directory not found: {sbs_dir}')
        sys.exit(1)

    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Get encoding settings from config
    crf = config['encoding']['crf']
    preset = config['encoding']['preset']

    # Free space config
    free_space_mode = config.get('free_space', {}).get('chunk_generator', 'none')
    if free_space_mode == 'sbs':
        print('Free space mode: deleting SBS files after encoding')

    # Get framerate from input video
    if not input_video.is_file():
        print(f'ERROR: Input video not found: {input_video}')
        sys.exit(1)

    framerate = get_video_framerate(input_video)
    if not framerate:
        print(f'ERROR: Could not detect framerate from input video: {input_video}')
        sys.exit(1)

    print(f'Detected framerate from input video: {framerate}')
    print(f'Encoding: CRF={crf}, Preset={preset}')
    print()

    # Find existing videos in chunks directory
    existing_videos = _find_existing_videos(chunks_dir)
    last_processed = _get_processed_frame_range(existing_videos)

    if existing_videos:
        print(f'Found {len(existing_videos)} existing video(s):')
        for start, end, path in existing_videos:
            print(f'  - {path.name} (frames {start} to {end})')
        print(f'  Resuming from frame {last_processed}')
        print()

    # Find all SBS frames
    all_frames = _find_sbs_files(sbs_dir)
    if not all_frames:
        print('ERROR: No sbs_*.png files found!')
        sys.exit(1)

    # Validate end_frame if specified
    if args.end_frame is not None:
        min_frame = all_frames[0][0]
        max_frame = all_frames[-1][0]

        if args.end_frame < min_frame:
            print(f'ERROR: --end-frame {args.end_frame} is lower than the smallest available frame!')
            print(f'  Available frame range: {min_frame} to {max_frame}')
            sys.exit(1)

        if args.end_frame > max_frame:
            print(f'ERROR: --end-frame {args.end_frame} is higher than the largest available frame!')
            print(f'  Available frame range: {min_frame} to {max_frame}')
            sys.exit(1)

    # Filter to unprocessed frames only
    frames_to_process = _filter_unprocessed_frames(all_frames, last_processed, args.end_frame)

    if not frames_to_process:
        print('All frames have already been processed. Nothing to do.')
        sys.exit(0)

    if len(frames_to_process) < 2:
        print(f'Only {len(frames_to_process)} frame(s) available. Need at least 2 frames to create a video.')
        sys.exit(0)

    # Validate the frame sequence
    if not _validate_frame_sequence(frames_to_process):
        sys.exit(1)

    print()

    # Create video
    start_frame_num = frames_to_process[0][0]
    end_frame_num = frames_to_process[-1][0]

    output_name = f'sbs_{start_frame_num:06d}_{end_frame_num:06d}.mkv'
    output_path = chunks_dir / output_name

    print(f'Frame range: {start_frame_num} - {end_frame_num} ({len(frames_to_process)} frames)')
    print()
    print('\033[36mNote: If interrupted, the whole chunk must be re-encoded chunk again.\033[0m')
    print()

    try:
        success = _create_video_clip(frames_to_process, output_path, framerate, crf, preset)
    except KeyboardInterrupt:
        print('\nOperation cancelled by user.')
        sys.exit(1)

    if success:
        print(f'Done! Video created: {output_name}')
        # Delete SBS files if free_space mode is 'sbs'
        if free_space_mode == 'sbs':
            # Keep the last frame for the next chunk (to maintain overlap)
            frames_to_delete = frames_to_process[:-1] if len(frames_to_process) > 1 else []
            if frames_to_delete:
                deleted_count = 0
                for _, sbs_path in tqdm(frames_to_delete, desc='Deleting SBS files', unit='file', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]', mininterval=0.5):
                    try:
                        sbs_path.unlink(missing_ok=True)
                        deleted_count += 1
                    except OSError:
                        pass  # Ignore deletion errors
                print(f'Deleted {deleted_count} SBS files to free space (kept last frame for next chunk).')
    else:
        print('ERROR: Video creation failed!')
        sys.exit(1)


if __name__ == '__main__':
    main()
