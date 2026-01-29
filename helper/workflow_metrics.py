"""
Workflow Metrics
================

File counting and frame number tracking for workflow orchestration.
Provides cached metrics to minimize filesystem access.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from helper.config_manager import get_path, load_config
from helper.frame_utils import extract_frame_number


__all__ = [
    'CHUNK_SIZE',
    'MIN_DEPTH_FOR_SBS',
    'DISK_SPACE_THRESHOLD_GB',
    'get_depth_count',
    'get_max_depth_number',
    'get_max_sbs_number',
    'get_last_chunk_end_frame',
    'get_total_frame_count',
    'get_video_progress',
    'is_all_chunks_complete',
    'get_next_chunk_end_frame',
    'invalidate_cache',
]


# Constants
CHUNK_SIZE = 1500
MIN_DEPTH_FOR_SBS = 1000
DISK_SPACE_THRESHOLD_GB = 10


def invalidate_cache() -> None:
    """Clear all cached metrics."""
    _get_file_count_cached.cache_clear()
    _get_max_frame_cached.cache_clear()
    _get_chunk_info_cached.cache_clear()


@lru_cache(maxsize=256)
def _get_file_count_cached(directory: str, pattern: str) -> int:
    """
    Count files matching pattern in directory.

    Parameters
    ----------
    directory : str
        Directory path as string (for caching).
    pattern : str
        Glob pattern to match.

    Returns
    -------
    int
        Number of matching files.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return 0

    return sum(1 for _ in dir_path.glob(pattern))


@lru_cache(maxsize=256)
def _get_max_frame_cached(directory: str, pattern: str) -> int:
    """
    Get maximum frame number from files matching pattern.

    Parameters
    ----------
    directory : str
        Directory path as string (for caching).
    pattern : str
        Glob pattern to match.

    Returns
    -------
    int
        Maximum frame number, or 0 if no files found.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return 0

    max_frame = 0
    for f in dir_path.glob(pattern):
        frame_num = extract_frame_number(str(f))
        if frame_num > max_frame:
            max_frame = frame_num

    return max_frame


def _cleanup_incomplete_chunks(chunks_dir: Path) -> None:
    """
    Delete incomplete chunk files (.mkv.tmp).

    These are left behind when chunk_generator is interrupted during encoding.

    Parameters
    ----------
    chunks_dir : Path
        Directory containing chunk files.
    """
    for tmp_file in chunks_dir.glob('sbs_*.mkv.tmp'):
        try:
            tmp_file.unlink()
        except OSError:
            pass  # Ignore deletion errors


@lru_cache(maxsize=128)
def _get_chunk_info_cached(chunks_dir: str) -> tuple[int, int]:
    """
    Get chunk coverage information.

    Also cleans up any incomplete .tmp chunk files before scanning.

    Parameters
    ----------
    chunks_dir : str
        Chunks directory path as string (for caching).

    Returns
    -------
    tuple
        (last_end_frame, total_chunks) tuple.
    """
    dir_path = Path(chunks_dir)
    if not dir_path.exists():
        return 0, 0

    # Clean up incomplete chunks before scanning
    _cleanup_incomplete_chunks(dir_path)

    pattern = re.compile(r'sbs_(\d+)_(\d+)\.mkv$')
    last_end = 0
    count = 0

    for f in dir_path.iterdir():
        if f.is_file():
            match = pattern.match(f.name)
            if match:
                end_frame = int(match.group(2))
                if end_frame > last_end:
                    last_end = end_frame
                count += 1

    return last_end, count


def get_depth_count(workflow_path: Path) -> int:
    """
    Count depth maps in workflow.

    Parameters
    ----------
    workflow_path : Path
        Path to workflow directory.

    Returns
    -------
    int
        Number of depth map files.
    """
    try:
        config = load_config(workflow_path)
        depth_dir = get_path(workflow_path, config, 'depth_maps')
        tif_count = _get_file_count_cached(str(depth_dir), 'depth_frame_*.tif')
        png_count = _get_file_count_cached(str(depth_dir), 'depth_frame_*.png')
        return tif_count + png_count
    except Exception:
        return 0


def get_max_depth_number(workflow_path: Path) -> int:
    """
    Get highest frame number from depth maps.

    Parameters
    ----------
    workflow_path : Path
        Path to workflow directory.

    Returns
    -------
    int
        Maximum depth frame number, or 0 if none found.
    """
    try:
        config = load_config(workflow_path)
        depth_dir = get_path(workflow_path, config, 'depth_maps')
        tif_max = _get_max_frame_cached(str(depth_dir), 'depth_frame_*.tif')
        png_max = _get_max_frame_cached(str(depth_dir), 'depth_frame_*.png')
        return max(tif_max, png_max)
    except Exception:
        return 0


def get_max_sbs_number(workflow_path: Path) -> int:
    """
    Get highest frame number from SBS images.

    Parameters
    ----------
    workflow_path : Path
        Path to workflow directory.

    Returns
    -------
    int
        Maximum SBS frame number, or 0 if none found.
    """
    try:
        config = load_config(workflow_path)
        sbs_dir = get_path(workflow_path, config, 'sbs')
        return _get_max_frame_cached(str(sbs_dir), 'sbs_*.png')
    except Exception:
        return 0


def get_last_chunk_end_frame(workflow_path: Path) -> int:
    """
    Get the last frame number covered by existing chunks.

    Parameters
    ----------
    workflow_path : Path
        Path to workflow directory.

    Returns
    -------
    int
        Last end frame from chunks, or 0 if none exist.
    """
    try:
        config = load_config(workflow_path)
        chunks_dir = get_path(workflow_path, config, 'chunks')
        last_end, _ = _get_chunk_info_cached(str(chunks_dir))
        return last_end
    except Exception:
        return 0


def get_total_frame_count(workflow_path: Path) -> int:
    """
    Get total frame count from input video via ffmpeg.

    Parameters
    ----------
    workflow_path : Path
        Path to workflow directory.

    Returns
    -------
    int
        Estimated total frame count, or 0 if detection fails.
    """
    try:
        from helper.ffmpeg_utils import estimate_frame_count
        config = load_config(workflow_path)
        input_video = get_path(workflow_path, config, 'input_video')
        return estimate_frame_count(input_video) or 0
    except Exception:
        return 0


def get_next_chunk_end_frame(workflow_path: Path, last_chunk_end: int, sbs_complete: bool = False) -> int | None:
    """
    Calculate next chunk end frame if enough SBS files exist.

    For intermediate chunks (during processing): Creates a chunk when enough frames
    are available. If the remaining frames after a chunk would be less than or equal
    to CHUNK_SIZE, the chunk is extended to include all remaining frames to avoid
    creating a small final chunk.

    For final chunks (when SBS generation is complete): Always creates a chunk if
    there are enough frames (at least 2), regardless of chunk size.

    Parameters
    ----------
    workflow_path : Path
        Path to workflow directory.
    last_chunk_end : int
        Last processed chunk end frame.
    sbs_complete : bool
        Whether SBS generation is complete. When True, creates final chunk regardless
        of size. When False, only creates intermediate chunks if remaining > CHUNK_SIZE.

    Returns
    -------
    int or None
        Next chunk end frame, or None if not enough SBS files.
    """
    max_sbs = get_max_sbs_number(workflow_path)

    if last_chunk_end == 0:
        potential_chunk_end = CHUNK_SIZE
    else:
        potential_chunk_end = last_chunk_end + CHUNK_SIZE

    # Check if we have enough frames for a full chunk
    if max_sbs >= potential_chunk_end:
        remaining_after_chunk = max_sbs - potential_chunk_end

        if sbs_complete:
            # Final processing: extend to include all remaining frames if they fit
            if remaining_after_chunk <= CHUNK_SIZE:
                return max_sbs
            return potential_chunk_end
        elif remaining_after_chunk > CHUNK_SIZE:
            # Intermediate chunk: enough frames remain for another full chunk
            return potential_chunk_end
        elif remaining_after_chunk > 0:
            # Remaining frames <= CHUNK_SIZE: extend this chunk to include them all
            return max_sbs

    # Check for final chunk when SBS is complete
    if sbs_complete:
        start_frame = last_chunk_end if last_chunk_end > 0 else 0
        remaining_frames = max_sbs - start_frame

        # ffmpeg requires at least 2 frames to create a video
        if remaining_frames >= 2:
            return max_sbs

    return None


def is_all_chunks_complete(workflow_path: Path) -> bool:
    """
    Check if all frames have been encoded into chunks.

    Uses max_sbs when SBS files exist, falls back to max_depth or total_frames
    when SBS files have been deleted (free_space_mode == 'sbs').

    Parameters
    ----------
    workflow_path : Path
        Path to workflow directory.

    Returns
    -------
    bool
        True if all chunks are complete.
    """
    last_chunk = get_last_chunk_end_frame(workflow_path)
    if last_chunk == 0:
        return False

    max_sbs = get_max_sbs_number(workflow_path)
    if max_sbs > 0:
        # SBS files exist, compare against them
        return last_chunk >= max_sbs

    # SBS files may have been deleted, use alternative reference
    max_depth = get_max_depth_number(workflow_path)
    if max_depth > 0:
        return last_chunk >= max_depth

    # Fallback to total frame count from video
    total_frames = get_total_frame_count(workflow_path)
    if total_frames > 0:
        return last_chunk >= total_frames

    return False


def get_video_progress(workflow_path: Path) -> str:
    """
    Get video encoding progress as display string.

    Returns a human-readable progress indicator showing chunk completion status.
    Checks for final concatenated video first, then shows chunk progress.

    Parameters
    ----------
    workflow_path : Path
        Path to workflow directory.

    Returns
    -------
    str
        Progress string: 'DONE' if video complete, 'X/Y' for chunk progress,
        or '-' if not started.
    """
    try:
        config = load_config(workflow_path)

        # Check if final video exists
        output_video = get_path(workflow_path, config, 'output_video')
        if output_video.exists():
            return 'DONE'

        # Get chunk progress
        last_chunk = get_last_chunk_end_frame(workflow_path)

        if last_chunk == 0:
            return '-'

        # Use total frame count from input video (not max_sbs which may be incomplete)
        total_frames = get_total_frame_count(workflow_path)
        if total_frames > 0:
            # Cap displayed progress to avoid showing more frames than total due to estimation variance
            display_progress = min(last_chunk, total_frames)
            return f'{display_progress}/{total_frames}'

        return str(last_chunk)

    except Exception:
        return '-'
