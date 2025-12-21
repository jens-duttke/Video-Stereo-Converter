import subprocess
from pathlib import Path


__all__ = [
    'get_video_framerate',
    'get_video_duration',
    'parse_framerate',
    'estimate_frame_count',
]


def parse_framerate(framerate_str: str) -> float | None:
    """
    Parse framerate string to float value.

    Parameters
    ----------
    framerate_str : str
        Framerate string (e.g., '24000/1001', '30/1', '29.97').

    Returns
    -------
    float or None
        Framerate as float, or None if parsing fails.
    """
    try:
        if '/' in framerate_str:
            num, den = framerate_str.split('/')
            return float(num) / float(den)
        return float(framerate_str)
    except (ValueError, ZeroDivisionError):
        return None


def get_video_framerate(video_path: Path | str) -> str | None:
    """
    Get framerate from video file using ffprobe.

    Parameters
    ----------
    video_path : Path or str
        Path to the video file.

    Returns
    -------
    str or None
        Framerate as string (e.g., '24000/1001' or '30/1'), or None if detection fails.
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_video_duration(video_path: Path | str) -> float | None:
    """
    Get video duration in seconds using ffprobe.

    Parameters
    ----------
    video_path : Path or str
        Path to the video file.

    Returns
    -------
    float or None
        Duration in seconds, or None if detection fails.
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def estimate_frame_count(video_path: Path | str) -> int | None:
    """
    Estimate frame count from video duration and framerate.

    Calculating the exact frame count requires decoding all frames, which takes as long as
    processing the entire video. This estimation using duration * framerate is very accurate
    (typically within ±1 frame for CFR videos) and returns instantly.

    Parameters
    ----------
    video_path : Path or str
        Path to the video file.

    Returns
    -------
    int or None
        Estimated frame count, or None if detection fails.
    """
    duration = get_video_duration(video_path)
    if duration is None:
        return None

    framerate_str = get_video_framerate(video_path)
    if not framerate_str:
        return None

    framerate = parse_framerate(framerate_str)
    if framerate is None:
        return None

    return int(duration * framerate)
