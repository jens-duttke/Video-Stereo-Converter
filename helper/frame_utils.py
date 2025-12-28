"""
Frame Number Extraction
=======================

Utilities for extracting frame numbers from filenames.
"""

from __future__ import annotations

import os
import re


__all__ = [
    'extract_frame_number'
]


def extract_frame_number(filepath: str) -> int:
    """
    Extract frame number from filename.
    
    Expects filename pattern: *_NUMBER.extension
    (e.g., 'frame_0001.png', 'depth_frame_0001561.tif', 'sbs_151565.png')

    Parameters
    ----------
    filepath : str
        Path to frame file.

    Returns
    -------
    int
        Frame number as integer, or -1 if pattern not found.
    """
    match = re.search(r'_(\d+)\.', os.path.basename(filepath))

    return int(match.group(1)) if match else -1
