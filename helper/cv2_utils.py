"""
OpenCV Utilities
================

Utilities for working with OpenCV.
"""

from __future__ import annotations

from contextlib import contextmanager

import cv2


__all__ = [
    'suppress_cv2_logging'
]


@contextmanager
def suppress_cv2_logging():
    """
    Temporarily suppress OpenCV logging to prevent error messages in stdout.

    Falls back to a no-op if the OpenCV version doesn't support getLogLevel/setLogLevel
    (added in OpenCV 4.5.4).

    Yields
    ------
    None
        Context manager yields nothing.

    Examples
    --------
    with suppress_cv2_logging():
        image = cv2.imread('image.jpg')
    """
    # getLogLevel was added in OpenCV 4.5.4
    if not hasattr(cv2, 'getLogLevel'):
        yield
        return

    old_level = cv2.getLogLevel()
    try:
        cv2.setLogLevel(0)
        yield
    finally:
        cv2.setLogLevel(old_level)