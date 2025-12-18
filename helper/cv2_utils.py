from contextlib import contextmanager

import cv2


__all__ = [
    'suppress_cv2_logging'
]


@contextmanager
def suppress_cv2_logging():
    """
    Temporarily suppress OpenCV logging to prevent error messages in stdout.

    Yields
    ------
    None
        Context manager yields nothing.

    Examples
    --------
    with suppress_cv2_logging():
        image = cv2.imread('image.jpg')
    """
    old_level = cv2.getLogLevel()
    try:
        cv2.setLogLevel(0)
        yield
    finally:
        cv2.setLogLevel(old_level)