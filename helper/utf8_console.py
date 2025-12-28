"""
UTF-8 Encoding Configuration
=============================

Configure console output encoding to prevent UnicodeEncodeError on Windows systems.
"""

from __future__ import annotations

import io
import sys


def setup_utf8_encoding() -> None:
    """
    Configure stdout and stderr to use UTF-8 encoding.

    Wraps system output streams with UTF-8 encoding to handle special
    characters correctly. Only applies reconfiguration if current encoding
    is not already UTF-8.

    Should be called at application startup before any output operations.

    Notes
    -----
    Uses 'replace' error handling to substitute unencodable characters
    rather than raising exceptions.
    """
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Auto-execute when imported
setup_utf8_encoding()
