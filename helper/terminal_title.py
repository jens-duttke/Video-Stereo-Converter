"""
Terminal Title Utility
======================

Sets the terminal window title to show the script name and arguments.
Works on Windows, Linux, and macOS without third-party dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def set_terminal_title(title: str | None = None) -> None:
    """
    Set terminal window title.

    If no title is provided, uses the script name and arguments.
    Does nothing if DISABLE_TERMINAL_TITLE environment variable is set.

    Parameters
    ----------
    title : str | None
        Custom title to set. If None, uses "script.py arg1 arg2" format.

    Notes
    -----
    - Windows: Uses kernel32.SetConsoleTitleW
    - Linux/macOS: Uses ANSI escape sequence OSC 0
    """
    if os.environ.get('DISABLE_TERMINAL_TITLE'):
        return

    if title is None:
        script_name = Path(sys.argv[0]).name
        args = ' '.join(sys.argv[1:])
        title = f'{script_name} {args}'.strip()

    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    else:
        # ANSI escape sequence OSC 0 (works in most terminal emulators)
        # \033]0;TITLE\007 sets window title
        sys.stdout.write(f'\033]0;{title}\007')
        sys.stdout.flush()


# Auto-execute when imported
set_terminal_title()
