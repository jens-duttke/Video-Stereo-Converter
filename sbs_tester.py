#!/usr/bin/env python3
"""
Stereo 3D Tester
================

Interactive tool to test and tune SBS generation parameters.
Loads stereo settings from workflow config.json and saves changes back.

Usage:
    python sbs_tester.py "D:/Video-Processing/workflow"
"""

from __future__ import annotations

import helper.utf8_console  # noqa: F401 # pyright: ignore[reportUnusedImport]
import helper.terminal_title  # noqa: F401 # pyright: ignore[reportUnusedImport]

import ctypes
import ctypes.wintypes
import sys
import threading
import time
import tkinter as tk
import warnings
import winsound
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
import torch

from helper.config_manager import ConfigError, find_valid_frames, get_frame_paths, load_config, update_stereo_params
from helper.stereo_core import StereoGenerator, StereoParams, load_image_pair


warnings.filterwarnings('ignore')


class PreviewWindow:
    """OpenCV-based preview window with fullscreen support for 3D monitors."""

    def __init__(self) -> None:
        """Initialize preview window."""
        self.window_name = 'SBS Preview (Press F for Fullscreen, ESC to exit fullscreen)'
        self.is_fullscreen = False
        self.current_image: Optional[np.ndarray] = None
        self.fullscreen_image: Optional[np.ndarray] = None
        self.running = True
        self.lock = threading.Lock()
        self.target_monitor = 0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 360)

        self._detect_monitors()

    def update_image(self, image: np.ndarray) -> None:
        """
        Update the preview image (thread-safe).

        Parameters
        ----------
        image : np.ndarray
            RGB image to display.
        """
        with self.lock:
            self.current_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.fullscreen_image = None

    def toggle_fullscreen(self, monitor: Optional[dict] = None) -> None:
        """Toggle fullscreen mode for 3D monitor."""
        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            if monitor is None:
                try:
                    rect = cv2.getWindowImageRect(self.window_name)
                    win_x, win_y = rect[0], rect[1]
                    for idx, mon in enumerate(self.monitors):
                        if (mon['x'] <= win_x < mon['x'] + mon['width'] and
                            mon['y'] <= win_y < mon['y'] + mon['height']):
                            self.target_monitor = idx
                            break
                except Exception:
                    pass

                monitor = self.monitors[self.target_monitor]

            cv2.moveWindow(self.window_name, monitor['x'], monitor['y'])
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow(self.window_name, monitor['x'], monitor['y'])

            print(f"Fullscreen on monitor {self.target_monitor}: {monitor['width']}x{monitor['height']*2} (stretched)")
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 360)

    def cycle_monitor(self) -> None:
        """Cycle through available monitors for fullscreen."""
        if len(self.monitors) > 1:
            self.target_monitor = (self.target_monitor + 1) % len(self.monitors)
            monitor = self.monitors[self.target_monitor]
            print(f"Fullscreen target: Monitor {self.target_monitor} ({monitor['width']}x{monitor['height']})")

            if self.is_fullscreen:
                self.toggle_fullscreen()
                self.toggle_fullscreen(monitor)

    def process_events(self) -> bool:
        """
        Process window events.

        Returns
        -------
        bool
            False if window should close, True otherwise.
        """
        with self.lock:
            if self.current_image is not None:
                if self.is_fullscreen:
                    if self.fullscreen_image is None:
                        monitor = self.monitors[self.target_monitor]
                        self.fullscreen_image = self._create_fullscreen_image(self.current_image, monitor)
                    cv2.imshow(self.window_name, self.fullscreen_image)
                else:
                    cv2.imshow(self.window_name, self.current_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('f') or key == ord('F'):
            self.toggle_fullscreen()
        elif key == ord('m') or key == ord('M'):
            self.cycle_monitor()
        elif key == 27:
            if self.is_fullscreen:
                self.toggle_fullscreen()
            else:
                return False

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        return True

    def close(self) -> None:
        """Close the preview window."""
        self.running = False
        cv2.destroyAllWindows()

    def _detect_monitors(self) -> None:
        """Detect available monitors."""
        try:
            user32 = ctypes.windll.user32
            self.monitors = [{
                'x': 0,
                'y': 0,
                'width': user32.GetSystemMetrics(0),
                'height': user32.GetSystemMetrics(1)
            }]

            try:
                monitors = []

                def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                    rect = lprcMonitor.contents
                    monitors.append({
                        'x': rect.left,
                        'y': rect.top,
                        'width': rect.right - rect.left,
                        'height': rect.bottom - rect.top
                    })
                    return True

                MONITORENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(ctypes.wintypes.RECT), ctypes.c_double)
                callback_func = MONITORENUMPROC(callback)
                user32.EnumDisplayMonitors(None, None, callback_func, 0)
                if monitors:
                    self.monitors = monitors
            except Exception:
                pass

            print(f'Detected {len(self.monitors)} monitor(s)')
            for i, m in enumerate(self.monitors):
                print(f"  Monitor {i}: {m['width']}x{m['height']} at ({m['x']}, {m['y']})")
        except Exception:
            self.monitors = [{'x': 0, 'y': 0, 'width': 1920, 'height': 1080}]

    def _create_fullscreen_image(self, image: np.ndarray, monitor: dict) -> np.ndarray:
        """Create image for 3D monitor fullscreen mode (stretched to double height)."""
        screen_w = monitor['width']
        screen_h = monitor['height']

        target_w = screen_w
        target_h = screen_h * 2

        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return resized


class ControlPanel:
    """Tkinter-based control panel for workflow-based parameter adjustment."""

    def __init__(
        self,
        workflow_path: Path,
        config: dict,
        valid_frames: list[int],
        on_parameter_change,
        on_frame_change,
        on_cache_invalidate
    ) -> None:
        """
        Initialize control panel.

        Parameters
        ----------
        workflow_path : Path
            Path to the workflow directory.
        config : dict
            Loaded configuration dictionary.
        valid_frames : list of int
            List of valid frame numbers.
        on_parameter_change : callable
            Callback when parameters change.
        on_frame_change : callable
            Callback when frame selection changes.
        on_cache_invalidate : callable
            Callback to invalidate cache.
        """
        self.workflow_path = workflow_path
        self.config = config
        self.valid_frames = valid_frames
        self.on_parameter_change = on_parameter_change
        self.on_frame_change = on_frame_change
        self._on_cache_invalidate = on_cache_invalidate

        self.root = tk.Tk()
        self.root.title(f'SBS Tester - {workflow_path.name}')
        self.root.geometry('700x650')
        self.root.resizable(True, True)

        # Frame number variable
        self.frame_number = tk.StringVar()
        self.frame_entry_valid = True

        # Load stereo params from config
        stereo = config['stereo']
        self.max_disparity = tk.DoubleVar(value=stereo['max_disparity'])
        self.convergence = tk.DoubleVar(value=stereo['convergence'])
        self.super_sampling = tk.DoubleVar(value=stereo['super_sampling'])
        self.edge_softness = tk.DoubleVar(value=stereo['edge_softness'])
        self.artifact_smoothing = tk.DoubleVar(value=stereo['artifact_smoothing'])
        self.depth_gamma = tk.DoubleVar(value=stereo['depth_gamma'])
        self.sharpen = tk.DoubleVar(value=stereo['sharpen'])

        self.show_depth = tk.BooleanVar(value=False)

        self._update_timer = None
        self._frame_entry_widget = None

        self._build_ui()

        # Set initial frame
        if valid_frames:
            self.frame_number.set(str(valid_frames[0]))
            self._validate_frame_number()

    def get_stereo_params(self) -> StereoParams:
        """Get current parameter values."""
        return StereoParams(
            max_disparity=self.max_disparity.get(),
            convergence=self.convergence.get(),
            super_sampling=self.super_sampling.get(),
            edge_softness=self.edge_softness.get(),
            artifact_smoothing=self.artifact_smoothing.get(),
            depth_gamma=self.depth_gamma.get(),
            sharpen=self.sharpen.get(),
        )

    def get_current_frame_number(self) -> int | None:
        """Get current frame number if valid."""
        try:
            frame_num = int(self.frame_number.get())
            if frame_num in self.valid_frames:
                return frame_num
        except ValueError:
            pass
        return None

    def set_status(self, text: str, color: str = 'black') -> None:
        """Update status label."""
        self.status_label.config(text=text, foreground=color)

    def update(self) -> None:
        """Process Tkinter events."""
        self.root.update()

    def is_running(self) -> bool:
        """Check if the control panel is still running."""
        try:
            self.root.winfo_exists()
            return True
        except tk.TclError:
            return False

    def close(self) -> None:
        """Close the control panel."""
        try:
            self.root.destroy()
        except Exception:
            pass

    def _build_ui(self) -> None:
        """Build the control panel UI."""
        main_frame = ttk.Frame(self.root, padding='10')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Workflow info
        workflow_frame = ttk.LabelFrame(main_frame, text='Workflow', padding='5')
        workflow_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(workflow_frame, text=f'Path: {self.workflow_path}').pack(anchor=tk.W)
        ttk.Label(workflow_frame, text=f'Valid frames: {len(self.valid_frames)}').pack(anchor=tk.W)

        # Frame selection
        frame_frame = ttk.LabelFrame(main_frame, text='Frame Selection', padding='5')
        frame_frame.pack(fill=tk.X, pady=(0, 10))

        frame_nav = ttk.Frame(frame_frame)
        frame_nav.pack(fill=tk.X)

        ttk.Button(frame_nav, text='< Prev', command=self._prev_frame).pack(side=tk.LEFT, padx=5)

        ttk.Label(frame_nav, text='Frame:').pack(side=tk.LEFT, padx=(20, 5))
        self._frame_entry_widget = ttk.Entry(frame_nav, textvariable=self.frame_number, width=10)
        self._frame_entry_widget.pack(side=tk.LEFT)
        self._frame_entry_widget.bind('<Return>', lambda e: self._on_frame_entry_change())
        self._frame_entry_widget.bind('<FocusOut>', lambda e: self._on_frame_entry_change())
        self.frame_number.trace_add('write', lambda *args: self._validate_frame_number())

        ttk.Button(frame_nav, text='Next >', command=self._next_frame).pack(side=tk.LEFT, padx=5)

        if self.valid_frames:
            range_text = f'(Range: {min(self.valid_frames)} - {max(self.valid_frames)})'
        else:
            range_text = '(No valid frames)'
        ttk.Label(frame_nav, text=range_text, foreground='gray').pack(side=tk.LEFT, padx=20)

        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text='Parameters', padding='5')
        param_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self._create_slider(param_frame, 0, 'Max Disparity', self.max_disparity, 5, 100, 0.5, 'Pixel displacement (14-60 optimal)')
        self._create_slider(param_frame, 1, 'Convergence', self.convergence, -50, 50, 1.0, 'Focal plane shift (-further, +closer)')
        self._create_slider(param_frame, 2, 'Super Sampling', self.super_sampling, 1.0, 4.0, 0.1, 'Internal upscale factor (1.0-3.0 optimal)')
        self._create_slider(param_frame, 3, 'Edge Softness', self.edge_softness, 0, 30, 0.5, 'Depth edge softening (0-30)')
        self._create_slider(param_frame, 4, 'Artifact Smoothing', self.artifact_smoothing, 0, 5, 0.1, 'Warping artifact smoothing (0-5)')
        self._create_slider(param_frame, 5, 'Depth Gamma', self.depth_gamma, 0.1, 2.0, 0.05, 'Gamma correction (0.2-0.5 optimal)')
        self._create_slider(param_frame, 6, 'Sharpen', self.sharpen, 0, 16, 0.5, 'Unsharp mask strength (0.5-16)')

        # View controls
        view_frame = ttk.Frame(main_frame)
        view_frame.pack(fill=tk.X, pady=(10, 0))

        self.depth_button = ttk.Button(view_frame, text='Show Depth Map (Hold)')
        self.depth_button.pack(side=tk.LEFT, padx=5)
        self.depth_button.bind('<ButtonPress-1>', lambda e: self._on_depth_button_press())
        self.depth_button.bind('<ButtonRelease-1>', lambda e: self._on_depth_button_release())

        ttk.Button(view_frame, text='Save to Config', command=self._save_to_config).pack(side=tk.RIGHT, padx=5)

        # Status
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)

        self.status_label = ttk.Label(status_frame, text='Ready', foreground='gray')
        self.status_label.pack(side=tk.LEFT)

        if torch.cuda.is_available():
            gpu_text = f'GPU: {torch.cuda.get_device_name(0)}'
        else:
            gpu_text = 'GPU: Not available (using CPU)'
        ttk.Label(status_frame, text=gpu_text, foreground='blue').pack(side=tk.RIGHT)

        ttk.Label(main_frame, text='Preview: F = Fullscreen (3D), M = Switch Monitor, ESC = Exit',
                  foreground='gray').pack(pady=(5, 0))

    def _create_slider(
        self, parent: ttk.Frame, row: int, label: str, variable: tk.DoubleVar,
        from_: float, to: float, resolution: float, tooltip: str
    ) -> None:
        """Create a labeled slider with value display."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)

        if resolution >= 1.0:
            decimal_places = 0
        elif resolution >= 0.1:
            decimal_places = 1
        elif resolution >= 0.01:
            decimal_places = 2
        else:
            decimal_places = 3

        value_label = ttk.Label(parent, text=f'{variable.get():.{decimal_places}f}', width=6)
        value_label.grid(row=row, column=2, padx=5)

        def on_change(val):
            val_float = float(val)
            rounded_val = round(val_float / resolution) * resolution
            variable.set(rounded_val)
            value_label.config(text=f'{rounded_val:.{decimal_places}f}')

        slider = ttk.Scale(parent, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL, length=200, command=on_change)
        slider.grid(row=row, column=1, padx=10, pady=5)

        slider.bind('<ButtonRelease-1>', lambda e: self._schedule_update())

        ttk.Label(parent, text=tooltip, foreground='gray', font=('', 8)).grid(row=row, column=3, sticky=tk.W, padx=5)

    def _validate_frame_number(self) -> None:
        """Validate frame number input and update entry styling."""
        try:
            frame_num = int(self.frame_number.get())
            valid = frame_num in self.valid_frames
        except ValueError:
            valid = False

        if self._frame_entry_widget:
            if valid:
                self._frame_entry_widget.configure(style='TEntry')
                self.frame_entry_valid = True
            else:
                # Create red background style
                style = ttk.Style()
                style.configure('Invalid.TEntry', fieldbackground='#ffcccc')
                self._frame_entry_widget.configure(style='Invalid.TEntry')
                self.frame_entry_valid = False

    def _on_frame_entry_change(self) -> None:
        """Handle frame entry change."""
        if self.frame_entry_valid:
            self.on_frame_change()

    def _prev_frame(self) -> None:
        """Navigate to previous valid frame."""
        current = self.get_current_frame_number()
        if current is None:
            if self.valid_frames:
                self.frame_number.set(str(self.valid_frames[0]))
                self.on_frame_change()
            return

        # Find previous valid frame
        idx = self.valid_frames.index(current)
        if idx > 0:
            self.frame_number.set(str(self.valid_frames[idx - 1]))
            self.on_frame_change()

    def _next_frame(self) -> None:
        """Navigate to next valid frame."""
        current = self.get_current_frame_number()
        if current is None:
            if self.valid_frames:
                self.frame_number.set(str(self.valid_frames[0]))
                self.on_frame_change()
            return

        # Find next valid frame
        idx = self.valid_frames.index(current)
        if idx < len(self.valid_frames) - 1:
            self.frame_number.set(str(self.valid_frames[idx + 1]))
            self.on_frame_change()

    def _on_depth_button_press(self) -> None:
        """Handle depth map button press."""
        self.show_depth.set(True)
        self.on_parameter_change()

    def _on_depth_button_release(self) -> None:
        """Handle depth map button release."""
        self.show_depth.set(False)
        self.on_parameter_change()

    def _schedule_update(self) -> None:
        """Schedule a parameter update with debouncing."""
        if self._update_timer is not None:
            self.root.after_cancel(self._update_timer)
        self._update_timer = self.root.after(100, self._trigger_update)

    def _trigger_update(self) -> None:
        """Trigger the parameter change callback."""
        self._update_timer = None
        if self._on_cache_invalidate:
            self._on_cache_invalidate()
        self.on_parameter_change()

    def _save_to_config(self) -> None:
        """Save current stereo parameters to config.json."""
        params = self.get_stereo_params()

        stereo_dict = {
            'max_disparity': params.max_disparity,
            'convergence': params.convergence,
            'super_sampling': params.super_sampling,
            'edge_softness': params.edge_softness,
            'artifact_smoothing': params.artifact_smoothing,
            'depth_gamma': params.depth_gamma,
            'sharpen': params.sharpen,
        }

        try:
            update_stereo_params(self.workflow_path, stereo_dict)
            messagebox.showinfo('Config Saved', 'Stereo parameters saved to config.json successfully!')
            self.set_status('Config saved', 'green')
        except Exception as e:
            messagebox.showerror('Save Error', f'Failed to save config: {e}')
            self.set_status(f'Save failed: {e}', 'red')


class SBSTesterApp:
    """Main application combining control panel and preview."""

    def __init__(self, workflow_path: Path) -> None:
        """
        Initialize the application.

        Parameters
        ----------
        workflow_path : Path
            Path to the workflow directory.
        """
        self.workflow_path = workflow_path

        # Load config
        try:
            self.config = load_config(workflow_path)
        except ConfigError as e:
            print(f'ERROR: {e}')
            sys.exit(1)

        # Find valid frames
        self.valid_frames = find_valid_frames(workflow_path, self.config)
        if not self.valid_frames:
            print('ERROR: No valid frame/depth pairs found in workflow')
            print('  Make sure both frames/ and depth_maps/ directories have matching files')
            sys.exit(1)

        print(f'Found {len(self.valid_frames)} valid frame/depth pairs')

        # Setup device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.generator = StereoGenerator(self.device)

        self.current_rgb: Optional[np.ndarray] = None
        self.current_depth: Optional[np.ndarray] = None

        self.cached_sbs: Optional[np.ndarray] = None
        self.cached_depth_sbs: Optional[np.ndarray] = None

        self.processing = False
        self.pending_update = False

        self.preview = PreviewWindow()
        self.control = ControlPanel(
            workflow_path=workflow_path,
            config=self.config,
            valid_frames=self.valid_frames,
            on_parameter_change=self._on_parameter_change,
            on_frame_change=self._on_frame_change,
            on_cache_invalidate=self._invalidate_cache
        )

        # Load initial frame
        self._on_frame_change()

    def run(self) -> None:
        """Main application loop."""
        print('SBS Tester started')
        print('  - Control panel: Adjust parameters')
        print('  - Preview window:')
        print('      F = Fullscreen (3D mode: stretched to 2x height)')
        print('      M = Switch target monitor')
        print('      ESC = Exit fullscreen')

        try:
            while self.control.is_running():
                self.control.update()

                if not self.preview.process_events():
                    break

        except KeyboardInterrupt:
            print('\nInterrupted')

        finally:
            self.preview.close()
            self.control.close()
            print('SBS Tester closed')

    def _on_frame_change(self) -> None:
        """Handle frame selection change."""
        frame_num = self.control.get_current_frame_number()
        if frame_num is None:
            return

        paths = get_frame_paths(self.workflow_path, self.config, frame_num)
        if paths is None:
            self.control.set_status(f'Frame {frame_num} not found', 'red')
            return

        frame_path, depth_path = paths

        try:
            self.control.set_status('Loading images...', 'blue')
            self.control.update()

            self.current_rgb, self.current_depth = load_image_pair(frame_path, depth_path)

            self.cached_sbs = None
            self.cached_depth_sbs = None

            self.control.set_status(f'Loaded frame {frame_num}: {self.current_rgb.shape[1]}x{self.current_rgb.shape[0]}', 'green')
            self._generate_preview()

        except Exception as e:
            self.control.set_status(f'Error: {e}', 'red')
            self.current_rgb = None
            self.current_depth = None

    def _on_parameter_change(self) -> None:
        """Handle parameter change."""
        if self.current_rgb is None:
            return

        if self.processing:
            self.pending_update = True
            return

        self._generate_preview()

    def _invalidate_cache(self) -> None:
        """Invalidate cached SBS when parameters change."""
        self.cached_sbs = None

    def _generate_preview(self) -> None:
        """Generate and display preview."""
        if self.current_rgb is None or self.current_depth is None:
            return

        show_depth = self.control.show_depth.get()

        if show_depth:
            if self.cached_depth_sbs is None:
                depth_normalized = (
                    (self.current_depth - self.current_depth.min()) /
                    (self.current_depth.max() - self.current_depth.min() + 1e-8) * 255
                ).astype(np.uint8)
                depth_rgb = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
                self.cached_depth_sbs = np.hstack([depth_rgb, depth_rgb])

            self.preview.update_image(self.cached_depth_sbs)
            self.control.set_status('Showing depth map', 'blue')
            return

        if self.cached_sbs is not None:
            self.preview.update_image(self.cached_sbs)
            self.control.set_status('Ready (from cache)', 'green')
            return

        self.processing = True
        self.control.set_status('Generating preview...', 'blue')
        self.control.update()

        start_time = time.time()

        try:
            sbs = self.generator.process_frame(
                self.current_rgb,
                self.current_depth,
                params=self.control.get_stereo_params()
            )

            elapsed = time.time() - start_time

            self.cached_sbs = sbs

            self.preview.update_image(sbs)
            self.control.set_status(f'Ready (took {elapsed:.2f}s)', 'green')

            winsound.Beep(800, 100)

        except Exception as e:
            self.control.set_status(f'Error: {e}', 'red')

        finally:
            self.processing = False

            if self.pending_update:
                self.pending_update = False
                self.control.root.after(10, self._generate_preview)


def main() -> None:
    """Main entry point for the SBS tester application."""
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='SBS Tester - Interactive parameter tuning',
        epilog=(
            'Example:\n'
            '  python sbs_tester.py "D:/Video-Processing/workflow"\n'
        )
    )

    parser.add_argument('workflow_path', type=Path, nargs='?', default=None, help='Path to workflow directory containing config.json')

    args = parser.parse_args()

    # If no workflow path provided, open folder dialog
    if args.workflow_path is None:
        root = tk.Tk()
        root.withdraw()
        workflow_path = filedialog.askdirectory(title='Select Workflow Directory')
        root.destroy()

        if not workflow_path:
            print('No workflow directory selected.')
            sys.exit(1)

        workflow_path = Path(workflow_path)
    else:
        workflow_path = args.workflow_path

    # Validate workflow directory
    if not workflow_path.is_dir():
        print(f'ERROR: Workflow directory not found: {workflow_path}')
        sys.exit(1)

    config_file = workflow_path / 'config.json'
    if not config_file.exists():
        print(f'ERROR: config.json not found in workflow directory: {workflow_path}')
        sys.exit(1)

    app = SBSTesterApp(workflow_path)
    app.run()


if __name__ == '__main__':
    main()
