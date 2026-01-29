#!/usr/bin/env python3
"""
Stereo 3D Generator
===================

Processes all frame/depth pairs to generate side-by-side stereo images.
Reads stereo parameters from workflow config.json.

Usage:
    python sbs_generator.py "D:/Video-Processing/workflow"
    python sbs_generator.py "D:/Video-Processing/workflow" --cpu
"""

from __future__ import annotations

import helper.utf8_console  # noqa: F401 # pyright: ignore[reportUnusedImport]
import helper.terminal_title  # noqa: F401 # pyright: ignore[reportUnusedImport]

import sys
import threading
import time
import warnings
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from queue import Queue
from typing import List, Tuple

import cv2
import torch
from tqdm import tqdm

from helper.config_manager import ConfigError, get_path, load_config
from helper.cv2_utils import suppress_cv2_logging
from helper.frame_utils import extract_frame_number
from helper.stereo_core import StereoGenerator, StereoParams, load_image_pair


warnings.filterwarnings('ignore')

# Exit code for GPU errors - signals parent process to handle GPU failure
GPU_ERROR_EXIT_CODE = 100


def _check_gpu_health(device: str) -> bool:
    """
    Verify GPU is still functioning correctly.

    Performs a simple tensor computation with known result to detect GPU driver crashes
    that may not raise exceptions but produce garbage data.

    Parameters
    ----------
    device : str
        Device string ('cuda' or 'cpu').

    Returns
    -------
    bool
        True if GPU is healthy or device is CPU, False if GPU appears corrupted.
    """
    if device != 'cuda':
        return True
    try:
        test = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        result = (test * 2).sum().item()
        return abs(result - 12.0) < 0.001
    except Exception:
        return False


def _find_frame_pairs(frames_dir: Path, depth_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Find matching frame/depth pairs.

    Depth map preference order: .tif first, then .png.

    Parameters
    ----------
    frames_dir : Path
        Directory containing frame images.
    depth_dir : Path
        Directory containing depth maps.

    Returns
    -------
    list of tuple
        List of (frame_path, depth_path, frame_num) tuples.
    """
    pairs = []
    missing_count = 0
    first_missing = None
    last_missing = None

    frame_files = sorted(frames_dir.glob('frame_*.png'))

    for frame_path in frame_files:
        frame_name = frame_path.stem
        frame_num = frame_name.replace('frame_', '')

        # Prefer .tif, fallback to .png
        depth_filepath = depth_dir / f'depth_frame_{frame_num}.tif'
        if not depth_filepath.exists():
            depth_filepath = depth_dir / f'depth_frame_{frame_num}.png'
            if not depth_filepath.exists():
                if first_missing is None:
                    first_missing = frame_num
                last_missing = frame_num
                missing_count += 1
                continue

        pairs.append((frame_path, depth_filepath, frame_num))

    if missing_count > 0:
        print(f'Missing depth maps: {missing_count} of {len(frame_files)} frames in range of frame_{first_missing} to frame_{last_missing}')

    return pairs


def main() -> None:
    """Main entry point for stereo 3D generation."""
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='Stereo 3D Generator - Create side-by-side stereo images',
        epilog=(
            'Example:\n'
            '  python sbs_generator.py "D:/Video-Processing/workflow"\n'
            '  python sbs_generator.py "D:/Video-Processing/workflow" --cpu\n'
        )
    )

    parser.add_argument('workflow_path', type=Path, help='Path to workflow directory containing config.json')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing')
    parser.add_argument('--no-interactive', action='store_true', help='Exit on error instead of waiting for user input (for orchestrator)')

    args = parser.parse_args()

    # Validate workflow directory
    if not args.workflow_path.is_dir():
        print(f'ERROR: Workflow directory not found: {args.workflow_path}')
        return

    # Load config
    try:
        config = load_config(args.workflow_path)
    except ConfigError as e:
        print(f'ERROR: {e}')
        return

    # Get paths from config
    frames_dir = get_path(args.workflow_path, config, 'frames')
    depth_dir = get_path(args.workflow_path, config, 'depth_maps')
    output_dir = get_path(args.workflow_path, config, 'sbs')

    if not frames_dir.exists():
        print(f'ERROR: Frames directory not found: {frames_dir}')
        return
    if not depth_dir.exists():
        print(f'ERROR: Depth directory not found: {depth_dir}')
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get stereo params from config
    stereo_config = config['stereo']
    params = StereoParams(
        max_disparity=stereo_config['max_disparity'],
        convergence=stereo_config['convergence'],
        super_sampling=stereo_config['super_sampling'],
        edge_softness=stereo_config['edge_softness'],
        artifact_smoothing=stereo_config['artifact_smoothing'],
        depth_gamma=stereo_config['depth_gamma'],
        sharpen=stereo_config['sharpen']
    )

    print('Scanning for frame pairs...')
    all_pairs = _find_frame_pairs(frames_dir, depth_dir)

    pairs = []
    skipped = 0
    for frame_path, depth_path, frame_num in all_pairs:
        output_path = output_dir / f'sbs_{frame_num}.png'
        if output_path.exists():
            skipped += 1
        else:
            pairs.append((frame_path, depth_path, frame_num))

    print(f'Found: {len(all_pairs)} frame pairs, {skipped} already processed, {len(pairs)} to process')

    if not pairs:
        print('All frames already processed.')
        return

    # Device selection
    if args.cpu or not torch.cuda.is_available():
        device = 'cpu'
        print('\033[31mUsing CPU\033[0m')
    else:
        device = 'cuda'
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')

    generator = StereoGenerator(device=device)

    print(f'Parameters: disparity={params.max_disparity}, convergence={params.convergence}, '
          f'super_sampling={params.super_sampling}, edge_softness={params.edge_softness}, '
          f'smoothing={params.artifact_smoothing}, gamma={params.depth_gamma}, sharpen={params.sharpen}')

    # Free space config
    free_space_mode = config.get('free_space', {}).get('sbs_generator', 'none')
    if free_space_mode == 'frame':
        print('Free space mode: deleting frame files after processing')
    elif free_space_mode == 'depth':
        print('Free space mode: deleting depth files after processing')
    elif free_space_mode == 'all':
        print('Free space mode: deleting frame and depth files after processing')

    # Threading setup
    load_queue: Queue = Queue(maxsize=2)
    save_queue: Queue = Queue(maxsize=4)
    stop_loader = threading.Event()
    save_error_event = threading.Event()

    def _loader_thread() -> None:
        for frame_path, depth_path, frame_num in pairs:
            if stop_loader.is_set():
                break
            try:
                rgb, depth = load_image_pair(frame_path, depth_path)
                load_queue.put((rgb, depth, frame_num, frame_path, depth_path))
            except Exception as e:
                print(f'  Error loading {frame_num}: {e}')
        load_queue.put(None)

    def _saver_thread() -> None:
        while not stop_loader.is_set():
            item = save_queue.get()
            if item is None:
                break

            sbs, output_path, frame_path, depth_path = item

            retries = 3
            success = False

            while not success:
                for attempt in range(retries):
                    try:
                        with suppress_cv2_logging():
                            result = cv2.imwrite(output_path, cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

                            if result:
                                success = True
                                break
                            else:
                                raise IOError(f'cv2.imwrite returned False for {output_path}')
                    except Exception as e:
                        save_error_event.set()
                        frame_num = extract_frame_number(output_path)
                        print(f'\nSave failed for SBS frame #{frame_num} ({attempt + 1}/{retries}): {e}')
                        if attempt < retries - 1:
                            time.sleep(60)

                if not success:
                    if args.no_interactive:
                        print('\nERROR: Failed to write output file. Exiting (non-interactive mode).')
                        stop_loader.set()
                        break
                    print(
                        '\nERROR: Failed to write output file.\n'
                        'Please resolve the storage issue and press any key to retry.'
                    )
                    try:
                        input()
                    except (EOFError, KeyboardInterrupt):
                        print('\nSave aborted by user.')
                        stop_loader.set()
                        break

            if success:
                save_error_event.clear()
                # Delete files based on free_space mode
                if free_space_mode in ('frame', 'all'):
                    try:
                        frame_path.unlink(missing_ok=True)
                    except OSError:
                        pass  # Ignore deletion errors
                if free_space_mode in ('depth', 'all'):
                    try:
                        depth_path.unlink(missing_ok=True)
                    except OSError:
                        pass  # Ignore deletion errors

            save_queue.task_done()

    # Process
    pbar = tqdm(total=len(all_pairs), initial=skipped, unit='img', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix} [{elapsed}<{remaining}, {rate_noinv_fmt}]', mininterval=0.5)

    loader = threading.Thread(target=_loader_thread, daemon=True)
    saver = threading.Thread(target=_saver_thread, daemon=True)
    loader.start()
    saver.start()

    processed_count = 0
    try:
        while not stop_loader.is_set():
            item = load_queue.get()
            if item is None:
                break

            rgb, depth, frame_num, frame_path, depth_path = item
            pbar.set_postfix_str(f'Frame: {extract_frame_number(frame_path)}')

            # Check GPU health before processing to detect driver crashes early
            if not _check_gpu_health(device):
                print('\nERROR: GPU health check failed - driver may have crashed')
                stop_loader.set()
                pbar.close()
                sys.exit(GPU_ERROR_EXIT_CODE)

            sbs = generator.process_frame(rgb, depth, params)

            if save_error_event.is_set():
                save_error_event.wait()

            output_path = str(output_dir / f'sbs_{frame_num}.png')
            save_queue.put((sbs, output_path, frame_path, depth_path))
            processed_count += 1

            pbar.update(1)

    except KeyboardInterrupt:
        print('\nInterrupted! Waiting for save queue...')
        stop_loader.set()

    pbar.close()

    if not stop_loader.is_set():
        save_queue.join()

    save_queue.put(None)
    saver.join()

    print(f'Done! Processed {processed_count} of {len(pairs)} frames.')


if __name__ == '__main__':
    main()
