#!/usr/bin/env python3
"""
Depth Map Generator
===================

Generates depth maps from RGB frames using ONNX models.
Supports GPU acceleration via DirectML and configurable batching.

Usage:
    python depth_map_generator.py "D:/Video-Processing/workflow"
    python depth_map_generator.py "D:/Video-Processing/workflow" --cpu --batch-size 4
"""

import helper.terminal_title  # noqa: F401 # pyright: ignore[reportUnusedImport]

import glob
import os
import threading
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from queue import Queue
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from helper.config_manager import ConfigError, get_path, load_config, merge_cli_args
from helper.cv2_utils import suppress_cv2_logging
from helper.frame_utils import extract_frame_number


MODEL_PATH = './models/DepthPro_optimized.onnx'


def _preprocess_single(img: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Preprocess single image on CPU using NumPy/OpenCV.

    Following DepthPro official preprocessing:
    - BGR to RGB conversion (OpenCV loads as BGR, model expects RGB)
    - Resize to model input size (1536x1536 by default)
    - Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5] to range [-1, 1]

    Parameters
    ----------
    img : np.ndarray
        BGR image as numpy array [H, W, 3] uint8.
    target_height : int
        Target height for resizing.
    target_width : int
        Target width for resizing.

    Returns
    -------
    np.ndarray
        Preprocessed image as float16 [3, H, W].
    """
    if img.shape[0] != target_height or img.shape[1] != target_width:
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = img.astype(np.float16)
    img = np.transpose(img, (2, 0, 1))

    return img


def _preprocess_batch_cpu(images_data: List[Tuple[np.ndarray, Tuple[int, int], str]], target_height: int, target_width: int) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    """
    Preprocess batch on CPU.

    Parameters
    ----------
    images_data : list of tuple
        List of (img_numpy, original_size, img_path).
    target_height : int
        Target height for preprocessing.
    target_width : int
        Target width for preprocessing.

    Returns
    -------
    tuple
        Batch array [B, 3, H, W] float16, original sizes, and image paths.
    """
    batch_size = len(images_data)
    batch_array = np.empty((batch_size, 3, target_height, target_width), dtype=np.float16)
    original_sizes = []
    img_paths = []

    for idx, (img, original_size, img_path) in enumerate(images_data):
        original_sizes.append(original_size)
        img_paths.append(img_path)
        batch_array[idx] = _preprocess_single(img, target_height, target_width)

    return batch_array, original_sizes, img_paths


def _get_output_path(input_filename: str, output_dir: Path, use_16bit: bool) -> str:
    """
    Generate output path for depth map based on input filename and settings.

    Parameters
    ----------
    input_filename : str
        Original input filename (e.g., 'frame_0001.png').
    output_dir : Path
        Output directory.
    use_16bit : bool
        If True, generates .tif extension, otherwise .png.

    Returns
    -------
    str
        Full output path with appropriate extension.
    """
    base_name = os.path.splitext(input_filename)[0]
    ext = '.tif' if use_16bit else '.png'
    return str(output_dir / f'depth_{base_name}{ext}')


def _save_depth_map(depth_map: np.ndarray, original_size: Tuple[int, int], output_path: str) -> bool:
    """
    Normalize and save depth map.

    For 8-bit, depth maps are saved as PNG.
    For 16-bit, depth maps are saved as 16-bit TIFF with DEFLATE compression.

    Parameters
    ----------
    depth_map : np.ndarray
        Depth map array.
    original_size : tuple of int
        Original image size (width, height).
    output_path : str
        Output file path with extension (.png or .tif).

    Returns
    -------
    bool
        True if save was successful, False otherwise.
    """
    depth_map_resized = cv2.resize(depth_map.astype(np.float32), original_size, interpolation=cv2.INTER_LINEAR)

    depth_min = depth_map_resized.min()
    depth_max = depth_map_resized.max()
    depth_range = depth_max - depth_min

    if depth_range > 0:
        depth_map_resized = (depth_map_resized - depth_min) / depth_range

        ext = Path(output_path).suffix.lower()
        if ext == '.tif':
            depth_map_16 = (depth_map_resized * 65535).round().astype(np.uint16)
            with suppress_cv2_logging():
                result = cv2.imwrite(output_path, depth_map_16, [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_DEFLATE])
        else:
            depth_map_8 = (depth_map_resized * 255).round().astype(np.uint8)
            with suppress_cv2_logging():
                result = cv2.imwrite(output_path, depth_map_8)

        return result

    return False


def main() -> None:
    """Main entry point for depth map generation."""
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='Depth Map Generator - Generate depth maps from RGB frames',
        epilog=(
            'Example:\n'
            '  python depth_map_generator.py "D:/Video-Processing/workflow"\n'
            '  python depth_map_generator.py "D:/Video-Processing/workflow" --cpu --batch-size 4\n'
            '  python depth_map_generator.py "D:/Video-Processing/workflow" --start-frame 100 --end-frame 500\n'
        )
    )

    parser.add_argument('workflow_path', type=Path, help='Path to workflow directory containing config.json')
    parser.add_argument('--start-frame', type=int, default=None, help='First frame to process (inclusive)')
    parser.add_argument('--end-frame', type=int, default=None, help='Last frame to process (inclusive)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for inference (default from config)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference (default auto-detects GPU)')

    args = parser.parse_args()

    # Validate workflow directory
    if not args.workflow_path.is_dir():
        print(f'ERROR: Workflow directory not found: {args.workflow_path}')
        return

    # Load and merge config
    try:
        config = load_config(args.workflow_path)
    except ConfigError as e:
        print(f'ERROR: {e}')
        return

    # Merge CLI args (batch_size)
    cli_overrides = {'batch_size': args.batch_size}
    config = merge_cli_args(config, cli_overrides)

    # Get paths from config
    input_dir = get_path(args.workflow_path, config, 'frames')
    output_dir = get_path(args.workflow_path, config, 'depth_maps')
    use_16bit = config['depth']['save_16bit']
    batch_size = config['depth']['batch_size']

    if not input_dir.exists():
        print(f'ERROR: Frames directory not found: {input_dir}')
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: Model not found: {MODEL_PATH}')
        return

    # Setup ONNX session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.intra_op_num_threads = os.cpu_count()
    sess_options.inter_op_num_threads = 1

    if args.cpu:
        providers = ['CPUExecutionProvider']
    else:
        # Prefer GPU providers in order: DirectML (Windows), CUDA (Linux/Windows), then CPU fallback
        available = ort.get_available_providers()
        gpu_priority = ['DmlExecutionProvider', 'CUDAExecutionProvider', 'ROCMExecutionProvider', 'CoreMLExecutionProvider']
        providers = [p for p in gpu_priority if p in available]
        providers.append('CPUExecutionProvider')

    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_height, input_width = input_shape[2], input_shape[3]

    provider_name = session.get_providers()[0]
    print(f'Using: {provider_name}' if provider_name != 'CPUExecutionProvider' else f'\033[31mUsing: {provider_name}\033[0m')
    print(f'Batch Size: {batch_size}')
    print(f'Output Format: {"16-bit TIFF" if use_16bit else "8-bit PNG"}')

    # Find image files
    all_image_files = sorted(glob.glob(str(input_dir / 'frame_*.png')))

    if args.start_frame is not None or args.end_frame is not None:
        filtered_files = []
        for filepath in all_image_files:
            frame_num = extract_frame_number(filepath)
            if args.start_frame is not None and frame_num < args.start_frame:
                continue
            if args.end_frame is not None and frame_num > args.end_frame:
                continue
            filtered_files.append(filepath)

        all_image_files = filtered_files
        start_str = str(args.start_frame) if args.start_frame is not None else 'start'
        end_str = str(args.end_frame) if args.end_frame is not None else 'end'
        print(f'Frame range: {start_str} to {end_str}')

    # Filter already processed
    image_files = []
    skipped_count = 0

    for img_path in all_image_files:
        filename = os.path.basename(img_path)
        output_path = _get_output_path(filename, output_dir, use_16bit)
        if Path(output_path).exists():
            skipped_count += 1
        else:
            image_files.append(img_path)

    print(f'Found: {len(all_image_files)} images, {skipped_count} already processed, {len(image_files)} to process')

    if len(image_files) == 0:
        print('All images already processed.')
        return

    # Setup threading
    load_queue: Queue = Queue(maxsize=2)
    save_queue: Queue = Queue(maxsize=4)
    stop_loader = threading.Event()
    save_error_event = threading.Event()

    def _load_image_from_disk(img_path: str) -> Tuple[np.ndarray, Tuple[int, int], str]:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        original_size = (img.shape[1], img.shape[0])
        return img, original_size, img_path

    def _load_and_preprocess_batch(batch_files: List[str]) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
        images_data = [_load_image_from_disk(path) for path in batch_files]
        return _preprocess_batch_cpu(images_data, input_height, input_width)

    def _loader_thread() -> None:
        batch_indices = list(range(0, len(image_files), batch_size))
        for batch_idx in batch_indices:
            if stop_loader.is_set():
                break
            batch_files = image_files[batch_idx:batch_idx + batch_size]
            try:
                batch_data = _load_and_preprocess_batch(batch_files)
                load_queue.put(batch_data)
            except Exception as e:
                print(f'  Error loading batch at {batch_idx}: {e}')
        load_queue.put(None)

    def _saver_thread() -> None:
        while not stop_loader.is_set():
            item = save_queue.get()
            if item is None:
                break

            depth_map, original_size, output_path = item
            retries = 3
            success = False

            while not success:
                for attempt in range(retries):
                    try:
                        result = _save_depth_map(depth_map, original_size, output_path)
                        if result:
                            success = True
                            break
                        else:
                            raise IOError(f'cv2.imwrite returned False for {output_path}')
                    except Exception as e:
                        save_error_event.set()
                        frame_num = extract_frame_number(output_path)
                        print(f'\nSave failed for depth map frame #{frame_num} ({attempt + 1}/{retries}): {e}')
                        if attempt < retries - 1:
                            time.sleep(60)

                if not success:
                    print(
                        '\nERROR: Failed to write depth map.\n'
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

            save_queue.task_done()

    # Process
    pbar = tqdm(total=len(all_image_files), initial=skipped_count, unit='img', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix} [{elapsed}<{remaining}, {rate_noinv_fmt}]')

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

            current_batch, current_sizes, current_paths = item
            pbar.set_postfix_str(f'Frame: {extract_frame_number(current_paths[-1])}')

            outputs = session.run([output_name], {input_name: current_batch})
            depth_maps = outputs[0]

            for j in range(len(current_paths)):
                if save_error_event.is_set():
                    save_error_event.wait()

                depth_map = depth_maps[j].squeeze().copy()
                filename = os.path.basename(current_paths[j])
                output_path = _get_output_path(filename, output_dir, use_16bit)

                save_queue.put((depth_map, current_sizes[j], output_path))
                processed_count += 1

            pbar.update(len(current_paths))

    except KeyboardInterrupt:
        print('\nInterrupted! Waiting for save queue...')
        stop_loader.set()

    pbar.close()

    if not stop_loader.is_set():
        save_queue.join()

    save_queue.put(None)
    saver.join()

    print(f'Done! Processed {processed_count} of {len(image_files)} images.')


if __name__ == '__main__':
    main()
