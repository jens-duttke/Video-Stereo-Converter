#!/usr/bin/env python3
"""
Workflow Initializer
====================

Creates a new workflow directory with standardized structure and default configuration.

Usage:
    python workflow_init.py --input-video "C:/Videos/input.mkv"
    python workflow_init.py --input-video "C:/Videos/input.mkv" --workflow-dir "D:/Video-Processing/workflow"
"""

import helper.terminal_title  # noqa: F401 # pyright: ignore[reportUnusedImport]

import argparse
import sys
from pathlib import Path

from helper.config_manager import create_default_config, save_config


def main():
    parser = argparse.ArgumentParser(
        description='Initialize a new workflow directory with default configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python workflow_init.py --input-video "C:/Videos/input.mkv"\n'
            '  python workflow_init.py --input-video "C:/Videos/input.mkv" --workflow-dir "D:/Video-Processing/workflow"\n'
            '\n'
            'If --workflow-dir is not specified, creates a "workflow" subdirectory next to the input video.\n'
        )
    )
    parser.add_argument('--input-video', type=Path, required=True, help='Path to the input video file (absolute path will be stored in config)')
    parser.add_argument('--workflow-dir', type=Path, default=None, help='Path to create the workflow directory (default: workflow/ next to input video)')

    args = parser.parse_args()

    # Validate input video exists
    input_video = args.input_video.resolve()
    if not input_video.is_file():
        print(f'ERROR: Input video does not exist: {input_video}')
        sys.exit(1)

    # Default workflow_dir to 'workflow' subdirectory next to input video
    if args.workflow_dir is None:
        workflow_dir = input_video.parent / 'workflow'
    else:
        workflow_dir = args.workflow_dir.resolve()

    # Check if workflow already exists
    config_file = workflow_dir / 'config.json'
    if config_file.exists():
        print(f'ERROR: Workflow already initialized: {config_file}')
        sys.exit(1)

    # Create workflow directory structure
    subdirs = ['frames', 'depth_maps', 'sbs', 'chunks']

    print(f'Creating workflow directory: {workflow_dir}')
    workflow_dir.mkdir(parents=True, exist_ok=True)

    for subdir in subdirs:
        subdir_path = workflow_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f'  Created: {subdir}/')

    # Create default config
    config = create_default_config(input_video)
    save_config(workflow_dir, config)
    print(f'  Created: config.json')

    print()
    print(f'Workflow initialized successfully!')
    print(f'  Input video: {input_video}')
    print(f'  Output video: {config["output_video"]}')
    print()
    print('Next steps:')
    print(f'  1. Extract frames:     python frame_extractor.py "{workflow_dir}"')
    print(f'  2. Generate depth:     python depth_map_generator.py "{workflow_dir}"')
    print(f'  3. Test settings:      python sbs_tester.py "{workflow_dir}"')
    print(f'  4. Generate SBS:       python sbs_generator.py "{workflow_dir}"')
    print(f'  5. Create chunks:      python chunk_generator.py "{workflow_dir}"')
    print(f'  6. Concatenate:        python video_concatenator.py "{workflow_dir}"')


if __name__ == '__main__':
    main()
