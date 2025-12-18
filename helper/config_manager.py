"""
Config Manager
==============

Centralized configuration management for workflow-based video processing.
Provides strict JSON schema validation, config loading/saving, and path resolution.
"""

import json
import re
from pathlib import Path
from typing import Any

from helper.stereo_core import StereoParams


__all__ = [
    'CONFIG_SCHEMA',
    'load_config',
    'save_config',
    'update_stereo_params',
    'get_path',
    'merge_cli_args',
    'find_valid_frames',
    'create_default_config',
    'ConfigError',
]

CONFIG_SCHEMA = {
    'input_video': {'type': str, 'example': 'C:/Videos/input.mkv'},
    'output_video': {'type': str, 'example': 'C:/Videos/input_stereo.mkv'},
    'directories': {
        'type': dict,
        'children': {
            'frames': {'type': str, 'example': 'frames'},
            'depth_maps': {'type': str, 'example': 'depth_maps'},
            'sbs': {'type': str, 'example': 'sbs'},
            'chunks': {'type': str, 'example': 'chunks'},
        }
    },
    'stereo': {
        'type': dict,
        'children': {
            'max_disparity': {'type': float, 'example': 50.0},
            'convergence': {'type': float, 'example': 0.0},
            'super_sampling': {'type': float, 'example': 3.0},
            'edge_softness': {'type': float, 'example': 20.0},
            'smoothing_strength': {'type': float, 'example': 1.0},
            'depth_gamma': {'type': float, 'example': 0.2},
            'sharpen': {'type': float, 'example': 14.0},
        }
    },
    'depth': {
        'type': dict,
        'children': {
            'save_16bit': {'type': bool, 'example': False},
            'batch_size': {'type': int, 'example': 1},
        }
    },
    'encoding': {
        'type': dict,
        'children': {
            'crf': {'type': int, 'example': 19},
            'preset': {'type': str, 'example': 'slow'},
        }
    },
    'free_space': {
        'type': dict,
        'children': {
            'sbs_generator': {'type': str, 'example': 'frame'},
            'chunk_generator': {'type': str, 'example': 'sbs'},
        }
    },
}


class ConfigError(Exception):
    """Raised when config validation fails."""
    pass


def _type_name(t: type) -> str:
    """Return human-readable type name."""
    type_names = {
        str: 'string',
        int: 'integer',
        float: 'float',
        bool: 'boolean',
        dict: 'object',
        list: 'array',
    }
    return type_names.get(t, t.__name__)


def _validate_value(value: Any, schema: dict, path: str, errors: dict) -> None:
    """
    Validate a single value against its schema.

    Parameters
    ----------
    value : Any
        The value to validate.
    schema : dict
        Schema definition with 'type' and optionally 'children'.
    path : str
        Dot-separated path for error messages.
    errors : dict
        Dictionary to collect errors by category.
    """
    expected_type = schema['type']

    # Special case: accept int for float fields
    if expected_type == float and isinstance(value, int) and not isinstance(value, bool):
        return

    if not isinstance(value, expected_type):
        actual_type = _type_name(type(value))
        expected = _type_name(expected_type)
        example = schema.get('example', '')
        errors.setdefault('wrong_type', []).append(f"  '{path}' (expected: {expected}, got: {actual_type}, example: {example})")


def _validate_dict(data: dict, schema: dict, path_prefix: str, errors: dict) -> None:
    """
    Recursively validate a dictionary against schema.

    Parameters
    ----------
    data : dict
        The dictionary to validate.
    schema : dict
        Schema definition with nested 'children'.
    path_prefix : str
        Current path prefix for error messages.
    errors : dict
        Dictionary to collect errors by category.
    """
    if 'children' not in schema:
        return

    children_schema = schema['children']
    data_keys = set(data.keys())
    schema_keys = set(children_schema.keys())

    # Check for missing keys
    missing = schema_keys - data_keys
    for key in missing:
        full_path = f'{path_prefix}.{key}' if path_prefix else key
        child_schema = children_schema[key]
        expected = _type_name(child_schema['type'])
        example = child_schema.get('example', '')
        errors.setdefault('missing', []).append(f"  '{full_path}' (expected: {expected}, example: {example})")

    # Check for unknown keys
    unknown = data_keys - schema_keys
    for key in unknown:
        full_path = f'{path_prefix}.{key}' if path_prefix else key
        errors.setdefault('unknown', []).append(f"  '{full_path}'")

    # Validate existing keys
    for key in data_keys & schema_keys:
        full_path = f'{path_prefix}.{key}' if path_prefix else key
        child_schema = children_schema[key]
        value = data[key]

        _validate_value(value, child_schema, full_path, errors)

        if child_schema['type'] == dict and isinstance(value, dict):
            _validate_dict(value, child_schema, full_path, errors)


def _validate_config(config: dict) -> None:
    """
    Validate config against schema.

    Parameters
    ----------
    config : dict
        The configuration dictionary to validate.

    Raises
    ------
    ConfigError
        If validation fails, with detailed error message.
    """
    errors: dict = {}

    # Check top-level structure
    config_keys = set(config.keys())
    schema_keys = set(CONFIG_SCHEMA.keys())

    # Check for missing top-level keys
    missing = schema_keys - config_keys
    for key in missing:
        child_schema = CONFIG_SCHEMA[key]
        expected = _type_name(child_schema['type'])
        example = child_schema.get('example', '')
        errors.setdefault('missing', []).append(f"  '{key}' (expected: {expected}, example: {example})")

    # Check for unknown top-level keys
    unknown = config_keys - schema_keys
    for key in unknown:
        errors.setdefault('unknown', []).append(f"  '{key}'")

    # Validate existing top-level keys
    for key in config_keys & schema_keys:
        child_schema = CONFIG_SCHEMA[key]
        value = config[key]

        _validate_value(value, child_schema, key, errors)

        if child_schema['type'] == dict and isinstance(value, dict):
            _validate_dict(value, child_schema, key, errors)

    if errors:
        msg_parts = ['Configuration validation failed:']
        if 'missing' in errors:
            msg_parts.append('Missing keys:')
            msg_parts.extend(errors['missing'])
        if 'wrong_type' in errors:
            msg_parts.append('Wrong type:')
            msg_parts.extend(errors['wrong_type'])
        if 'unknown' in errors:
            msg_parts.append('Unknown keys:')
            msg_parts.extend(errors['unknown'])

        raise ConfigError('\n'.join(msg_parts))


def create_default_config(input_video: Path) -> dict:
    """
    Create a default configuration dictionary.

    Parameters
    ----------
    input_video : Path
        Absolute path to the input video file.

    Returns
    -------
    dict
        Default configuration with all required fields.
    """
    defaults = StereoParams()
    input_stem = input_video.stem
    output_video_path = input_video.parent / f'{input_stem}_stereo.mkv'

    return {
        'input_video': str(input_video.resolve()).replace('\\', '/'),
        'output_video': str(output_video_path).replace('\\', '/'),
        'directories': {
            'frames': 'frames',
            'depth_maps': 'depth_maps',
            'sbs': 'sbs',
            'chunks': 'chunks',
        },
        'stereo': {
            'max_disparity': defaults.max_disparity,
            'convergence': defaults.convergence,
            'super_sampling': defaults.super_sampling,
            'edge_softness': defaults.edge_softness,
            'smoothing_strength': defaults.smoothing_strength,
            'depth_gamma': defaults.depth_gamma,
            'sharpen': defaults.sharpen,
        },
        'depth': {
            'save_16bit': False,
            'batch_size': 1,
        },
        'encoding': {
            'crf': 19,
            'preset': 'slow',
        },
        'free_space': {
            'sbs_generator': 'frame',
            'chunk_generator': 'sbs',
        },
    }


def load_config(workflow_path: Path) -> dict:
    """
    Load and validate config from workflow directory.

    Parameters
    ----------
    workflow_path : Path
        Path to the workflow directory containing config.json.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    ConfigError
        If config file is missing, invalid JSON, or fails validation.
    """
    config_file = Path(workflow_path) / 'config.json'

    if not config_file.exists():
        raise ConfigError(f"Config file not found: {config_file}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in config file: {e}")

    _validate_config(config)

    return config


def save_config(workflow_path: Path, config: dict) -> None:
    """
    Save config to workflow directory with tab indentation.

    Parameters
    ----------
    workflow_path : Path
        Path to the workflow directory.
    config : dict
        Configuration dictionary to save.
    """
    config_file = Path(workflow_path) / 'config.json'

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent='\t')
        f.write('\n')


def update_stereo_params(workflow_path: Path, stereo_params: dict) -> None:
    """
    Update only the stereo parameters section of config.

    Parameters
    ----------
    workflow_path : Path
        Path to the workflow directory.
    stereo_params : dict
        Dictionary with stereo parameter values to update.

    Raises
    ------
    ConfigError
        If config cannot be loaded or stereo params are invalid.
    """
    config = load_config(workflow_path)
    config['stereo'].update(stereo_params)
    _validate_config(config)
    save_config(workflow_path, config)


def get_path(workflow_path: Path, config: dict, key: str) -> Path:
    """
    Resolve a directory path from config relative to workflow.

    Parameters
    ----------
    workflow_path : Path
        Path to the workflow directory.
    config : dict
        Configuration dictionary.
    key : str
        Directory key (e.g., 'frames', 'depth_maps', 'sbs', 'chunks', 'output').

    Returns
    -------
    Path
        Absolute path to the directory.

    Raises
    ------
    KeyError
        If key is not found in directories config.
    """
    workflow_path = Path(workflow_path)

    if key == 'input_video':
        return Path(config['input_video'])

    if key == 'output_video':
        output_path = config['output_video']
        if Path(output_path).is_absolute():
            return Path(output_path)
        return workflow_path / output_path

    if key not in config['directories']:
        raise KeyError(f"Unknown directory key: {key}")

    return workflow_path / config['directories'][key]


def merge_cli_args(config: dict, cli_args: dict) -> dict:
    """
    Merge CLI arguments into config, with CLI taking precedence.

    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    cli_args : dict
        CLI argument dictionary. Keys can be dot-separated for nested values
        (e.g., 'depth.batch_size') or simple keys that map to known locations.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    result = json.loads(json.dumps(config))  # Deep copy

    # Simple key mappings to config paths
    key_mappings = {
        'batch_size': ('depth', 'batch_size'),
        'save_16bit': ('depth', 'save_16bit'),
        'crf': ('encoding', 'crf'),
        'preset': ('encoding', 'preset'),
        'max_disparity': ('stereo', 'max_disparity'),
        'convergence': ('stereo', 'convergence'),
        'super_sampling': ('stereo', 'super_sampling'),
        'edge_softness': ('stereo', 'edge_softness'),
        'smoothing_strength': ('stereo', 'smoothing_strength'),
        'depth_gamma': ('stereo', 'depth_gamma'),
        'sharpen': ('stereo', 'sharpen'),
    }

    for key, value in cli_args.items():
        if value is None:
            continue

        if key in key_mappings:
            section, param = key_mappings[key]
            result[section][param] = value

    return result


def find_valid_frames(workflow_path: Path, config: dict) -> list[int]:
    """
    Find frame numbers where both frame and depth map exist.

    Depth map preference order: .tif first, then .png.

    Parameters
    ----------
    workflow_path : Path
        Path to the workflow directory.
    config : dict
        Configuration dictionary.

    Returns
    -------
    list of int
        Sorted list of valid frame numbers.
    """
    workflow_path = Path(workflow_path)
    frames_dir = workflow_path / config['directories']['frames']
    depth_dir = workflow_path / config['directories']['depth_maps']

    if not frames_dir.exists() or not depth_dir.exists():
        return []

    # Find all frame numbers in frames directory
    frame_pattern = re.compile(r'^frame_(\d+)\.png$')
    frame_numbers = set()

    for f in frames_dir.iterdir():
        if f.is_file():
            match = frame_pattern.match(f.name)
            if match:
                frame_numbers.add(int(match.group(1)))

    # Find depth map numbers (prefer .tif, fallback to .png)
    depth_pattern = re.compile(r'^depth_frame_(\d+)\.(tif|png)$')
    depth_numbers = set()

    for f in depth_dir.iterdir():
        if f.is_file():
            match = depth_pattern.match(f.name)
            if match:
                depth_numbers.add(int(match.group(1)))

    # Return intersection sorted
    valid = frame_numbers & depth_numbers
    return sorted(valid)


def get_frame_paths(workflow_path: Path, config: dict, frame_num: int) -> tuple[Path, Path] | None:
    """
    Get frame and depth map paths for a specific frame number.

    Parameters
    ----------
    workflow_path : Path
        Path to the workflow directory.
    config : dict
        Configuration dictionary.
    frame_num : int
        Frame number to look up.

    Returns
    -------
    tuple of Path or None
        (frame_path, depth_path) if both exist, None otherwise.
    """
    workflow_path = Path(workflow_path)
    frames_dir = workflow_path / config['directories']['frames']
    depth_dir = workflow_path / config['directories']['depth_maps']

    frame_path = frames_dir / f'frame_{frame_num:06d}.png'
    if not frame_path.exists():
        return None

    # Prefer .tif, fallback to .png
    depth_path = depth_dir / f'depth_frame_{frame_num:06d}.tif'
    if not depth_path.exists():
        depth_path = depth_dir / f'depth_frame_{frame_num:06d}.png'
        if not depth_path.exists():
            return None

    return frame_path, depth_path
