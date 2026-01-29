"""
Workflow State Manager
======================

Manages workflow orchestration state with YAML persistence.
Provides atomic load/save operations and step status management.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

from helper.config_manager import ConfigError, get_path, load_config
from helper.workflow_metrics import is_all_chunks_complete


__all__ = [
    'StepStatus',
    'STEP_ORDER',
    'PERSISTENT_STEPS',
    'MUTEX_STEPS',
    'get_step_status',
    'set_step_pending',
    'set_step_running',
    'set_step_done',
    'set_step_failed',
    'set_step_error',
    'load_workflows',
    'save_workflows',
]


class StepStatus:
    """Step status constants."""
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    DONE = 'DONE'
    FAILED = 'FAILED'
    ERROR = 'ERROR'


STEP_ORDER = ['frame_extractor', 'depth_map_generator', 'sbs_generator', 'chunk_generator', 'video_concatenator']
PERSISTENT_STEPS = ['frame_extractor', 'depth_map_generator', 'sbs_generator']
MUTEX_STEPS = ['frame_extractor', 'chunk_generator', 'video_concatenator']


def normalize_path(path: Path | str) -> str:
    """
    Normalize a path to absolute form with forward slashes.

    Parameters
    ----------
    path : Path or str
        Path to normalize.

    Returns
    -------
    str
        Normalized absolute path string with forward slashes.
    """
    return str(Path(path).resolve()).replace('\\', '/')


def get_step_status(step_value: str | dict | None) -> str:
    """
    Extract status string from step value.

    Parameters
    ----------
    step_value : str, dict, or None
        Step value from YAML (either status string or dict with 'status' key).

    Returns
    -------
    str
        Status string (PENDING, RUNNING, DONE, FAILED, ERROR).
    """
    if step_value is None:
        return StepStatus.PENDING

    if isinstance(step_value, str):
        return step_value

    return step_value.get('status', StepStatus.PENDING)





def set_step_pending(workflow: dict, step_name: str) -> None:
    """
    Set step to PENDING status.

    Parameters
    ----------
    workflow : dict
        Workflow dictionary (step_name -> status).
    step_name : str
        Name of the step.
    """
    workflow[step_name] = StepStatus.PENDING


def set_step_running(workflow: dict, step_name: str) -> None:
    """
    Set step to RUNNING status.

    Parameters
    ----------
    workflow : dict
        Workflow dictionary (step_name -> status).
    step_name : str
        Name of the step.
    """
    workflow[step_name] = StepStatus.RUNNING


def set_step_done(workflow: dict, step_name: str) -> None:
    """
    Set step to DONE status.

    Parameters
    ----------
    workflow : dict
        Workflow dictionary (step_name -> status).
    step_name : str
        Name of the step.
    """
    workflow[step_name] = StepStatus.DONE


def set_step_failed(workflow: dict, step_name: str) -> None:
    """
    Set step to FAILED status (first failure, will be retried).

    Parameters
    ----------
    workflow : dict
        Workflow dictionary (step_name -> status).
    step_name : str
        Name of the step.
    """
    workflow[step_name] = StepStatus.FAILED


def set_step_error(workflow: dict, step_name: str) -> None:
    """
    Set step to ERROR status (repeated failure, needs manual intervention).

    Parameters
    ----------
    workflow : dict
        Workflow dictionary (step_name -> status).
    step_name : str
        Name of the step.
    """
    workflow[step_name] = StepStatus.ERROR


def _create_default_workflow() -> dict:
    """Create a new workflow state dictionary with default values."""
    return {step: StepStatus.PENDING for step in PERSISTENT_STEPS}


def _migrate_workflow(workflow: dict | str | None) -> dict:
    """
    Migrate workflow format and handle status transitions.

    Parameters
    ----------
    workflow : dict, str, or None
        Existing workflow data, 'DONE' string for completed workflows, or None.

    Returns
    -------
    dict
        Migrated workflow dictionary.
    """
    if workflow is None:
        return _create_default_workflow()

    # Handle simplified 'DONE' format for completed workflows
    if workflow == StepStatus.DONE:
        return {step: StepStatus.DONE for step in STEP_ORDER}

    # Handle old 'steps' format - flatten it
    if 'steps' in workflow:
        steps = workflow['steps']
        workflow = steps

    # Ensure all steps exist
    for step in STEP_ORDER:
        if step not in workflow:
            workflow[step] = StepStatus.PENDING

    # Reset FAILED to PENDING for retry on restart
    # Note: RUNNING is kept as-is - orchestrator will prioritize these for restart
    for step_name in STEP_ORDER:
        step_value = workflow.get(step_name)
        status = get_step_status(step_value)

        if status == StepStatus.FAILED:
            set_step_pending(workflow, step_name)

    # Remove legacy fields
    workflow.pop('retry_count', None)
    workflow.pop('last_updated', None)

    return workflow


def load_workflows(yaml_path: Path) -> dict[str, dict]:
    """
    Load workflows from YAML file with migration support.

    Parameters
    ----------
    yaml_path : Path
        Path to workflows.yaml file.

    Returns
    -------
    dict
        Dictionary mapping normalized workflow paths to workflow state dicts.

    Examples
    --------
    Minimal YAML format:

        H:/test21/workflow:
        H:/test22/workflow:

    Full format:

        H:/test21/workflow:
          frame_extractor: DONE
          depth_map_generator: RUNNING
          sbs_generator: PENDING

    Completed workflow (simplified format):

        H:/test21/workflow: DONE
    """
    if not yaml_path.exists():
        return {}

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f'Invalid workflows file format: expected dict, got {type(data).__name__}')

    workflows: dict[str, dict] = {}

    for path, workflow in data.items():
        normalized = normalize_path(path)
        migrated = _migrate_workflow(workflow)
        workflows[normalized] = migrated

    return workflows


def _merge_workflows(current: dict[str, dict], from_file: dict[str, dict]) -> dict[str, dict]:
    """
    Merge workflows from file with current orchestrator state.

    Preserves manual edits from the file while applying orchestrator status changes.
    New workflows from the file are added, existing workflows keep their
    orchestrator-managed status while preserving any extra fields from the file.
    The original order from the file is preserved, with new workflows appended at the end.

    Parameters
    ----------
    current : dict
        Current orchestrator state (workflow_path -> workflow_dict).
    from_file : dict
        Freshly loaded state from YAML file.

    Returns
    -------
    dict
        Merged workflows dictionary preserving original file order.
    """
    merged: dict[str, dict] = {}

    # First, process workflows in the order they appear in the file
    for path in from_file:
        file_wf = from_file[path]
        current_wf = current.get(path)

        if current_wf is None:
            # Workflow only in file - use file version (migrated)
            migrated = _migrate_workflow(file_wf)
            merged[path] = migrated
        else:
            # Handle simplified 'DONE' format from file
            if file_wf == StepStatus.DONE:
                file_wf = {step: StepStatus.DONE for step in STEP_ORDER}

            # Workflow exists in both - merge step by step
            merged_wf = {}

            # Start with file version to preserve any extra fields
            for key, value in file_wf.items():
                if key not in STEP_ORDER:
                    # Preserve non-step fields from file
                    merged_wf[key] = value

            # Apply orchestrator status for known steps
            # Skip transient steps (chunk_generator, video_concatenator) - they are derived from filesystem
            for step in STEP_ORDER:
                if step in ('chunk_generator', 'video_concatenator'):
                    # Transient steps are not stored - always default to PENDING
                    # Their actual status is determined dynamically from filesystem
                    merged_wf[step] = StepStatus.PENDING
                elif step in current_wf:
                    # Use orchestrator status (it has the authoritative runtime state)
                    merged_wf[step] = current_wf[step]
                elif step in file_wf:
                    # Step only in file (manual edit) - use file version
                    merged_wf[step] = file_wf[step]
                else:
                    # Step missing in both - use default
                    merged_wf[step] = StepStatus.PENDING

            merged[path] = merged_wf

    return merged


def _is_workflow_complete(workflow_path: str, workflow: dict) -> bool:
    """
    Check if a workflow is fully complete (output video exists or config missing).

    Parameters
    ----------
    workflow_path : str
        Path to the workflow directory.
    workflow : dict
        Workflow dictionary (step_name -> status).

    Returns
    -------
    bool
        True if output video exists or workflow config is missing, False otherwise.
    """
    # Check persistent steps first (quick check)
    for step in PERSISTENT_STEPS:
        step_value = workflow.get(step)
        if get_step_status(step_value) != StepStatus.DONE:
            return False

    # Check if output video exists (definitive check)
    try:
        path = Path(workflow_path)
        config = load_config(path)
        output_video = get_path(path, config, 'output_video')
        return output_video.exists()
    except (ConfigError, OSError):
        # Config missing or unreadable - workflow is complete (removed/cleaned up)
        return True


def _filter_persistent_steps(workflows: dict[str, dict]) -> dict[str, str | dict]:
    """
    Filter workflows to only include persistent steps for YAML output.

    Removes transient steps (chunk_generator, video_concatenator) as they are
    derived from filesystem. Simplifies completed workflows to 'DONE' string.

    Parameters
    ----------
    workflows : dict
        Dictionary mapping workflow paths to state dicts.

    Returns
    -------
    dict
        Dictionary with only persistent steps or 'DONE' for completed workflows.
    """
    result: dict[str, str | dict] = {}

    for path, workflow in workflows.items():
        if _is_workflow_complete(path, workflow):
            result[path] = StepStatus.DONE
        else:
            # Only include persistent steps
            filtered = {step: workflow[step] for step in PERSISTENT_STEPS if step in workflow}
            result[path] = filtered

    return result


def save_workflows(yaml_path: Path, workflows: dict[str, dict]) -> dict[str, dict]:
    """
    Atomically save workflows to YAML file with merge support.

    Before saving, re-reads the file and merges any manual changes.
    This preserves manual edits made during runtime while applying
    the orchestrator's status updates. Only persistent steps are saved;
    transient steps are derived from filesystem at runtime.

    Parameters
    ----------
    yaml_path : Path
        Path to workflows.yaml file.
    workflows : dict
        Dictionary mapping workflow paths to state dicts.

    Returns
    -------
    dict
        The merged workflows dictionary that was saved.
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Re-read file to capture any manual changes
    from_file: dict[str, dict] = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                for path, workflow in data.items():
                    from_file[normalize_path(path)] = workflow if workflow else {}
        except (yaml.YAMLError, OSError):
            # If reading fails, proceed with current state only
            pass

    # Merge current state with file state
    merged = _merge_workflows(workflows, from_file)

    # Filter to persistent steps only for YAML output
    yaml_output = _filter_persistent_steps(merged)

    fd, temp_path = tempfile.mkstemp(dir=yaml_path.parent, suffix='.yaml')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            yaml.safe_dump(yaml_output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        os.replace(temp_path, yaml_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise

    return merged
