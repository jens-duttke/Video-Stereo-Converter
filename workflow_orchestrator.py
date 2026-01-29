#!/usr/bin/env python3
"""
Workflow Orchestrator
=====================

Automatically orchestrates multiple video processing workflows.
Manages parallel execution of depth_map_generator, sbs_generator, and sequential
execution of frame_extractor, chunk_generator, and video_concatenator.

Usage:
    python workflow_orchestrator.py workflows.yaml
    python workflow_orchestrator.py workflows.yaml --validate-only
"""

from __future__ import annotations

import helper.utf8_console  # noqa: F401 # pyright: ignore[reportUnusedImport]

import asyncio
import os
import signal
import sys
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import psutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from helper.config_manager import ConfigError, get_path, load_config
from helper.workflow_metrics import (
    DISK_SPACE_THRESHOLD_GB,
    MIN_DEPTH_FOR_SBS,
    get_depth_count,
    get_last_chunk_end_frame,
    get_max_depth_number,
    get_max_sbs_number,
    get_next_chunk_end_frame,
    get_video_progress,
    invalidate_cache,
    is_all_chunks_complete,
)
from helper.workflow_state import (
    MUTEX_STEPS,
    PERSISTENT_STEPS,
    STEP_ORDER,
    StepStatus,
    get_step_status,
    load_workflows,
    save_workflows,
    set_step_done,
    set_step_error,
    set_step_failed,
    set_step_pending,
    set_step_running,
)
from helper.terminal_title import set_terminal_title


# Constants
SCHEDULER_INTERVAL = 5.0
FALLBACK_CHECK_INTERVAL = 3600.0
PREFETCH_WORKFLOWS = 2
GPU_COOLDOWN_SECONDS = 30.0  # Wait time after GPU failure before starting new GPU processes

# Concurrency limits
MAX_DEPTH_PROCESSES = 1
MAX_SBS_PROCESSES = 2
MAX_MUTEX_PROCESSES = 1


def _save_and_sync(state: 'OrchestratorState') -> None:
    """
    Save workflows and synchronize state with merged result.

    This ensures that manual edits to workflows.yaml (e.g., adding new workflows)
    are picked up by the orchestrator during runtime.

    Parameters
    ----------
    state : OrchestratorState
        Orchestrator state to synchronize.
    """
    merged = save_workflows(state.yaml_path, state.workflows)
    state.workflows.clear()
    state.workflows.update(merged)


@dataclass
class ProcessInfo:
    """Information about a running subprocess."""
    workflow_path: str
    step_name: str
    process: asyncio.subprocess.Process
    pid: int = 0
    task: asyncio.Task | None = None
    output_task: asyncio.Task | None = None
    output_lines: list[str] = field(default_factory=list)
    stderr_buffer: str = ''
    last_progress_line: str = ''  # Current progress bar line


@dataclass
class OrchestratorState:
    """Global orchestrator state."""
    workflows: dict[str, dict]
    yaml_path: Path
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    active_processes: dict[str, ProcessInfo] = field(default_factory=dict)
    schedule_needed: asyncio.Event = field(default_factory=asyncio.Event)  # Single event to trigger scheduling
    console: Console = field(default_factory=Console)
    console_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    live: Live | None = None  # Live display for progress bars
    log_messages: list[str] = field(default_factory=list)  # Recent log messages
    gpu_cooldown_until: float = 0.0  # Timestamp until which GPU processes should not start


def _build_live_display(state: OrchestratorState) -> Group:
    """Build the live display showing all active processes and recent logs."""
    elements = []

    # Short names for step types
    step_short_names = {
        'frame_extractor': 'Frame',
        'depth_map_generator': 'Depth',
        'sbs_generator': 'SBS',
        'chunk_generator': 'Chunk',
        'video_concatenator': 'Concat',
    }

    # Recent log messages (non-progress output)
    if state.log_messages:
        log_lines = []
        for msg in state.log_messages[-10:]:  # Show last 10 messages
            log_lines.append(msg)
        # Use Text.from_markup to properly interpret Rich markup
        log_text = Text.from_markup('\n'.join(log_lines))
        elements.append(log_text)

    # Active processes with progress bars
    if state.active_processes:
        progress_lines = []
        for process_key, process_info in state.active_processes.items():
            step_short = step_short_names.get(process_info.step_name, process_info.step_name)
            workflow_name = _get_workflow_name(process_info.workflow_path)
            progress = process_info.last_progress_line or 'Starting...'
            progress_lines.append(f'[cyan][{step_short}|{workflow_name}][/cyan] {progress}')

        if progress_lines:
            elements.append(Panel('\n'.join(progress_lines), title='Active Processes', border_style='blue'))

    if not elements:
        return Group(Text('No active processes'))

    return Group(*elements)


def _get_disk_space_gb(path: Path) -> float:
    """Get free disk space in GB for the given path."""
    try:
        usage = psutil.disk_usage(str(path))
        return usage.free / (1024 ** 3)
    except (OSError, FileNotFoundError):
        return 0.0


def _format_timestamp() -> str:
    """Format current timestamp for log messages."""
    return datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')


def _get_workflow_name(workflow_path: str) -> str:
    """Extract a short name from workflow path for display."""
    path = Path(workflow_path)
    return path.parent.name if path.name == 'workflow' else path.name


def _count_active_by_step(state: OrchestratorState, step_name: str) -> int:
    """Count active processes for a specific step type."""
    return sum(1 for p in state.active_processes.values() if p.step_name == step_name)


def _count_active_mutex(state: OrchestratorState) -> int:
    """Count active mutex processes (frame_extractor, chunk_generator, video_concatenator)."""
    return sum(1 for p in state.active_processes.values() if p.step_name in MUTEX_STEPS)


def _fix_stale_sbs_status(state: OrchestratorState) -> bool:
    """
    Fix SBS workflows with PENDING or RUNNING status that should be DONE.

    When sbs_generator is PENDING or RUNNING but no active process, and all SBS
    frames are generated (max_sbs >= max_depth > 0), set status to DONE.

    Returns
    -------
    bool
        True if any status was fixed.
    """
    fixed = False

    for workflow_path, workflow in state.workflows.items():
        sbs_status = get_step_status(workflow.get('sbs_generator', StepStatus.PENDING))

        # Only fix if SBS is PENDING or RUNNING without active process
        if sbs_status not in (StepStatus.PENDING, StepStatus.RUNNING):
            continue

        process_key = f'{workflow_path}:sbs_generator'
        if process_key in state.active_processes:
            continue

        # Check if all SBS frames are generated
        path = Path(workflow_path)
        max_depth = get_max_depth_number(path)
        max_sbs = get_max_sbs_number(path)

        if max_sbs >= max_depth > 0:
            set_step_done(workflow, 'sbs_generator')
            fixed = True

    return fixed


def _fix_stale_depth_status(state: OrchestratorState) -> bool:
    """
    Fix depth workflows with PENDING status that should be RUNNING.

    When a workflow has depth_map_generator=PENDING but depth maps already exist
    (from a previous run), set status to RUNNING so it gets prioritized for restart.

    Returns
    -------
    bool
        True if any status was fixed.
    """
    fixed = False

    for workflow_path, workflow in state.workflows.items():
        depth_status = get_step_status(workflow.get('depth_map_generator', StepStatus.PENDING))

        # Only fix if depth is PENDING
        if depth_status != StepStatus.PENDING:
            continue

        # Check if depth maps already exist
        path = Path(workflow_path)
        depth_count = get_depth_count(path)

        if depth_count > 0:
            set_step_running(workflow, 'depth_map_generator')
            fixed = True

    return fixed


def _validate_workflow(workflow_path: str) -> tuple[bool, str]:
    """
    Validate a workflow directory and config.

    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    path = Path(workflow_path)

    if not path.is_dir():
        return False, f'Workflow directory does not exist: {workflow_path}'

    try:
        config = load_config(path)
    except ConfigError as e:
        return False, f'Config error: {e}'

    input_video = get_path(path, config, 'input_video')
    if not input_video.is_file():
        return False, f'Input video not found: {input_video}'

    return True, ''


def _is_all_steps_done(workflow: dict) -> bool:
    """Check if all steps in a workflow are marked as DONE."""
    return all(
        get_step_status(workflow.get(step, StepStatus.PENDING)) == StepStatus.DONE
        for step in STEP_ORDER
    )


def _is_workflow_complete(workflow: dict, workflow_path: str) -> bool:
    """
    Check if a workflow is complete (no more work to do).

    A workflow is complete when:
    - Any persistent step has ERROR status (permanent failure), OR
    - All persistent steps are DONE and the output video exists

    Parameters
    ----------
    workflow : dict
        Workflow dictionary (step_name -> status).
    workflow_path : str
        Path to the workflow directory.
    """
    # Check persistent steps
    all_done = True
    for step in PERSISTENT_STEPS:
        status = get_step_status(workflow.get(step, StepStatus.PENDING))

        # ERROR means permanent failure - workflow is complete (nothing more can be done)
        if status == StepStatus.ERROR:
            return True

        # PENDING or RUNNING or FAILED means work still pending
        if status != StepStatus.DONE:
            all_done = False

    if not all_done:
        return False

    # All persistent steps are DONE - check if output video exists
    path = Path(workflow_path)
    try:
        config = load_config(path)
        output_video = get_path(path, config, 'output_video')
        return output_video.exists()
    except ConfigError:
        # If config is missing, consider workflow complete (error state)
        return True


def _are_all_workflows_complete(state: OrchestratorState) -> bool:
    """
    Check if all workflows are complete and no active processes remain.

    Returns
    -------
    bool
        True if all workflows are in a terminal state and no processes are running.
    """
    if state.active_processes:
        return False

    return all(
        _is_workflow_complete(wf, wf_path)
        for wf_path, wf in state.workflows.items()
    )


def _validate_all_workflows(state: OrchestratorState) -> bool:
    """Validate all workflows and update state for invalid ones."""
    all_valid = True

    for workflow_path, workflow in state.workflows.items():
        # Skip validation for fully completed workflows
        if _is_all_steps_done(workflow):
            continue

        is_valid, error_msg = _validate_workflow(workflow_path)

        if not is_valid:
            state.console.print(f'[red]ERROR[/red]: {error_msg}')
            all_valid = False

            # Mark first pending step as error
            for step_name in STEP_ORDER:
                status = get_step_status(workflow.get(step_name, StepStatus.PENDING))
                if status == StepStatus.PENDING:
                    set_step_error(workflow, step_name)
                    break

    return all_valid


def _can_start_depth(state: OrchestratorState, workflow_path: str, workflow: dict) -> bool:
    """Check if depth_map_generator can start for this workflow."""
    # Check GPU cooldown after driver failure
    if time.time() < state.gpu_cooldown_until:
        return False

    # Check concurrency limit (only active processes count)
    if _count_active_by_step(state, 'depth_map_generator') >= MAX_DEPTH_PROCESSES:
        return False

    # Check frame_extractor is done
    frame_status = get_step_status(workflow.get('frame_extractor', StepStatus.PENDING))
    if frame_status != StepStatus.DONE:
        return False

    # Check depth status - allow PENDING, FAILED (retry), or RUNNING (restart)
    depth_status = get_step_status(workflow.get('depth_map_generator', StepStatus.PENDING))
    if depth_status in (StepStatus.DONE, StepStatus.ERROR):
        return False

    # If RUNNING, only allow if no active process (needs restart)
    if depth_status == StepStatus.RUNNING:
        process_key = f'{workflow_path}:depth_map_generator'
        if process_key in state.active_processes:
            return False

    return True


def _can_start_sbs(state: OrchestratorState, workflow_path: str, workflow: dict) -> bool:
    """Check if sbs_generator can start for this workflow."""
    # Check GPU cooldown after driver failure
    if time.time() < state.gpu_cooldown_until:
        return False

    # Check concurrency limit (only active processes count)
    if _count_active_by_step(state, 'sbs_generator') >= MAX_SBS_PROCESSES:
        return False

    # Check already running for this workflow (with active process)
    process_key = f'{workflow_path}:sbs_generator'
    if process_key in state.active_processes:
        return False

    # Check depth is running or done
    depth_status = get_step_status(workflow.get('depth_map_generator', StepStatus.PENDING))
    if depth_status not in (StepStatus.RUNNING, StepStatus.DONE):
        return False

    # Check depth count threshold (only enforce minimum while depth is still running)
    # When depth is complete, process all frames even if video is short
    path = Path(workflow_path)
    depth_count = get_depth_count(path)
    if depth_status != StepStatus.DONE and depth_count < MIN_DEPTH_FOR_SBS:
        return False

    # Check SBS status - allow PENDING, FAILED (retry), or RUNNING (restart)
    sbs_status = get_step_status(workflow.get('sbs_generator', StepStatus.PENDING))
    if sbs_status in (StepStatus.DONE, StepStatus.ERROR):
        return False

    # Check if SBS has caught up with depth (regardless of depth status)
    max_depth = get_max_depth_number(path)
    max_sbs = get_max_sbs_number(path)
    if max_depth > 0 and max_sbs >= max_depth:
        return False

    return True


def _can_start_chunk(state: OrchestratorState, workflow_path: str, workflow: dict) -> bool:
    """Check if chunk_generator can start for this workflow."""
    # Check mutex limit
    if _count_active_mutex(state) >= MAX_MUTEX_PROCESSES:
        return False

    # Check already running for this workflow
    process_key = f'{workflow_path}:chunk_generator'
    if process_key in state.active_processes:
        return False

    # Check SBS count threshold and if more chunks are needed (filesystem check)
    path = Path(workflow_path)
    last_chunk_end = get_last_chunk_end_frame(path)
    sbs_status = get_step_status(workflow.get('sbs_generator', StepStatus.PENDING))
    sbs_complete = sbs_status == StepStatus.DONE
    next_end = get_next_chunk_end_frame(path, last_chunk_end, sbs_complete)

    if next_end is None:
        return False

    return True


def _can_start_concat(state: OrchestratorState, workflow_path: str, workflow: dict) -> bool:
    """Check if video_concatenator can start for this workflow."""
    # Check mutex limit
    if _count_active_mutex(state) >= MAX_MUTEX_PROCESSES:
        return False

    # Check already running for this workflow
    process_key = f'{workflow_path}:video_concatenator'
    if process_key in state.active_processes:
        return False

    # Check sbs_generator is done
    sbs_status = get_step_status(workflow.get('sbs_generator', StepStatus.PENDING))
    if sbs_status != StepStatus.DONE:
        return False

    # Check all chunks are complete (filesystem check)
    path = Path(workflow_path)
    if not is_all_chunks_complete(path):
        return False

    # Check if output video already exists (filesystem check)
    try:
        config = load_config(path)
        output_video = get_path(path, config, 'output_video')
        if output_video.exists():
            return False
    except Exception:
        pass

    return True


def _can_start_frame_extractor(state: OrchestratorState, workflow_path: str, workflow: dict) -> bool:
    """Check if frame_extractor can start for this workflow."""
    # Check mutex limit
    if _count_active_mutex(state) >= MAX_MUTEX_PROCESSES:
        return False

    # Check frame_extractor status - allow PENDING, FAILED (retry), or RUNNING (restart)
    frame_status = get_step_status(workflow.get('frame_extractor', StepStatus.PENDING))
    if frame_status in (StepStatus.DONE, StepStatus.ERROR):
        return False

    # If RUNNING, only allow if no active process (needs restart after crash)
    if frame_status == StepStatus.RUNNING:
        process_key = f'{workflow_path}:frame_extractor'
        if process_key in state.active_processes:
            return False

    return True


def _get_prefetch_candidates(state: OrchestratorState) -> list[str]:
    """
    Get workflow paths that need frame extraction for prefetch.

    Prefetches frames for workflows starting from the top of the list,
    prioritizing workflows before and up to PREFETCH_WORKFLOWS positions
    after the currently running or next-to-run depth_map_generator.
    """
    # Build ordered list of workflow paths
    workflow_paths = list(state.workflows.keys())

    # First, collect ALL startable frame_extractor workflows from top to bottom
    # Include PENDING and RUNNING (without active process, needs restart)
    startable_frame_workflows = []
    for workflow_path in workflow_paths:
        workflow = state.workflows[workflow_path]
        frame_status = get_step_status(workflow.get('frame_extractor', StepStatus.PENDING))
        if frame_status == StepStatus.PENDING:
            startable_frame_workflows.append(workflow_path)
        elif frame_status == StepStatus.RUNNING:
            # RUNNING without active process needs restart
            process_key = f'{workflow_path}:frame_extractor'
            if process_key not in state.active_processes:
                startable_frame_workflows.append(workflow_path)

    if not startable_frame_workflows:
        return []

    # Find the position of the current/next depth workflow
    depth_position = -1

    for i, workflow_path in enumerate(workflow_paths):
        workflow = state.workflows[workflow_path]
        depth_status = get_step_status(workflow.get('depth_map_generator', StepStatus.PENDING))

        # Currently running depth
        if depth_status == StepStatus.RUNNING:
            depth_position = i
            break

        # First pending depth that could run (frame_extractor must be done)
        if depth_status == StepStatus.PENDING:
            frame_status = get_step_status(workflow.get('frame_extractor', StepStatus.PENDING))
            if frame_status == StepStatus.DONE:
                depth_position = i
                break

    # If no depth running or ready, find first workflow needing depth
    if depth_position == -1:
        for i, workflow_path in enumerate(workflow_paths):
            workflow = state.workflows[workflow_path]
            depth_status = get_step_status(workflow.get('depth_map_generator', StepStatus.PENDING))
            if depth_status == StepStatus.PENDING:
                depth_position = i
                break

    # If still no depth position found, just return first startable from top
    if depth_position == -1:
        return startable_frame_workflows[:PREFETCH_WORKFLOWS + 1]

    # Build candidate set: all workflows from position 0 up to depth_position + PREFETCH_WORKFLOWS
    max_position = min(depth_position + PREFETCH_WORKFLOWS + 1, len(workflow_paths))
    candidates = []
    for i in range(max_position):
        workflow_path = workflow_paths[i]
        workflow = state.workflows[workflow_path]
        frame_status = get_step_status(workflow.get('frame_extractor', StepStatus.PENDING))
        if frame_status == StepStatus.PENDING:
            candidates.append(workflow_path)
        elif frame_status == StepStatus.RUNNING:
            # RUNNING without active process needs restart
            process_key = f'{workflow_path}:frame_extractor'
            if process_key not in state.active_processes:
                candidates.append(workflow_path)

    return candidates


def _build_command(step_name: str, workflow_path: str, workflow: dict) -> list[str]:
    """Build the command to run for a step."""
    python_exe = sys.executable
    script_map = {
        'frame_extractor': 'frame_extractor.py',
        'depth_map_generator': 'depth_map_generator.py',
        'sbs_generator': 'sbs_generator.py',
        'chunk_generator': 'chunk_generator.py',
        'video_concatenator': 'video_concatenator.py',
    }

    script = script_map.get(step_name)
    if not script:
        return []

    cmd = [python_exe, script, workflow_path]

    # Add --no-interactive for scripts that may wait for user input
    if step_name in ('depth_map_generator', 'sbs_generator'):
        cmd.append('--no-interactive')

    # Add --end-frame for chunk_generator
    if step_name == 'chunk_generator':
        path = Path(workflow_path)
        last_chunk_end = get_last_chunk_end_frame(path)
        sbs_status = get_step_status(workflow.get('sbs_generator', StepStatus.PENDING))
        sbs_complete = sbs_status == StepStatus.DONE
        next_end = get_next_chunk_end_frame(path, last_chunk_end, sbs_complete)
        if next_end is not None:
            cmd.extend(['--end-frame', str(next_end)])

    return cmd


async def _read_output(state: OrchestratorState, process_info: ProcessInfo) -> None:
    """Read and display output from subprocess, handling progress bars."""
    label = f'{process_info.step_name}|{_get_workflow_name(process_info.workflow_path)}'
    buffer = b''

    while True:
        try:
            # Read chunks to handle both \n and \r (for progress bars)
            chunk = await process_info.process.stdout.read(1024)
            if not chunk:
                break

            buffer += chunk

            # Split on both \n and \r
            while b'\n' in buffer or b'\r' in buffer:
                # Find earliest line ending
                n_pos = buffer.find(b'\n')
                r_pos = buffer.find(b'\r')

                if n_pos == -1:
                    split_pos = r_pos
                    is_carriage_return = True
                elif r_pos == -1:
                    split_pos = n_pos
                    is_carriage_return = False
                else:
                    if r_pos < n_pos:
                        split_pos = r_pos
                        is_carriage_return = True
                    else:
                        split_pos = n_pos
                        is_carriage_return = False

                line = buffer[:split_pos]
                buffer = buffer[split_pos + 1:]

                if line:
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    if decoded:
                        process_info.output_lines.append(decoded)

                        # Keep last 50 lines for display
                        if len(process_info.output_lines) > 50:
                            process_info.output_lines.pop(0)

                        # Capture for error reporting
                        process_info.stderr_buffer = (process_info.stderr_buffer + decoded + '\n')[-500:]

                        # Always update last_progress_line so we always show something
                        process_info.last_progress_line = decoded

                        # Non-progress output also goes to log
                        # Progress bars typically use \r or contain % or it/s
                        if not is_carriage_return and '%' not in decoded and 'it/s' not in decoded:
                            async with state.console_lock:
                                state.log_messages.append(f'{_format_timestamp()} [cyan][{label}][/cyan] {decoded}')
                                # Keep last 20 log messages
                                if len(state.log_messages) > 20:
                                    state.log_messages.pop(0)

        except asyncio.CancelledError:
            break
        except Exception:
            break

    # Clear progress line when done
    process_info.last_progress_line = ''


async def _handle_gpu_failure(state: OrchestratorState) -> None:
    """
    Handle GPU failure by terminating all GPU processes and setting cooldown.

    Called when depth_map_generator or sbs_generator fails due to GPU driver crash.
    Terminates all GPU-dependent processes and sets a cooldown period to allow
    the GPU driver to recover before starting new processes.

    Parameters
    ----------
    state : OrchestratorState
        Orchestrator state containing active processes.
    """
    gpu_processes = [
        (key, info) for key, info in state.active_processes.items()
        if info.step_name in ('depth_map_generator', 'sbs_generator')
    ]

    # Set cooldown regardless of whether there are processes to terminate
    state.gpu_cooldown_until = time.time() + GPU_COOLDOWN_SECONDS

    if not gpu_processes:
        async with state.console_lock:
            state.log_messages.append(
                f'{_format_timestamp()} [yellow]GPU failure detected - cooldown for {GPU_COOLDOWN_SECONDS:.0f}s before restarting GPU processes[/yellow]'
            )
        return

    async with state.console_lock:
        state.log_messages.append(
            f'{_format_timestamp()} [yellow]GPU failure detected - terminating {len(gpu_processes)} GPU process(es), '
            f'cooldown for {GPU_COOLDOWN_SECONDS:.0f}s[/yellow]'
        )

    for process_key, process_info in gpu_processes:
        try:
            process_info.process.terminate()
            workflow_name = _get_workflow_name(process_info.workflow_path)
            async with state.console_lock:
                state.log_messages.append(
                    f'{_format_timestamp()} [yellow]TERMINATED[/yellow]: {process_info.step_name} for {workflow_name}'
                )
        except ProcessLookupError:
            pass  # Process already terminated


async def _monitor_process(state: OrchestratorState, process_info: ProcessInfo) -> None:
    """Monitor a subprocess until completion."""
    workflow_path = process_info.workflow_path
    step_name = process_info.step_name
    workflow = state.workflows.get(workflow_path)

    if not workflow:
        return

    try:
        # Wait for process to complete
        return_code = await process_info.process.wait()

        # Process completed
        process_key = f'{workflow_path}:{step_name}'

        if return_code == 0:
            # Success
            if step_name == 'chunk_generator':
                # Invalidate cache so next iteration sees new chunk files
                # Status is determined dynamically from filesystem - no state update needed
                invalidate_cache()
            elif step_name == 'video_concatenator':
                # Status is determined dynamically from filesystem - no state update needed
                invalidate_cache()
            elif step_name == 'sbs_generator':
                # Check if SBS needs to run again (re-trigger)
                depth_status = get_step_status(workflow.get('depth_map_generator', StepStatus.PENDING))
                if depth_status == StepStatus.DONE:
                    path = Path(workflow_path)
                    max_depth = get_max_depth_number(path)
                    max_sbs = get_max_sbs_number(path)
                    if max_sbs >= max_depth:
                        set_step_done(workflow, step_name)
                    else:
                        set_step_pending(workflow, step_name)
                else:
                    # Depth still running, need to re-run SBS later
                    set_step_pending(workflow, step_name)
            else:
                # frame_extractor, depth_map_generator
                set_step_done(workflow, step_name)

            async with state.console_lock:
                state.log_messages.append(f'{_format_timestamp()} [green]DONE[/green]: {step_name} for {_get_workflow_name(workflow_path)}')
        else:
            # Error handling
            # Extract last few lines of output for error context
            error_output = process_info.stderr_buffer.strip()
            if error_output:
                # Get last 5 non-empty lines
                error_lines = [line for line in error_output.split('\n') if line.strip()][-5:]
                error_context = '\n'.join(error_lines)
            else:
                error_context = ''

            # Check if this is a GPU-related failure that should terminate all GPU processes
            # Exit code 100 is used by sbs_generator for GPU health check failures
            is_gpu_failure = step_name == 'depth_map_generator' or return_code == 100
            if is_gpu_failure:
                await _handle_gpu_failure(state)

            # For transient steps (chunk_generator, video_concatenator), just log the error
            # They will be retried automatically on next scheduler run based on filesystem state
            if step_name in ('chunk_generator', 'video_concatenator'):
                async with state.console_lock:
                    state.log_messages.append(
                        f'{_format_timestamp()} [red]FAILED[/red]: {step_name} for {_get_workflow_name(workflow_path)} '
                        f'(exit code: {return_code}) - will retry automatically'
                    )
                    if error_context:
                        for line in error_context.split('\n'):
                            state.log_messages.append(f'  [yellow]{line}[/yellow]')
            else:
                # For persistent steps, track failure count
                current_status = get_step_status(workflow.get(step_name, StepStatus.PENDING))

                if current_status == StepStatus.FAILED:
                    # Second failure - permanent error
                    set_step_error(workflow, step_name)
                    async with state.console_lock:
                        state.log_messages.append(
                            f'{_format_timestamp()} [red bold]ERROR[/red bold]: {step_name} for {_get_workflow_name(workflow_path)} '
                            f'(exit code: {return_code}) - permanent failure, needs manual intervention'
                        )
                        if error_context:
                            for line in error_context.split('\n'):
                                state.log_messages.append(f'  [red]{line}[/red]')
                else:
                    # First failure - will retry
                    set_step_failed(workflow, step_name)
                    async with state.console_lock:
                        state.log_messages.append(
                            f'{_format_timestamp()} [red]FAILED[/red]: {step_name} for {_get_workflow_name(workflow_path)} '
                            f'(exit code: {return_code}) - will retry'
                        )
                        if error_context:
                            for line in error_context.split('\n'):
                                state.log_messages.append(f'  [yellow]{line}[/yellow]')

        # Save state and trigger scheduling (only if state was modified)
        if step_name not in ('chunk_generator', 'video_concatenator'):
            _save_and_sync(state)
        invalidate_cache()
        state.schedule_needed.set()

    except asyncio.CancelledError:
        # Graceful shutdown
        process_info.process.terminate()
        try:
            await asyncio.wait_for(process_info.process.wait(), timeout=30)
        except asyncio.TimeoutError:
            process_info.process.kill()

    finally:
        # Remove from active processes
        process_key = f'{workflow_path}:{step_name}'
        state.active_processes.pop(process_key, None)


async def _start_process(state: OrchestratorState, workflow_path: str, step_name: str) -> bool:
    """Start a subprocess for the given step."""
    workflow = state.workflows.get(workflow_path)
    if not workflow:
        return False

    # Check disk space
    path = Path(workflow_path)
    free_space = _get_disk_space_gb(path.parent)
    if free_space < DISK_SPACE_THRESHOLD_GB:
        async with state.console_lock:
            state.log_messages.append(
                f'{_format_timestamp()} [red]WARNING[/red]: Low disk space ({free_space:.1f} GB), '
                f'blocking new processes'
            )
        return False

    cmd = _build_command(step_name, workflow_path, workflow)
    if not cmd:
        return False

    # Set environment variable to prevent subprocesses from changing terminal title
    env = os.environ.copy()
    env['DISABLE_TERMINAL_TITLE'] = '1'

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=Path(__file__).parent,
            env=env
        )

        process_info = ProcessInfo(
            workflow_path=workflow_path,
            step_name=step_name,
            process=process,
            task=None  # Set below
        )

        # Start output reader
        output_task = asyncio.create_task(_read_output(state, process_info))
        process_info.output_task = output_task

        # Start monitor
        monitor_task = asyncio.create_task(_monitor_process(state, process_info))
        process_info.task = monitor_task

        # Register process
        process_key = f'{workflow_path}:{step_name}'
        state.active_processes[process_key] = process_info

        # Update workflow state for persistent steps only
        # chunk_generator and video_concatenator status is derived from filesystem
        if step_name not in ('chunk_generator', 'video_concatenator'):
            set_step_running(workflow, step_name)
            _save_and_sync(state)

        async with state.console_lock:
            state.log_messages.append(f'{_format_timestamp()} [blue]STARTED[/blue]: {step_name} for {_get_workflow_name(workflow_path)} (PID: {process.pid})')

        return True

    except Exception as e:
        async with state.console_lock:
            state.log_messages.append(f'{_format_timestamp()} [red]ERROR[/red]: Failed to start {step_name}: {e}')
        return False


async def _schedule_step(state: OrchestratorState, step_name: str, can_start_func) -> int:
    """
    Schedule processes for a step type using PENDING-first priority.

    Returns number of processes started.
    """
    started = 0

    # Get current count for this step type
    if step_name in MUTEX_STEPS:
        current_count = _count_active_mutex(state)
        max_count = MAX_MUTEX_PROCESSES
    elif step_name == 'depth_map_generator':
        current_count = _count_active_by_step(state, step_name)
        max_count = MAX_DEPTH_PROCESSES
    elif step_name == 'sbs_generator':
        current_count = _count_active_by_step(state, step_name)
        max_count = MAX_SBS_PROCESSES
    else:
        return 0

    # Collect candidates
    # For transient steps (chunk_generator, video_concatenator), status is determined by can_start_func
    # For persistent steps, collect by priority: RUNNING > PENDING > FAILED

    if step_name in ('chunk_generator', 'video_concatenator'):
        # Transient steps: just check if they can start
        candidates = []
        for workflow_path, workflow in state.workflows.items():
            if can_start_func(state, workflow_path, workflow):
                candidates.append(workflow_path)
    else:
        # Persistent steps: collect by status priority
        running_workflows = []
        pending_workflows = []
        failed_workflows = []

        for workflow_path, workflow in state.workflows.items():
            step_value = workflow.get(step_name, StepStatus.PENDING)
            status = get_step_status(step_value)

            if not can_start_func(state, workflow_path, workflow):
                continue

            if status == StepStatus.RUNNING:
                # RUNNING without active process = needs restart (highest priority)
                process_key = f'{workflow_path}:{step_name}'
                if process_key not in state.active_processes:
                    running_workflows.append(workflow_path)
            elif status == StepStatus.PENDING:
                pending_workflows.append(workflow_path)
            elif status == StepStatus.FAILED:
                failed_workflows.append(workflow_path)

        # Process in priority order: RUNNING (restart) first, then PENDING, then FAILED (retry)
        # Within each category, maintain workflow order from YAML (dict iteration order)
        candidates = running_workflows + pending_workflows + failed_workflows

    # Sort candidates by their position in workflows dict to maintain YAML order
    workflow_order = {path: i for i, path in enumerate(state.workflows.keys())}
    candidates.sort(key=lambda p: workflow_order.get(p, float('inf')))

    for workflow_path in candidates:
        # Re-check current active count before each start
        if step_name in MUTEX_STEPS:
            current_active = _count_active_mutex(state)
        else:
            current_active = _count_active_by_step(state, step_name)

        if current_active >= max_count:
            break

        if await _start_process(state, workflow_path, step_name):
            started += 1

    return started


async def _schedule_frame_extractor_prefetch(state: OrchestratorState) -> int:
    """Schedule frame_extractor for prefetch (2 workflows ahead)."""
    if _count_active_mutex(state) >= MAX_MUTEX_PROCESSES:
        return 0

    candidates = _get_prefetch_candidates(state)
    started = 0

    for workflow_path in candidates:
        if _count_active_mutex(state) >= MAX_MUTEX_PROCESSES:
            break

        workflow = state.workflows.get(workflow_path)
        if workflow and _can_start_frame_extractor(state, workflow_path, workflow):
            if await _start_process(state, workflow_path, 'frame_extractor'):
                started += 1
                break  # Only start one at a time

    return started


async def _scheduler_loop(state: OrchestratorState) -> None:
    """Main scheduler loop."""
    last_fallback_check = datetime.now()

    # Check immediately if all workflows are already complete
    if _are_all_workflows_complete(state):
        async with state.console_lock:
            state.log_messages.append(f'{_format_timestamp()} [green]All workflows already completed![/green]')
        state.stop_event.set()
        return

    while not state.stop_event.is_set():
        try:
            # Wait for schedule trigger or timeout
            try:
                await asyncio.wait_for(
                    state.schedule_needed.wait(),
                    timeout=SCHEDULER_INTERVAL
                )
                # Clear the event for next trigger
                state.schedule_needed.clear()
                invalidate_cache()
            except asyncio.TimeoutError:
                pass  # Timeout is normal, continue with periodic check

            # Fallback check
            now = datetime.now()
            if (now - last_fallback_check).total_seconds() >= FALLBACK_CHECK_INTERVAL:
                invalidate_cache()
                # Sync state to pick up any manual changes to workflows.yaml
                _save_and_sync(state)
                last_fallback_check = now

            # Fix stale statuses
            stale_fixed = _fix_stale_sbs_status(state)
            stale_fixed = _fix_stale_depth_status(state) or stale_fixed
            if stale_fixed:
                _save_and_sync(state)

            # Schedule new processes
            # Priority order: video_concatenator > chunk_generator > sbs_generator > depth_map_generator > frame_extractor

            # 1. video_concatenator (highest priority for completion)
            await _schedule_step(state, 'video_concatenator', _can_start_concat)

            # 2. chunk_generator
            await _schedule_step(state, 'chunk_generator', _can_start_chunk)

            # 3. sbs_generator
            await _schedule_step(state, 'sbs_generator', _can_start_sbs)

            # 4. depth_map_generator
            await _schedule_step(state, 'depth_map_generator', _can_start_depth)

            # 5. frame_extractor (prefetch)
            await _schedule_frame_extractor_prefetch(state)

            # Check if all workflows are complete
            if _are_all_workflows_complete(state):
                async with state.console_lock:
                    state.log_messages.append(f'{_format_timestamp()} [green]All workflows completed![/green]')
                state.stop_event.set()
                break

        except asyncio.CancelledError:
            break
        except Exception as e:
            async with state.console_lock:
                state.log_messages.append(f'{_format_timestamp()} [red]Scheduler error[/red]: {e}')
            await asyncio.sleep(SCHEDULER_INTERVAL)


async def _live_display_loop(state: OrchestratorState) -> None:
    """Update the live display periodically."""
    while not state.stop_event.is_set():
        try:
            if state.live:
                state.live.update(_build_live_display(state))
            await asyncio.sleep(0.2)  # Update 5 times per second
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.2)


def _build_status_table(state: OrchestratorState) -> Table:
    """Build a status table for display."""
    table = Table(title='Workflow Orchestrator Status', expand=True)
    table.add_column('Workflow', style='cyan')
    table.add_column('Frame', style='white')
    table.add_column('Depth', style='white')
    table.add_column('SBS', style='white')
    table.add_column('Video', style='white')

    status_colors = {
        StepStatus.PENDING: 'dim',
        StepStatus.RUNNING: 'yellow',
        StepStatus.DONE: 'green',
        StepStatus.ERROR: 'red',
        StepStatus.FAILED: 'red bold',
    }

    # Only show first 3 steps from STEP_ORDER (persistent steps)
    display_steps = ['frame_extractor', 'depth_map_generator', 'sbs_generator']

    for workflow_path, workflow in state.workflows.items():
        name = _get_workflow_name(workflow_path)
        row = [name]

        for step_name in display_steps:
            status = get_step_status(workflow.get(step_name, StepStatus.PENDING))
            color = status_colors.get(status, 'white')
            row.append(f'[{color}]{status}[/{color}]')

        # Add video progress column
        video_progress = get_video_progress(Path(workflow_path))
        if video_progress == 'DONE':
            row.append('[green]DONE[/green]')
        elif video_progress == '-':
            row.append('[dim]-[/dim]')
        else:
            row.append(f'[yellow]{video_progress}[/yellow]')

        table.add_row(*row)

    return table


def _terminate_process_tree(pid: int) -> None:
    """
    Terminate a process and all its children.

    Uses psutil to find and terminate child processes before the parent.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Terminate children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Terminate parent
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass

        # Wait for termination with timeout
        gone, alive = psutil.wait_procs(children + [parent], timeout=5)

        # Force kill any remaining
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

    except psutil.NoSuchProcess:
        pass


async def _shutdown(state: OrchestratorState, all_complete: bool = False) -> None:
    """
    Graceful shutdown of all processes.

    Parameters
    ----------
    state : OrchestratorState
        Orchestrator state.
    all_complete : bool
        If True, all workflows completed successfully.
    """
    if all_complete:
        state.console.print('[green]All workflows completed![/green]')
    else:
        state.console.print('[yellow]Shutting down...[/yellow]')

    # Terminate all process trees using psutil
    for process_info in list(state.active_processes.values()):
        if process_info.process.pid:
            _terminate_process_tree(process_info.process.pid)

    # Cancel all monitor tasks
    for process_info in list(state.active_processes.values()):
        if process_info.output_task and not process_info.output_task.done():
            process_info.output_task.cancel()
        if process_info.task and not process_info.task.done():
            process_info.task.cancel()

    # Wait for all tasks to complete
    tasks = [p.task for p in state.active_processes.values() if p.task]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Mark all RUNNING as FAILED (interrupted by shutdown)
    # Note: We intentionally keep RUNNING status - no error occurred.
    # On next startup, RUNNING workflows will be prioritized for restart.

    _save_and_sync(state)
    if not all_complete:
        state.console.print('[green]Shutdown complete[/green]')


async def _run_orchestrator(state: OrchestratorState) -> None:
    """Main orchestrator entry point."""
    # Validate workflows
    state.console.print('[blue]Validating workflows...[/blue]')
    if not _validate_all_workflows(state):
        state.console.print('[yellow]Some workflows have validation errors[/yellow]')

    # Save workflows (migration already handled RUNNING -> FAILED on load)
    _save_and_sync(state)

    # Print initial status
    table = _build_status_table(state)
    state.console.print(table)

    # Setup signal handler
    def signal_handler():
        state.stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: signal_handler())

    # Start with live display
    with Live(_build_live_display(state), console=state.console, refresh_per_second=4) as live:
        state.live = live

        # Start scheduler and live display update tasks
        scheduler_task = asyncio.create_task(_scheduler_loop(state))
        live_task = asyncio.create_task(_live_display_loop(state))

        # Wait for stop event
        try:
            await state.stop_event.wait()
        except asyncio.CancelledError:
            pass

        # Shutdown tasks
        scheduler_task.cancel()
        live_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass
        try:
            await live_task
        except asyncio.CancelledError:
            pass

        state.live = None

    # Check if all workflows completed (vs manual interrupt)
    all_complete = _are_all_workflows_complete(state)
    await _shutdown(state, all_complete=all_complete)


def main() -> None:
    """Main entry point."""
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='Workflow Orchestrator - Automate multi-workflow video processing',
        epilog=(
            'Example:\n'
            '  python workflow_orchestrator.py workflows.yaml\n'
            '  python workflow_orchestrator.py workflows.yaml --validate-only\n'
            '\n'
            'The workflows.yaml file should contain a list of workflow directories:\n'
            '  - H:/video1/workflow\n'
            '  - H:/video2/workflow\n'
        )
    )

    parser.add_argument('yaml_path', type=Path, help='Path to workflows.yaml file')
    parser.add_argument('--validate-only', action='store_true', help='Only validate workflows, do not run')

    args = parser.parse_args()

    # Set terminal title for orchestrator
    set_terminal_title(f'workflow_orchestrator.py {args.yaml_path}')

    console = Console()

    # Load workflows
    if not args.yaml_path.exists():
        console.print(f'[red]ERROR[/red]: Workflows file not found: {args.yaml_path}')
        sys.exit(1)

    try:
        workflows = load_workflows(args.yaml_path)
    except Exception as e:
        console.print(f'[red]ERROR[/red]: Failed to load workflows: {e}')
        sys.exit(1)

    if not workflows:
        console.print('[yellow]No workflows found in file[/yellow]')
        sys.exit(0)

    console.print(f'[blue]Loaded {len(workflows)} workflow(s)[/blue]')

    # Create state
    state = OrchestratorState(
        workflows=workflows,
        yaml_path=args.yaml_path,
        console=console
    )

    # Validate only mode
    if args.validate_only:
        all_valid = _validate_all_workflows(state)
        table = _build_status_table(state)
        console.print(table)
        sys.exit(0 if all_valid else 1)

    # Run orchestrator
    try:
        asyncio.run(_run_orchestrator(state))
    except KeyboardInterrupt:
        console.print('[yellow]Interrupted[/yellow]')
        sys.exit(1)


if __name__ == '__main__':
    main()
