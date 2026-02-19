"""Scenario loading for LIBERO + LCM runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LiberoLcmScenario:
    """Configuration for running a LIBERO environment over LCM."""

    benchmark_name: str = "libero_object"
    task_order_index: int = 0
    task_id: int = 0
    bddl_root: str | None = None
    camera_height: int = 256
    camera_width: int = 256
    has_renderer: bool = True
    has_offscreen_renderer: bool = True
    lcm_url: str = "udpm://239.255.76.67:7667?ttl=1"
    command_channel: str = "LIBERO_COMMAND"
    hand_command_channel: str = "LIBERO_HAND_COMMAND"
    event_channel: str = "LIBERO_EVENT"
    state_channel: str = "LIBERO_STATE"
    robot_state_channel: str = "LIBERO_ROBOT_STATE"
    frame_channel: str = "LIBERO_FRAME"
    control_hz: float = 20.0
    action_dim: int = 7
    continuous_step_with_last_command: bool = True
    auto_step_idle: bool = False
    auto_reset_on_done: bool = True
    gripper_open_mm: float = 110.0
    gripper_close_mm: float = 0.0
    max_camera_frames: int = 2
    frame_layout: str = "horizontal"
    flip_vertical: bool = True
    interactive_free_camera: bool = True
    interactive_renderer: str | None = None
    interactive_render_camera: str | None = None
    lcm_show_native_viewer: bool = True


def load_scenario(path: Path) -> LiberoLcmScenario:
    """Load scenario YAML into a typed config."""
    data = _load_yaml(path)
    return LiberoLcmScenario(**data)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        msg = "Scenario file must contain a YAML mapping."
        raise ValueError(msg)
    return loaded
