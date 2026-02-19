"""LCM-controlled LIBERO environment runner."""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import lcm
import numpy as np
from drake import (
    lcmt_drake_signal,
    lcmt_iiwa_command,
    lcmt_iiwa_status,
    lcmt_robot_state,
    lcmt_schunk_wsg_command,
)

from .scenario import LiberoLcmScenario, load_scenario


def run_env_server(scenario_path: Path) -> None:
    """Start LIBERO offscreen env server and handle LCM commands."""
    scenario = load_scenario(scenario_path)
    env = _build_libero_env(scenario)
    client = lcm.LCM(scenario.lcm_url)

    commands: deque[np.ndarray] = deque()
    events: deque[str] = deque()
    running = True
    step_count = 0

    obs = env.reset()
    viewer = _maybe_launch_native_viewer(env=env, enable=scenario.lcm_show_native_viewer)
    done = False
    info: dict[str, Any] = {}
    last_command = np.zeros(scenario.action_dim, dtype=np.float32)

    def _on_arm_command(_channel: str, data: bytes) -> None:
        action = _decode_iiwa_command(data)
        commands.append(_fit_action_dim(action, scenario.action_dim))

    def _on_hand_command(_channel: str, data: bytes) -> None:
        target_position_mm, _force = _decode_schunk_command(data)
        gripper_scalar = _mm_to_normalized_gripper(
            target_position_mm,
            scenario.gripper_open_mm,
            scenario.gripper_close_mm,
        )
        action = last_command.copy()
        if scenario.action_dim > 0:
            action[-1] = gripper_scalar
        commands.append(action)

    def _on_event(_channel: str, data: bytes) -> None:
        events.append(_decode_control_event(data))

    client.subscribe(scenario.command_channel, _on_arm_command)
    client.subscribe(scenario.hand_command_channel, _on_hand_command)
    client.subscribe(scenario.event_channel, _on_event)

    _publish_snapshot(
        client=client,
        scenario=scenario,
        obs=obs,
        step=step_count,
        done=done,
        info=info,
        commanded=last_command,
    )

    tick_s = 1.0 / max(scenario.control_hz, 1e-3)
    next_tick = time.monotonic() + tick_s
    idle_action = np.zeros(scenario.action_dim, dtype=np.float32)

    while running:
        client.handle_timeout(5)
        now = time.monotonic()
        if now < next_tick:
            continue
        next_tick = now + tick_s

        command_updated = False
        if events:
            event = events.popleft()
            if event == "reset":
                obs = env.reset()
                done = False
                info = {}
                step_count = 0
                last_command.fill(0.0)
                viewer = _refresh_viewer(viewer=viewer, env=env)
            elif event == "close":
                running = False
        while commands:
            last_command = commands.popleft()
            command_updated = True

        should_step_with_last = (
            scenario.continuous_step_with_last_command or command_updated
        )
        if running and should_step_with_last:
            obs, _reward, done, info = env.step(last_command)
            step_count += 1
        elif running and scenario.auto_step_idle:
            obs, _reward, done, info = env.step(idle_action)
            step_count += 1

        if done and scenario.auto_reset_on_done:
            obs = env.reset()
            done = False
            info = {}
            viewer = _refresh_viewer(viewer=viewer, env=env)

        _publish_snapshot(
            client=client,
            scenario=scenario,
            obs=obs,
            step=step_count,
            done=done,
            info=info,
            commanded=last_command,
        )
        if viewer is not None and viewer.is_running():
            viewer.sync()

    if viewer is not None:
        viewer.close()
    env.close()


def run_interactive(scenario_path: Path) -> None:
    """Run native MuJoCo interactive viewer loop without LCM."""
    scenario = load_scenario(scenario_path)
    env = _build_interactive_env(scenario)
    action = np.zeros(scenario.action_dim, dtype=np.float32)
    tick_s = 1.0 / max(scenario.control_hz, 1e-3)

    print("Interactive MuJoCo viewer started. Press Ctrl+C to exit.")
    _run_native_mujoco_viewer(env=env, action=action, tick_s=tick_s)


def run_viewer_smoke_test(steps: int = 100000) -> None:
    """Launch a minimal native MuJoCo viewer for mouse camera testing."""
    import robosuite as suite

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=max(steps, 1),
    )
    low, high = env.action_spec
    action = np.zeros_like(low, dtype=np.float32)
    action = np.clip(action, low, high)

    print("Native viewer smoke test started. Try mouse drag / scroll.")
    _run_native_mujoco_viewer(
        env=env,
        action=action,
        tick_s=1.0 / 20.0,
        max_steps=max(steps, 1),
    )


def _build_libero_env(scenario: LiberoLcmScenario) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(scenario.benchmark_name)(scenario.task_order_index)
    task = benchmark.get_task(scenario.task_id)
    bddl_root = scenario.bddl_root or get_libero_path("bddl_files")
    bddl_file = Path(bddl_root) / task.problem_folder / task.bddl_file

    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=scenario.camera_height,
        camera_widths=scenario.camera_width,
        has_renderer=scenario.has_renderer,
        has_offscreen_renderer=scenario.has_offscreen_renderer,
    )


def _build_interactive_env(scenario: LiberoLcmScenario) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs.env_wrapper import ControlEnv

    benchmark = get_benchmark(scenario.benchmark_name)(scenario.task_order_index)
    task = benchmark.get_task(scenario.task_id)
    bddl_root = scenario.bddl_root or get_libero_path("bddl_files")
    bddl_file = Path(bddl_root) / task.problem_folder / task.bddl_file

    render_camera = (
        None if scenario.interactive_free_camera else scenario.interactive_render_camera
    )

    env_kwargs: dict[str, Any] = {}
    if scenario.interactive_renderer is not None:
        env_kwargs["renderer"] = scenario.interactive_renderer

    return ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=max(int(round(scenario.control_hz)), 1),
        render_camera=render_camera,
        **env_kwargs,
    )


def _run_native_mujoco_viewer(
    *,
    env: Any,
    action: np.ndarray,
    tick_s: float,
    max_steps: int | None = None,
) -> None:
    """Drive simulation while syncing native MuJoCo viewer."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco.viewer

    steps = 0
    try:
        env.reset()
        model = env.sim.model._model
        data = env.sim.data._data
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                _obs, _reward, done, _info = env.step(action)
                viewer.sync()
                if done:
                    env.reset()
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
                time.sleep(tick_s)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def _maybe_launch_native_viewer(*, env: Any, enable: bool) -> Any | None:
    if not enable:
        return None
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco.viewer

    model = env.sim.model._model
    data = env.sim.data._data
    return mujoco.viewer.launch_passive(model, data)


def _refresh_viewer(*, viewer: Any | None, env: Any) -> Any | None:
    if viewer is None:
        return None
    was_running = viewer.is_running()
    viewer.close()
    if not was_running:
        return None
    return _maybe_launch_native_viewer(env=env, enable=True)


def _fit_action_dim(action: np.ndarray, expected_dim: int) -> np.ndarray:
    output = np.zeros(expected_dim, dtype=np.float32)
    n = min(expected_dim, action.size)
    output[:n] = action[:n]
    return output


def _publish_snapshot(
    *,
    client: lcm.LCM,
    scenario: LiberoLcmScenario,
    obs: Any,
    step: int,
    done: bool,
    info: dict[str, Any],
    commanded: np.ndarray,
) -> None:
    frames = _extract_frames(obs, max_frames=scenario.max_camera_frames)
    if frames:
        if scenario.flip_vertical:
            frames = [np.flipud(frame) for frame in frames]
        frame = _compose_frames(frames, layout=scenario.frame_layout)
        client.publish(
            scenario.frame_channel,
            _encode_frame_signal(
                frame=frame,
                step=step,
                done=done,
                success=bool(info.get("success", False)),
            ),
        )

    measured = _extract_measured(obs, scenario.action_dim)
    external = np.zeros_like(measured)
    if external.size > 0:
        external[0] = float(step)
    if external.size > 1:
        external[1] = float(done)
    if external.size > 2:
        external[2] = float(bool(info.get("success", False)))
    if external.size > 3:
        external[3] = float(len(frames))

    client.publish(
        scenario.state_channel,
        _encode_iiwa_status(
            measured=measured,
            commanded=commanded,
            external=external,
        ),
    )
    client.publish(
        scenario.robot_state_channel,
        _encode_robot_state(measured),
    )


def _extract_measured(obs: Any, action_dim: int) -> np.ndarray:
    measured = np.zeros(action_dim, dtype=np.float32)
    if not isinstance(obs, dict):
        return measured
    for value in obs.values():
        if isinstance(value, np.ndarray) and value.ndim == 1 and value.size >= action_dim:
            measured[:] = np.asarray(value[:action_dim], dtype=np.float32)
            return measured
    return measured


def _extract_frames(obs: Any, max_frames: int) -> list[np.ndarray]:
    raw_frames: list[np.ndarray] = []
    _collect_frame_candidates(obs, raw_frames)
    if max_frames > 0:
        raw_frames = raw_frames[:max_frames]
    return [_to_u8_rgb(frame) for frame in raw_frames]


def _collect_frame_candidates(obs: Any, output: list[np.ndarray]) -> None:
    if isinstance(obs, np.ndarray) and obs.ndim == 3:
        output.append(obs)
        return

    if isinstance(obs, dict):
        keys = sorted(obs.keys())
        prioritized = [k for k in keys if "image" in str(k).lower()]
        for key in keys:
            if key not in prioritized:
                prioritized.append(key)
        for key in prioritized:
            _collect_frame_candidates(obs[key], output)
        return

    if isinstance(obs, list | tuple):
        for item in obs:
            _collect_frame_candidates(item, output)


def _to_u8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    return frame


def _compose_frames(frames: list[np.ndarray], layout: str) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]

    mode = layout.strip().lower()
    if mode == "vertical":
        target_width = max(frame.shape[1] for frame in frames)
        padded = [_pad_frame(frame, width=target_width) for frame in frames]
        separator = np.zeros((2, target_width, 3), dtype=np.uint8)
        return np.vstack(_interleave_with_separator(padded, separator))

    target_height = max(frame.shape[0] for frame in frames)
    padded = [_pad_frame(frame, height=target_height) for frame in frames]
    separator = np.zeros((target_height, 2, 3), dtype=np.uint8)
    composite = [padded[0]]
    for frame in padded[1:]:
        composite.append(separator)
        composite.append(frame)
    return np.hstack(composite)


def _interleave_with_separator(
    frames: list[np.ndarray], separator: np.ndarray
) -> list[np.ndarray]:
    output: list[np.ndarray] = [frames[0]]
    for frame in frames[1:]:
        output.append(separator)
        output.append(frame)
    return output


def _pad_frame(
    frame: np.ndarray, *, height: int | None = None, width: int | None = None
) -> np.ndarray:
    target_h = height if height is not None else frame.shape[0]
    target_w = width if width is not None else frame.shape[1]
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[: frame.shape[0], : frame.shape[1], :] = frame
    return canvas


def _mm_to_normalized_gripper(value_mm: float, open_mm: float, close_mm: float) -> float:
    width = max(open_mm - close_mm, 1e-6)
    alpha = (value_mm - close_mm) / width
    alpha = np.clip(alpha, 0.0, 1.0)
    return float(alpha * 2.0 - 1.0)


def _decode_iiwa_command(data: bytes) -> np.ndarray:
    msg = lcmt_iiwa_command.decode(data)
    return np.asarray(msg.joint_position, dtype=np.float32)


def _decode_schunk_command(data: bytes) -> tuple[float, float]:
    msg = lcmt_schunk_wsg_command.decode(data)
    return float(msg.target_position_mm), float(msg.force)


def _decode_control_event(data: bytes) -> str:
    msg = lcmt_drake_signal.decode(data)
    if msg.dim < 1 or len(msg.coord) < 1:
        return ""
    return str(msg.coord[0]).strip().lower()


def _encode_iiwa_status(
    *,
    measured: np.ndarray,
    commanded: np.ndarray,
    external: np.ndarray | None = None,
) -> bytes:
    n = int(measured.size)
    msg = lcmt_iiwa_status()
    msg.utime = time.monotonic_ns() // 1000
    msg.num_joints = n
    msg.joint_position_measured = measured.astype(float).tolist()
    msg.joint_velocity_estimated = np.zeros(n, dtype=float).tolist()
    msg.joint_position_commanded = commanded.astype(float).tolist()
    msg.joint_torque_commanded = np.zeros(n, dtype=float).tolist()
    msg.joint_torque_measured = np.zeros(n, dtype=float).tolist()
    msg.joint_torque_external = (
        external.astype(float).tolist()
        if external is not None
        else np.zeros(n, dtype=float).tolist()
    )
    msg.joint_position_ipo = measured.astype(float).tolist()
    return msg.encode()


def _encode_robot_state(state: np.ndarray) -> bytes:
    msg = lcmt_robot_state()
    msg.utime = time.monotonic_ns() // 1000
    msg.num_joints = int(state.size)
    msg.joint_position = state.astype(float).tolist()
    msg.joint_name = [f"j{i}" for i in range(state.size)]
    return msg.encode()


def _encode_frame_signal(
    *,
    frame: np.ndarray,
    step: int,
    done: bool,
    success: bool,
) -> bytes:
    h, w, c = frame.shape
    flattened = frame.reshape(-1).astype(float)
    prefix = np.asarray(
        [float(step), float(h), float(w), float(c), float(done), float(success)],
        dtype=float,
    )
    msg = lcmt_drake_signal()
    msg.dim = int(prefix.size + flattened.size)
    msg.coord = [
        "step",
        "height",
        "width",
        "channels",
        "done",
        "success",
    ] + [f"px_{i}" for i in range(flattened.size)]
    msg.val = np.concatenate((prefix, flattened)).tolist()
    msg.timestamp = time.monotonic_ns() // 1000
    return msg.encode()
