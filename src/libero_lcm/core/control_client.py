"""Command publisher for LIBERO LCM env server."""

from __future__ import annotations

import argparse
import time

import lcm
import numpy as np
from drake import lcmt_drake_signal, lcmt_iiwa_command, lcmt_schunk_wsg_command


def send_control_command(
    *,
    lcm_url: str,
    channel: str,
    action: list[float],
) -> None:
    """Publish a single arm command to the env server."""
    client = lcm.LCM(lcm_url)
    msg = lcmt_iiwa_command()
    action_np = np.asarray(action, dtype=np.float32)
    msg.utime = time.monotonic_ns() // 1000
    msg.num_joints = int(action_np.size)
    msg.joint_position = action_np.astype(float).tolist()
    msg.num_torques = int(action_np.size)
    msg.joint_torque = np.zeros(action_np.size, dtype=float).tolist()
    client.publish(channel, msg.encode())


def send_hand_command(
    *,
    lcm_url: str,
    channel: str,
    target_position_mm: float,
    force: float = 40.0,
) -> None:
    """Publish gripper command as lcmt_schunk_wsg_command."""
    client = lcm.LCM(lcm_url)
    msg = lcmt_schunk_wsg_command()
    msg.utime = time.monotonic_ns() // 1000
    msg.target_position_mm = float(target_position_mm)
    msg.force = float(force)
    client.publish(channel, msg.encode())


def send_event_command(*, lcm_url: str, channel: str, kind: str) -> None:
    """Publish control event (reset / close) as lcmt_drake_signal."""
    client = lcm.LCM(lcm_url)
    msg = lcmt_drake_signal()
    msg.dim = 1
    msg.coord = [kind]
    msg.val = [1.0]
    msg.timestamp = time.monotonic_ns() // 1000
    client.publish(channel, msg.encode())


def run_random_control(
    *,
    lcm_url: str,
    channel: str,
    hand_channel: str,
    action_dim: int,
    steps: int,
    rate_hz: float,
    seed: int,
) -> None:
    """Send random actions repeatedly for quick smoke testing."""
    rng = np.random.default_rng(seed)
    period = 1.0 / max(rate_hz, 1e-3)
    for _ in range(steps):
        action = rng.uniform(low=-1.0, high=1.0, size=(action_dim,))
        send_control_command(
            lcm_url=lcm_url,
            channel=channel,
            action=action.astype(float).tolist(),
        )
        if action_dim > 0:
            target_position_mm = normalized_to_mm(float(action[-1]))
            send_hand_command(
                lcm_url=lcm_url,
                channel=hand_channel,
                target_position_mm=target_position_mm,
            )
        time.sleep(period)


def parse_action_csv(text: str) -> list[float]:
    """Parse action string like `0,0,0,0,0,0,0`."""
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def normalized_to_mm(value: float, open_mm: float = 110.0, close_mm: float = 0.0) -> float:
    alpha = (np.clip(value, -1.0, 1.0) + 1.0) / 2.0
    return float(close_mm + alpha * (open_mm - close_mm))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish control commands to LIBERO LCM server."
    )
    parser.add_argument(
        "--lcm-url",
        default="udpm://239.255.76.67:7667?ttl=1",
        help="LCM multicast URL.",
    )
    parser.add_argument(
        "--command-channel",
        default="LIBERO_COMMAND",
        help="Arm command channel name.",
    )
    parser.add_argument(
        "--hand-command-channel",
        default="LIBERO_HAND_COMMAND",
        help="Gripper command channel name.",
    )
    parser.add_argument(
        "--event-channel",
        default="LIBERO_EVENT",
        help="Reset / close event channel name.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("reset", help="Reset environment.")
    subparsers.add_parser("close", help="Stop environment process.")

    step = subparsers.add_parser("step", help="Step once with action.")
    step.add_argument(
        "--action",
        required=True,
        help="CSV action values, e.g. 0,0,0,0,0,0,0",
    )

    random_step = subparsers.add_parser(
        "random",
        help="Send random actions for N steps.",
    )
    random_step.add_argument("--steps", type=int, default=100)
    random_step.add_argument("--rate-hz", type=float, default=10.0)
    random_step.add_argument("--action-dim", type=int, default=7)
    random_step.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"reset", "close"}:
        send_event_command(
            lcm_url=args.lcm_url,
            channel=args.event_channel,
            kind=args.command,
        )
        return

    if args.command == "step":
        action = parse_action_csv(args.action)
        send_control_command(
            lcm_url=args.lcm_url,
            channel=args.command_channel,
            action=action,
        )
        if action:
            send_hand_command(
                lcm_url=args.lcm_url,
                channel=args.hand_command_channel,
                target_position_mm=normalized_to_mm(action[-1]),
            )
        return

    if args.command == "random":
        run_random_control(
            lcm_url=args.lcm_url,
            channel=args.command_channel,
            hand_channel=args.hand_command_channel,
            action_dim=args.action_dim,
            steps=args.steps,
            rate_hz=args.rate_hz,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
