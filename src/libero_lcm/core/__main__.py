"""CLI for LIBERO LCM tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .control_client import (
    normalized_to_mm,
    parse_action_csv,
    run_random_control,
    send_control_command,
    send_event_command,
    send_hand_command,
)
from .env_server import run_env_server, run_interactive, run_viewer_smoke_test
from .scenario import load_scenario
from .visualizer import run_visualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LIBERO environment over LCM with visualization tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m libero_lcm.core run-env --scenario config/libero_lcm_scenario.yaml
  python -m libero_lcm.core run-interactive --scenario config/libero_lcm_scenario.yaml
  python -m libero_lcm.core test-viewer --steps 10000
  python -m libero_lcm.core visualize
  python -m libero_lcm.core control reset
  python -m libero_lcm.core control step --action 0,0,0,0,0,0,0
  python -m libero_lcm.core control random --steps 200
        """,
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_env = subparsers.add_parser(
        "run-env",
        help="Run LIBERO env server and receive LCM commands.",
    )
    run_env.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to scenario yaml.",
    )

    run_interactive_parser = subparsers.add_parser(
        "run-interactive",
        help="Run native MuJoCo interactive viewer locally (no LCM).",
    )
    run_interactive_parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to scenario yaml.",
    )

    test_viewer = subparsers.add_parser(
        "test-viewer",
        help="Minimal native MuJoCo viewer smoke test for mouse control.",
    )
    test_viewer.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Simulation steps before auto-exit.",
    )

    vis = subparsers.add_parser(
        "visualize",
        help="Visualize frames coming from the running env server.",
    )
    vis.add_argument(
        "--lcm-url",
        default="udpm://239.255.76.67:7667?ttl=1",
        help="LCM multicast URL.",
    )
    vis.add_argument("--frame-channel", default="LIBERO_FRAME")
    vis.add_argument("--state-channel", default="LIBERO_STATE")

    control = subparsers.add_parser(
        "control",
        help="Publish command messages to env server.",
    )
    control.add_argument(
        "--lcm-url",
        default="udpm://239.255.76.67:7667?ttl=1",
        help="LCM multicast URL.",
    )
    control.add_argument("--command-channel", default="LIBERO_COMMAND")
    control.add_argument("--hand-command-channel", default="LIBERO_HAND_COMMAND")
    control.add_argument("--event-channel", default="LIBERO_EVENT")

    control_sub = control.add_subparsers(dest="control_command", required=True)
    control_sub.add_parser("reset")
    control_sub.add_parser("close")

    step = control_sub.add_parser("step")
    step.add_argument("--action", required=True)

    random_step = control_sub.add_parser("random")
    random_step.add_argument("--steps", type=int, default=100)
    random_step.add_argument("--rate-hz", type=float, default=10.0)
    random_step.add_argument("--seed", type=int, default=0)
    random_step.add_argument("--action-dim", type=int, default=7)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    match args.subcommand:
        case "run-env":
            # Validate early so configuration errors are easier to debug.
            load_scenario(args.scenario)
            run_env_server(args.scenario)
        case "visualize":
            run_visualizer(
                lcm_url=args.lcm_url,
                frame_channel=args.frame_channel,
                state_channel=args.state_channel,
            )
        case "run-interactive":
            load_scenario(args.scenario)
            run_interactive(args.scenario)
        case "test-viewer":
            run_viewer_smoke_test(steps=args.steps)
        case "control":
            _run_control_command(args)


def _run_control_command(args: argparse.Namespace) -> None:
    command = args.control_command
    if command in {"reset", "close"}:
        send_event_command(
            lcm_url=args.lcm_url,
            channel=args.event_channel,
            kind=command,
        )
        return

    if command == "step":
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

    if command == "random":
        run_random_control(
            lcm_url=args.lcm_url,
            channel=args.command_channel,
            hand_channel=args.hand_command_channel,
            action_dim=args.action_dim,
            steps=args.steps,
            rate_hz=args.rate_hz,
            seed=args.seed,
        )
        return

    msg = f"Unsupported control command: {command}"
    raise ValueError(msg)


if __name__ == "__main__":
    sys.exit(main())
