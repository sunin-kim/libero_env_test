"""Realtime frame viewer for LIBERO LCM stream."""

from __future__ import annotations

import argparse
import time
from collections import deque

import lcm
import matplotlib.pyplot as plt
import numpy as np
from drake import lcmt_drake_signal, lcmt_iiwa_status


def run_visualizer(
    *,
    lcm_url: str,
    frame_channel: str,
    state_channel: str,
) -> None:
    """Subscribe to frame / state channels and render continuously."""
    client = lcm.LCM(lcm_url)
    frames: deque[tuple[np.ndarray, int, bool, bool, float]] = deque(maxlen=1)
    states: deque[dict[str, np.ndarray]] = deque(maxlen=1)

    def _on_frame(_channel: str, data: bytes) -> None:
        frames.append(_decode_frame_signal(data))

    def _on_state(_channel: str, data: bytes) -> None:
        states.append(_decode_iiwa_status(data))

    client.subscribe(frame_channel, _on_frame)
    client.subscribe(state_channel, _on_state)

    plt.ion()
    fig, ax = plt.subplots(num="LIBERO LCM Visualizer")
    image_artist = None
    title = ax.set_title("waiting for frames...")
    ax.axis("off")
    fig.tight_layout()

    while plt.fignum_exists(fig.number):
        client.handle_timeout(10)
        if frames:
            frame, step, done, success, timestamp = frames[-1]
            if image_artist is None:
                image_artist = ax.imshow(frame)
            else:
                image_artist.set_data(frame)

            if states:
                state = states[-1]
                measured = state["measured"]
                q0 = float(measured[0]) if measured.size > 0 else 0.0
                title.set_text(
                    "step=%d q0=%.3f done=%s success=%s t=%.3f"
                    % (step, q0, done, success, timestamp)
                )
            else:
                title.set_text(f"step={step} t={timestamp:.3f}")
            fig.canvas.draw_idle()
        plt.pause(0.001)
        time.sleep(0.001)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize LIBERO frames streamed over LCM."
    )
    parser.add_argument(
        "--lcm-url",
        default="udpm://239.255.76.67:7667?ttl=1",
        help="LCM multicast URL.",
    )
    parser.add_argument(
        "--frame-channel",
        default="LIBERO_FRAME",
        help="Frame channel name.",
    )
    parser.add_argument(
        "--state-channel",
        default="LIBERO_STATE",
        help="State channel name.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_visualizer(
        lcm_url=args.lcm_url,
        frame_channel=args.frame_channel,
        state_channel=args.state_channel,
    )


def _decode_iiwa_status(data: bytes) -> dict[str, np.ndarray]:
    msg = lcmt_iiwa_status.decode(data)
    return {
        "measured": np.asarray(msg.joint_position_measured, dtype=np.float32),
        "commanded": np.asarray(msg.joint_position_commanded, dtype=np.float32),
        "external": np.asarray(msg.joint_torque_external, dtype=np.float32),
    }


def _decode_frame_signal(data: bytes) -> tuple[np.ndarray, int, bool, bool, float]:
    msg = lcmt_drake_signal.decode(data)
    if len(msg.val) < 6:
        raise ValueError("Invalid frame signal.")
    step = int(msg.val[0])
    h = int(msg.val[1])
    w = int(msg.val[2])
    c = int(msg.val[3])
    done = bool(int(msg.val[4]))
    success = bool(int(msg.val[5]))
    pixels = np.asarray(msg.val[6:], dtype=np.float32)
    frame = pixels.reshape((h, w, c))
    frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return frame, step, done, success, float(msg.timestamp) / 1_000_000.0


if __name__ == "__main__":
    main()
