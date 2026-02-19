import pickle
import time as time_module
import numpy as np
import numpy.typing as npt
from typing import Union, Dict, List
from matplotlib.pyplot import get_cmap

from pydrake.all import (
    Meshcat,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Simulator,
    Role,
    Rgba,
    MultibodyPlant,
    SceneGraph,
    RigidTransform,
    Quaternion,
    AddFrameTriadIllustration,
    PiecewisePose,
    PiecewisePolynomial,
)
from pydrake.geometry import MeshcatAnimation

Q_CLOSE = -0.74734999  # rad (fully closed) - 0
Q_OPEN = 0.445058959  # rad (fully open) - 1


class SigmaHandVisualizer:
    """
    Visualize trajectories of multiple SigmaHand robots in Meshcat.

    Usage:
        # 1. Initialize with config (defines which hands to visualize and their appearance)
        config = {
            "tracking": {"color": [0.8, 0.8, 0.0, 0.4]},
            "actual": {"color": "real"},
        }
        visualizer = SigmaHandVisualizer(config)

        # 2. Add trajectory data for each configured hand
        visualizer.add_trajectory("tracking", time, ee_pose, hand)
        visualizer.add_trajectory("actual", time, ee_pose, hand)

        # 3. Record and publish
        visualizer.record_trajectory()

    Waypoint visualization usage:
        # Generate config for waypoints with colormap
        waypoint_config = SigmaHandVisualizer.generate_waypoint_config(
            num_waypoints=4, cmap_name="plasma", alpha=0.3
        )
        config = {"actual": {"color": "real"}, **waypoint_config}
        visualizer = SigmaHandVisualizer(config)

        # Add static waypoint poses
        visualizer.add_waypoints(ee_poses, hands, times)
        visualizer.record_trajectory()

    Config format:
        {
            "hand_name": {
                "color": [r, g, b, a] | "real",  # "real" uses textured model
            },
            ...
        }

    Trajectory data format:
        - time: List[float] or np.ndarray of timestamps
        - ee_pose: List or np.ndarray of shape (T, 7) with [x, y, z, qw, qx, qy, qz]
        - hand: List or np.ndarray of shape (T,) with scalar hand position [0=open, 1=closed]

    Args:
        config: Dictionary mapping hand names to their display config.
                Each config should have a "color" key that is either "real"
                (for textured model) or [r, g, b, a] for solid color.
        meshcat: Optional existing Meshcat instance to reuse. If None, creates a new one.
        meshcat_port: Port for the Meshcat server (only used if meshcat is None).
    """

    @staticmethod
    def generate_waypoint_config(
        num_waypoints: int,
        cmap_name: str = "plasma",
        alpha: float = 0.3,
        prefix: str = "waypoint",
        include_animated: bool = True,
        animated_name: str = "animated",
    ) -> Dict[str, dict]:
        """
        Generate config entries for waypoint visualization with colormap colors.

        Args:
            num_waypoints: Number of waypoints to visualize.
            cmap_name: Matplotlib colormap name (e.g., "plasma", "viridis", "coolwarm").
            alpha: Transparency for waypoint hands (0=transparent, 1=opaque).
            prefix: Prefix for waypoint names (e.g., "waypoint" -> "waypoint_0", "waypoint_1", ...).
            include_animated: If True, includes an animated "real" robot that follows the trajectory.
            animated_name: Name for the animated robot entry.

        Returns:
            Config dictionary to merge with your main config.

        Example:
            waypoint_config = SigmaHandVisualizer.generate_waypoint_config(4, "plasma", 0.3)
            visualizer = SigmaHandVisualizer(waypoint_config)
            visualizer.add_waypoints(ee_poses, hands, times)  # Also adds animated trajectory
        """
        cmap = get_cmap(cmap_name)
        config = {}

        # Add animated robot config
        if include_animated:
            config[animated_name] = {"color": "real"}

        # Add waypoint configs with colormap colors
        for i in range(num_waypoints):
            color = list(cmap(i / max(num_waypoints - 1, 1)))
            color[3] = alpha
            config[f"{prefix}_{i}"] = {"color": color}

        return config

    def __init__(
        self, config: dict[str, dict], meshcat: Meshcat = None, meshcat_port: int = 7500
    ):
        self.config = config
        self.trajectories: Dict[str, dict] = {}  # Will store trajectory data per hand

        # Set up plant and meshcat visualizer
        if meshcat is not None:
            self.meshcat = meshcat
        else:
            self.meshcat = Meshcat(port=meshcat_port)
        self._setup_plant()

    def _clear_visualizations(self):
        """Clear only the visualizations created by this visualizer, not all Meshcat content."""
        # Delete plant visualizations (prefixed paths)
        for name in self.config.keys():
            self.meshcat.Delete(f"plant_{name}")

        # Delete trajectory line segments
        for name in self.trajectories.keys():
            self.meshcat.Delete(f"desired_ee_pose_{name}")

        # Delete waypoint path line segments
        self.meshcat.Delete("waypoint_path")

        # Delete controls
        try:
            self.meshcat.DeleteSlider("time")
            self.meshcat.DeleteSlider("rate")
            self.meshcat.DeleteButton("Play")
            self.meshcat.DeleteButton("Stop")
        except Exception:
            pass  # Controls might not exist

        self.meshcat.DeleteRecording()

    def cleanup(self):
        """Clean up all visualizations. Call before discarding this visualizer."""
        self._clear_visualizations()
        self._cleanup_plant()
        self.trajectories.clear()

    def clear_scene(self):
        """Clear visualizations and rebuild plants with current config."""
        self._clear_visualizations()
        self.trajectories.clear()
        self._setup_plant()  # Rebuild plants to restore geometry

    def add_trajectory(
        self,
        name: str,
        time: Union[List[float], npt.NDArray],
        ee_pose: Union[List, npt.NDArray],
        hand: Union[List[float], npt.NDArray],
    ):
        """
        Add or update trajectory data for a configured hand.

        Args:
            name: Name of the hand (must match a key in the config).
            time: Array of timestamps.
            ee_pose: Array of shape (T, 7) with [x, y, z, qw, qx, qy, qz] poses.
            hand: Array of shape (T,) with scalar hand positions [0=open, 1=closed].

        Raises:
            KeyError: If name is not in the config.
        """
        if name not in self.config:
            raise KeyError(
                f"Hand '{name}' not found in config. Available: {list(self.config.keys())}"
            )

        time = np.asarray(time)
        ee_pose = np.asarray(ee_pose)
        hand = np.asarray(hand)

        # Build trajectory interpolators
        ee_pose_traj = PiecewisePose.MakeLinear(
            times=time,
            poses=[
                self._posquat_to_rigid_transform(ee_pose[t]) for t in range(len(time))
            ],
        )
        hand_traj = PiecewisePolynomial.FirstOrderHold(
            breaks=time,
            samples=hand.reshape(1, -1),
        )

        self.trajectories[name] = {
            "time": time,
            "ee_pose": ee_pose,
            "hand": hand,
            "ee_pose_traj": ee_pose_traj,
            "hand_traj": hand_traj,
        }

    def add_waypoints(
        self,
        ee_poses: Union[List, npt.NDArray],
        hands: Union[List[float], npt.NDArray],
        times: Union[List[float], npt.NDArray],
        prefix: str = "waypoint",
        animated_name: str = "animated",
        line_color: List[float] = None,
        line_width: float = 4.0,
    ):
        """
        Add static waypoint visualizations at each breakpoint.

        Each waypoint creates a static hand that remains at the specified pose
        throughout the entire trajectory duration. If an animated robot is configured,
        it will follow the full trajectory through all waypoints.

        Args:
            ee_poses: Array of shape (N, 7) with waypoint poses [x, y, z, qw, qx, qy, qz].
            hands: Array of shape (N,) with hand positions at each waypoint [0=open, 1=closed].
            times: Array of timestamps for the trajectory (used for duration).
            prefix: Prefix for waypoint names (must match generate_waypoint_config prefix).
            animated_name: Name of the animated robot (must match generate_waypoint_config).
            line_color: RGBA color for the path line [r, g, b, a]. Defaults to orange.
            line_width: Width of the path line.

        Raises:
            KeyError: If waypoint config entries are missing.

        Example:
            config = SigmaHandVisualizer.generate_waypoint_config(4)
            visualizer = SigmaHandVisualizer(config)
            visualizer.add_waypoints(ee_poses, hands, times)
        """
        ee_poses = np.asarray(ee_poses)
        hands = np.asarray(hands)
        times = np.asarray(times)

        if line_color is None:
            line_color = [0.8, 0.4, 0.0, 1.0]  # Orange

        num_waypoints = len(ee_poses)
        num_timesteps = len(times)

        # Add static waypoint hands
        for i in range(num_waypoints):
            name = f"{prefix}_{i}"
            if name not in self.config:
                raise KeyError(
                    f"Waypoint '{name}' not found in config. "
                    f"Use generate_waypoint_config({num_waypoints}, prefix='{prefix}') "
                    f"when creating the visualizer."
                )

            # Create static trajectory: same pose at all timesteps
            static_ee_pose = np.tile(ee_poses[i], (num_timesteps, 1))
            static_hand = np.full(num_timesteps, hands[i])

            self.add_trajectory(
                name=name,
                time=times,
                ee_pose=static_ee_pose,
                hand=static_hand,
            )

        # Add animated robot trajectory if configured
        if animated_name in self.config:
            self.add_trajectory(
                name=animated_name,
                time=times,
                ee_pose=ee_poses,
                hand=hands,
            )

        # Draw line segments connecting waypoints
        if num_waypoints > 1:
            self.meshcat.SetLineSegments(
                path="waypoint_path",
                start=ee_poses[:-1, :3].T,
                end=ee_poses[1:, :3].T,
                line_width=line_width,
                rgba=Rgba(*line_color),
            )

    def _posquat_to_quatpos(self, posquat: npt.NDArray):
        """Convert posquat [x, y, z, qw, qx, qy, qz] to quatpos [qw, qx, qy, qz, x, y, z]"""
        assert len(posquat) == 7
        quatpos = np.zeros(7)
        quatpos[:4] = posquat[3:7]
        quatpos[4:7] = posquat[0:3]
        return quatpos

    def _posquat_to_rigid_transform(self, posquat: npt.NDArray):
        """Convert posquat [x, y, z, qw, qx, qy, qz] to quatpos [qw, qx, qy, qz, x, y, z]"""
        return RigidTransform(Quaternion(posquat[3:7]), posquat[0:3])

    def _hand_scalar_to_joint(self, hand_position: float | list[float]):
        """Convert scalar hand position [0, 1] to joint angles [q1, q2, q3, q4, q5, q6]"""
        q_des = (Q_CLOSE - Q_OPEN) * np.array(hand_position) + Q_OPEN
        return q_des * np.array([1.0, -1.0, 1.0, -1.0, -1.0, -1.0])

    def _calc_plant_positions(
        self, ee_pose: RigidTransform, hand: Union[float, list[float], npt.NDArray]
    ):
        """Calculate the full plant position vector from end-effector pose and hand scalar position."""
        position_drake = np.zeros(13)
        position_drake[:4] = ee_pose.rotation().ToQuaternion().wxyz()
        position_drake[4:7] = ee_pose.translation()
        position_drake[7:13] = self._hand_scalar_to_joint(hand)
        return position_drake

    def _cleanup_plant(self):
        """Explicitly delete old Drake objects to prevent memory leaks."""
        # Delete in reverse order of dependency
        if hasattr(self, "simulator"):
            del self.simulator
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "diagram"):
            del self.diagram
        if hasattr(self, "visualizer_dict"):
            self.visualizer_dict.clear()
            del self.visualizer_dict
        if hasattr(self, "scene_graph_dict"):
            self.scene_graph_dict.clear()
            del self.scene_graph_dict
        if hasattr(self, "plant_dict"):
            self.plant_dict.clear()
            del self.plant_dict

    def _setup_plant(self):
        """Set up Drake plant and visualizers based on config."""
        # Clean up any existing plant objects first
        self._cleanup_plant()

        builder = DiagramBuilder()
        self.plant_dict: Dict[str, MultibodyPlant] = {}
        self.scene_graph_dict: Dict[str, SceneGraph] = {}
        self.visualizer_dict: Dict[str, MeshcatVisualizer] = {}

        for name, hand_config in self.config.items():
            color = hand_config.get("color", "real")

            # Set up plant
            plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
            plant.set_name(f"plant_{name}")
            scene_graph.set_name(f"scene_graph_{name}")

            # Set up visualizer params
            visualizer_params = MeshcatVisualizerParams()
            visualizer_params.role = Role.kIllustration
            if not isinstance(color, str):
                visualizer_params.default_color = Rgba(*color)
            visualizer_params.prefix = f"plant_{name}"

            visualizer = MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                visualizer_params,
            )

            # Load model
            parser = Parser(plant, scene_graph)
            parser.package_map().AddPackageXml("carbonsix-models/package.xml")
            if color == "real":
                model_path = (
                    "directives/direct_teaching/sigma_hand_right_direct_teaching.yaml"
                )
            else:
                model_path = "directives/direct_teaching/sigma_hand_right_direct_teaching_no_color.yaml"
            ProcessModelDirectives(LoadModelDirectives(model_path), plant, parser)

            AddFrameTriadIllustration(
                frame=plant.GetFrameByName("ee"),
                length=0.05,
                radius=0.001,
                name=f"base_link_triad_{name}",
                scene_graph=scene_graph,
            )

            plant.Finalize()
            self.plant_dict[name] = plant
            self.scene_graph_dict[name] = scene_graph
            self.visualizer_dict[name] = visualizer

        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.simulator = Simulator(self.diagram, self.context)

    def _update_visualization_at_time(self, t: float):
        """Update all hand visualizations to show state at time t."""
        self.context.SetTime(t)

        for name in self.trajectories.keys():
            traj = self.trajectories[name]
            X_WE = RigidTransform(traj["ee_pose_traj"].value(t))
            hand_t = traj["hand_traj"].value(t)

            plant = self.plant_dict[name]
            plant_context = plant.GetMyContextFromRoot(self.context)

            X_BE = plant.CalcRelativeTransform(
                plant_context,
                plant.GetFrameByName("base_link"),
                plant.GetFrameByName("ee"),
            )
            X_WB = X_WE.multiply(X_BE.inverse())

            plant.SetPositions(
                plant_context,
                self._calc_plant_positions(X_WE, hand_t),
            )
            plant.SetFreeBodyPose(
                plant_context,
                plant.GetBodyByName("base_link"),
                X_WB,
            )
            self.visualizer_dict[name].ForcedPublish(
                self.visualizer_dict[name].GetMyContextFromRoot(self.context)
            )

    def publish_trajectory(self, save_html: str = None):
        """
        Publish trajectory visualization to Meshcat with API-based time control.

        This draws trajectory lines and adds UI elements (time slider, rate slider,
        play/stop buttons). Time control is done via API calls:
        - set_time(t): Set visualization to specific time
        - play_animation(): Play through the animation programmatically

        For Meshcat's built-in animation player, use publish_recording() instead.

        Args:
            save_html: If provided, save static HTML to this path.

        Raises:
            RuntimeError: If no trajectories have been added.
        """
        if not self.trajectories:
            raise RuntimeError("No trajectories added. Use add_trajectory() first.")

        # Get time range from trajectories
        t_min = float("inf")
        t_max = float("-inf")
        for traj in self.trajectories.values():
            t_min = min(t_min, traj["time"][0])
            t_max = max(t_max, traj["time"][-1])

        # Store time range
        self._t_min = t_min
        self._t_max = t_max

        # Add UI elements (for display, controlled via API)
        self.meshcat.AddSlider(
            name="time",
            min=t_min,
            max=t_max,
            step=0.01,
            value=t_min,
        )
        self.meshcat.AddSlider(
            name="rate",
            min=0.1,
            max=3.0,
            step=0.1,
            value=1.0,
        )
        self.meshcat.AddButton("Play")
        self.meshcat.AddButton("Stop")

        # Draw trajectory lines
        for name in self.trajectories.keys():
            traj = self.trajectories[name]
            color = self.config[name].get("color", "real")
            line_color = color if not isinstance(color, str) else [0.8, 0.4, 0.0, 1.0]

            self.meshcat.SetLineSegments(
                path=f"desired_ee_pose_{name}",
                start=traj["ee_pose"][:-1, :3].T,
                end=traj["ee_pose"][1:, :3].T,
                line_width=4.0,
                rgba=Rgba(*line_color),
            )

        # Set initial visualization to t_min
        self._update_visualization_at_time(t_min)
        print(
            f"Trajectory published. Use time slider or set_time() to control. Range: [{t_min}, {t_max}]"
        )

        if save_html:
            with open(save_html, "w") as f:
                f.write(self.meshcat.StaticHtml())
            print(f"Saved static HTML to {save_html}")

    def publish_recording(
        self, dt: float = 0.02, save_html: str = None, loop: bool = True
    ):
        """
        Record and publish trajectory animation using Meshcat's built-in animation.

        This uses StartRecording/StopRecording to create an animation that can be
        played using Meshcat's built-in animation controls. The animation loops
        by default.

        Args:
            dt: Time step between recorded frames.
            save_html: If provided, save static HTML to this path.
            loop: If True (default), animation loops continuously.

        Raises:
            RuntimeError: If no trajectories have been added.
        """
        if not self.trajectories:
            raise RuntimeError("No trajectories added. Use add_trajectory() first.")

        # Get time range from trajectories
        t_min = float("inf")
        t_max = float("-inf")
        for traj in self.trajectories.values():
            t_min = min(t_min, traj["time"][0])
            t_max = max(t_max, traj["time"][-1])

        # Calculate number of steps
        num_steps = int((t_max - t_min) / dt) + 1

        print(f"Recording animation: t=[{t_min}, {t_max}], dt={dt}, frames={num_steps}")

        self.meshcat.DeleteRecording()
        self.meshcat.StartRecording()

        for step in range(num_steps):
            t = t_min + dt * step
            self.context.SetTime(t)

            for name in self.trajectories.keys():
                traj = self.trajectories[name]
                X_WE = RigidTransform(traj["ee_pose_traj"].value(t))
                hand_t = traj["hand_traj"].value(t)

                plant = self.plant_dict[name]
                plant_context = plant.GetMyContextFromRoot(self.context)

                X_BE = plant.CalcRelativeTransform(
                    plant_context,
                    plant.GetFrameByName("base_link"),
                    plant.GetFrameByName("ee"),
                )
                X_WB = X_WE.multiply(X_BE.inverse())

                plant.SetPositions(
                    plant_context,
                    self._calc_plant_positions(X_WE, hand_t),
                )
                plant.SetFreeBodyPose(
                    plant_context,
                    plant.GetBodyByName("base_link"),
                    X_WB,
                )
                self.visualizer_dict[name].ForcedPublish(
                    self.visualizer_dict[name].GetMyContextFromRoot(self.context)
                )

        # Ensure we hit the final frame
        self._update_visualization_at_time(t_max)

        # Draw trajectory lines
        for name in self.trajectories.keys():
            traj = self.trajectories[name]
            color = self.config[name].get("color", "real")
            line_color = color if not isinstance(color, str) else [0.8, 0.4, 0.0, 1.0]

            self.meshcat.SetLineSegments(
                path=f"desired_ee_pose_{name}",
                start=traj["ee_pose"][:-1, :3].T,
                end=traj["ee_pose"][1:, :3].T,
                line_width=4.0,
                rgba=Rgba(*line_color),
            )

        self.meshcat.StopRecording()

        # Get the recorded animation and configure loop mode
        animation = self.meshcat.get_mutable_recording()
        if loop:
            animation.set_repetitions(1000)  # repeat 1000 times
            animation.set_loop_mode(MeshcatAnimation.LoopMode.kLoopRepeat)
        else:
            animation.set_loop_mode(MeshcatAnimation.LoopMode.kLoopOnce)
        self.meshcat.SetAnimation(animation)

        loop_str = "looping" if loop else "once"
        print(
            f"Recording published ({loop_str}). Use Meshcat's animation controls to play."
        )

        if save_html:
            with open(save_html, "w") as f:
                f.write(self.meshcat.StaticHtml())
            print(f"Saved static HTML to {save_html}")

    # Keep record_trajectory as an alias for backward compatibility
    def record_trajectory(
        self, dt: float = 0.02, num_steps: int = 51, save_html: str = None
    ):
        """Deprecated: Use publish_recording() instead."""
        print(
            "Warning: record_trajectory() is deprecated. Use publish_recording() or publish_trajectory() instead."
        )
        self.publish_recording(dt=dt, save_html=save_html)

    def get_state_at_time(self, t: float) -> Dict[str, dict]:
        """
        Query the animation state at a specific time.

        Args:
            t: Time in seconds to query.

        Returns:
            Dictionary mapping hand names to their state at time t:
            {
                "hand_name": {
                    "ee_pose": [x, y, z, qw, qx, qy, qz],
                    "hand": float (0=open, 1=closed),
                },
                ...
            }

        Raises:
            RuntimeError: If no trajectories have been added.
        """
        if not self.trajectories:
            raise RuntimeError("No trajectories added. Use add_trajectory() first.")

        result = {}
        for name, traj in self.trajectories.items():
            # Get interpolated pose at time t
            ee_pose_mat = traj["ee_pose_traj"].value(t)
            X_WE = RigidTransform(ee_pose_mat)

            # Extract position and quaternion
            pos = X_WE.translation().tolist()
            quat = X_WE.rotation().ToQuaternion()
            quat_wxyz = [quat.w(), quat.x(), quat.y(), quat.z()]

            # Get interpolated hand position
            hand_val = float(traj["hand_traj"].value(t).item())

            result[name] = {
                "ee_pose": pos + quat_wxyz,  # [x, y, z, qw, qx, qy, qz]
                "hand": hand_val,
            }

        return result

    def set_time(self, t: float):
        """
        Set the visualization to a specific time.

        Updates all hand visualizations and syncs the time slider.
        Requires that publish_trajectory() has been called first.

        Args:
            t: Time in seconds to display.

        Raises:
            RuntimeError: If no trajectories have been added.
        """
        if not self.trajectories:
            raise RuntimeError("No trajectories added. Use add_trajectory() first.")

        # Update visualization
        self._update_visualization_at_time(t)

        # Sync the slider
        try:
            self.meshcat.SetSliderValue("time", t)
        except Exception:
            pass  # Slider might not exist yet

    def play_animation(
        self, realtime_rate: float = 1.0, dt: float = 0.02, loop: bool = False
    ) -> None:
        """
        Play the animation in real-time (or at specified rate).

        Args:
            realtime_rate: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed, etc.)
            dt: Time step between frames in simulation time.
            loop: If True, loop the animation indefinitely (until interrupted).

        Raises:
            RuntimeError: If no trajectories have been added.
        """
        if not self.trajectories:
            raise RuntimeError("No trajectories added. Use add_trajectory() first.")

        # Get time range
        t_min = float("inf")
        t_max = float("-inf")
        for traj in self.trajectories.values():
            t_min = min(t_min, traj["time"][0])
            t_max = max(t_max, traj["time"][-1])

        # Calculate sleep time between frames
        sleep_time = dt / realtime_rate

        print(
            f"Playing animation: t=[{t_min}, {t_max}], rate={realtime_rate}x, dt={dt}"
        )

        try:
            while True:
                t = t_min
                while t < t_max:
                    self.set_time(t)
                    time_module.sleep(sleep_time)
                    t += dt

                # Ensure we hit the final frame exactly
                self.set_time(t_max)

                if not loop:
                    break

        except KeyboardInterrupt:
            print("\nAnimation stopped.")

        print("Animation complete.")

    def get_time_range(self) -> Dict[str, tuple]:
        """
        Get the valid time range for each trajectory.

        Returns:
            Dictionary mapping hand names to (start_time, end_time) tuples.
        """
        if not self.trajectories:
            return {}

        result = {}
        for name, traj in self.trajectories.items():
            times = traj["time"]
            result[name] = (float(times[0]), float(times[-1]))

        return result


if __name__ == "__main__":
    with open("batch_np.pkl", "rb") as f:
        trajectory_data = pickle.load(f)

    # Define config: which hands to visualize and their appearance
    config = {
        "tracking": {"color": [0.8, 0.8, 0.0, 0.4]},
        "actual": {"color": "real"},
    }

    # Initialize visualizer with config
    visualizer = SigmaHandVisualizer(config)

    # Add trajectory data
    times = [0.02 * i for i in range(50)]
    visualizer.add_trajectory(
        name="tracking",
        time=times,
        ee_pose=trajectory_data["action"]["ee_pose"][2],
        hand=trajectory_data["action"]["hand"][2],
    )
    visualizer.add_trajectory(
        name="actual",
        time=times,
        ee_pose=trajectory_data["action"]["ee_pose"][5],
        hand=trajectory_data["action"]["hand"][5],
    )

    # Record and publish using Meshcat's built-in animation
    visualizer.publish_recording()
    input("Press Enter to exit...")
