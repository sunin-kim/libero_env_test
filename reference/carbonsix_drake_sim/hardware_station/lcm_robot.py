import time

import numpy as np
import numpy.typing as npt
from drake import (
    lcmt_iiwa_command,
    lcmt_iiwa_status,
    lcmt_robot_state,
    lcmt_schunk_wsg_command,
    lcmt_drake_signal,
)

from pydrake.all import (
    BasicVector,
    Context,
    LeafSystem,
    Value,
    ValueProducer,
)

"""
LCM Status and Command Sender and Receiver for articulated robots.

Note that for the lcmt type, we will use lcmt_iiwa_status / lcmt_iiwa_command
as these support variable number of joints.
"""


class RobotStatusSender(LeafSystem):
    """
    Convert Robot status to LCM message.
    -----------------------------------------
    Input Ports:
        - position_measured: joint positions
        - velocity_estimated: joint velocities
        - position_commanded: commanded joint positions
        - torque_commanded: commanded joint torques

    Output Ports:
        - lcm_message: LCM message of lcmt_iiwa_status.
    """

    def __init__(self, n_joints: int) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.joint_position_input_port = self.DeclareVectorInputPort(
            name="position_measured", size=self.n_joints
        )
        self.joint_velocity_input_port = self.DeclareVectorInputPort(
            name="velocity_estimated", size=self.n_joints
        )
        self.joint_position_commanded_input_port = self.DeclareVectorInputPort(
            name="position_commanded", size=self.n_joints
        )
        self.joint_torque_commanded_input_port = self.DeclareVectorInputPort(
            name="torque_commanded", size=self.n_joints
        )
        self.joint_torque_measured_input_port = self.DeclareVectorInputPort(
            name="torque_measured", size=self.n_joints
        )
        self.joint_torque_external_input_port = self.DeclareVectorInputPort(
            name="torque_external", size=self.n_joints
        )

        self.DeclareAbstractOutputPort(
            name="status",
            alloc=lambda: Value(lcmt_iiwa_status()),
            calc=self.calc_output,
        )

    def calc_output(self, context: Context, output: Value) -> None:
        msg = lcmt_iiwa_status()
        msg.utime = time.monotonic_ns()
        msg.num_joints = self.n_joints
        msg.joint_position_measured = self.joint_position_input_port.Eval(context)
        msg.joint_velocity_estimated = self.joint_velocity_input_port.Eval(context)
        msg.joint_position_commanded = self.joint_position_commanded_input_port.Eval(
            context
        )

        msg.joint_torque_commanded = self.joint_torque_commanded_input_port.Eval(
            context
        )
        msg.joint_position_ipo = self.joint_position_input_port.Eval(
            context
        )  # Not used.
        msg.joint_torque_measured = self.joint_torque_measured_input_port.Eval(context)
        msg.joint_torque_external = self.joint_torque_external_input_port.Eval(context)
        output.set_value(msg)


class RobotStatusReceiver(LeafSystem):
    """
    Convert LCM Message to Robot status.
    -----------------------------------------
    Input Ports:
        - lcm_message: LCM message of lcmt_iiwa_status

    Output Ports:
        - position_measured: joint positions
        - velocity_estimated: joint velocities
        - position_commanded: commanded joint positions
        - torque_commanded: commanded joint torques
    """

    def __init__(self, n_joints: int, default_joint_positions: npt.NDArray) -> None:
        super().__init__()
        assert len(default_joint_positions) == n_joints
        self.n_joints = n_joints

        self.message_input_port = self.DeclareAbstractInputPort(
            name="dsr_status", model_value=Value(lcmt_iiwa_status())
        )

        self.joint_position_output_port = self.DeclareVectorOutputPort(
            name="position_measured",
            size=self.n_joints,
            calc=self.calc_output_joint_position,
        )
        self.joint_velocity_output_port = self.DeclareVectorOutputPort(
            name="velocity_estimated",
            size=self.n_joints,
            calc=self.calc_output_joint_velocity,
        )
        self.joint_position_commanded_output_port = self.DeclareVectorOutputPort(
            self.DeclareVectorOutputPort(
                name="position_commanded",
                size=self.n_joints,
                calc=self.calc_output_joint_position_commanded,
            )
        )
        self.joint_torque_commanded_output_port = self.DeclareVectorOutputPort(
            self.DeclareVectorOutputPort(
                name="torque_commanded",
                size=self.n_joints,
                calc=self.calc_output_joint_torque_commanded,
            )
        )

        # Default Joint positions as discrete state.
        self.default_joint_position = self.DeclareDiscreteState(default_joint_positions)

        self.status_msg_cache = self.DeclareCacheEntry(
            "status_msg_cache",
            ValueProducer(
                allocate=Value(lcmt_iiwa_status()).Clone(),
                calc=self.calc_status_msg,
            ),
            {
                self.input_port_ticket(self.message_input_port.get_index()),
                self.discrete_state_ticket(self.default_joint_position),
            },
        )

    def calc_status_msg(self, context: Context, cache_value: Value) -> None:
        new_status_msg = self.message_input_port.Eval(context)
        # Fallback in case of no message.
        if new_status_msg.encode() == lcmt_iiwa_status().encode():
            new_status_msg = lcmt_iiwa_status()
            new_status_msg.joint_position_measured = np.zeros(self.n_joints)
            new_status_msg.joint_velocity_estimated = np.zeros(self.n_joints)
            new_status_msg.joint_position_commanded = np.zeros(self.n_joints)
            new_status_msg.joint_torque_commanded = np.zeros(self.n_joints)
        cache_value.set_value(new_status_msg)

    def calc_output_joint_position(self, context: Context, output: BasicVector) -> None:
        # Retrieve status message from cache.
        msg = self.get_cache_entry(self.status_msg_cache.cache_index()).Eval(context)
        output.SetFromVector(msg.joint_position_measured)

    def calc_output_joint_velocity(self, context: Context, output: BasicVector) -> None:
        # Retrieve status message from cache.
        msg = self.get_cache_entry(self.status_msg_cache.cache_index()).Eval(context)
        output.SetFromVector(msg.joint_velocity_estimated)

    def calc_output_joint_position_commanded(
        self, context: Context, output: BasicVector
    ) -> None:
        # Retrieve status message from cache.
        msg = self.get_cache_entry(self.status_msg_cache.cache_index()).Eval(context)
        output.SetFromVector(msg.joint_position_commanded)

    def calc_output_joint_torque_commanded(
        self, context: Context, output: BasicVector
    ) -> None:
        # Retrieve status message from cache.
        msg = self.get_cache_entry(self.status_msg_cache.cache_index()).Eval(context)
        output.SetFromVector(msg.joint_torque_commanded)


class RobotCommandSender(LeafSystem):
    """
    Convert Robot command to LCM message.
    -----------------------------------------
    Input Ports:
        - position: commanded joint positions
        - feedforward_torque: commanded joint torques

    Output Ports:
        - lcm_message: LCM message of lcmt_iiwa_command
    """

    def __init__(self, n_joints: int) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.joint_position_input_port = self.DeclareVectorInputPort(
            name="position_commanded", size=self.n_joints
        )
        self.joint_torque_input_port = self.DeclareVectorInputPort(
            name="torque_commanded", size=self.n_joints
        )
        self.DeclareAbstractOutputPort(
            name="dsr_command",
            alloc=lambda: Value(lcmt_iiwa_command()),
            calc=self.calc_output,
        )

    def calc_output(self, context: Context, output: Value) -> None:
        msg = lcmt_iiwa_command()
        msg.utime = time.monotonic_ns()
        msg.num_joints = self.n_joints
        msg.joint_position = self.joint_position_input_port.Eval(context)
        msg.num_torques = self.n_joints
        msg.joint_torque = self.joint_torque_input_port.Eval(context)
        output.set_value(msg)


class RobotCommandReceiver(LeafSystem):
    """
    Convert LCM Message to Robot command.
    -----------------------------------------
    Input Ports:
        - lcm_message: LCM message of lcmt_iiwa_command

    Output Ports:
        - position_commanded: joint positions
        - torque_commanded: feedforward commanded torques.
    """

    def __init__(self, n_joints: int, default_joint_positions: npt.NDArray) -> None:
        super().__init__()
        self.n_joints = n_joints
        assert len(default_joint_positions) == n_joints

        self.message_input_port = self.DeclareAbstractInputPort(
            name="command", model_value=Value(lcmt_iiwa_command())
        )

        self.joint_position_output_port = self.DeclareVectorOutputPort(
            name="position_commanded",
            size=n_joints,
            calc=self.calc_output_joint_position,
        )
        self.joint_torque_output_port = self.DeclareVectorOutputPort(
            name="torque_commanded",
            size=n_joints,
            calc=self.calc_output_joint_torque,
        )
        # Default Joint positions as discrete state.
        self.default_joint_position_vec = default_joint_positions
        self.default_joint_position = self.DeclareDiscreteState(default_joint_positions)

        self.command_msg_cache = self.DeclareCacheEntry(
            "command_msg_cache",
            ValueProducer(
                allocate=Value(lcmt_iiwa_command()).Clone,
                calc=self.calc_command_msg,
            ),
            {
                self.input_port_ticket(self.message_input_port.get_index()),
                self.discrete_state_ticket(self.default_joint_position),
            },
        )

    def calc_command_msg(self, context: Context, cache_value: Value) -> None:
        new_command_msg = self.message_input_port.Eval(context)
        # Fallback in case of no message.
        if new_command_msg.encode() == lcmt_iiwa_command().encode():
            new_command_msg = lcmt_iiwa_command()
            new_command_msg.joint_position = self.default_joint_position_vec
            new_command_msg.joint_torque = np.zeros(self.n_joints)

        cache_value.set_value(new_command_msg)

    def calc_output_joint_position(self, context: Context, output: BasicVector) -> None:
        # Retrieve command message from cache.
        msg = self.get_cache_entry(self.command_msg_cache.cache_index()).Eval(context)
        output.SetFromVector(msg.joint_position)

    def calc_output_joint_torque(self, context: Context, output: BasicVector) -> None:
        # Retrieve command message from cache.
        msg = self.get_cache_entry(self.command_msg_cache.cache_index()).Eval(context)
        output.SetFromVector(msg.joint_torque)


class PoseStatusSender(LeafSystem):
    """
    Convert Pose status to LCM message.
    -----------------------------------------
    Input Ports:
        - position: ee pose

    Output Ports:
        - lcm_message: LCM message of lcmt_robot_state
    """

    def __init__(self) -> None:
        super().__init__()
        self.position_input_port = self.DeclareVectorInputPort(name="position", size=7)
        self.DeclareAbstractOutputPort(
            name="status",
            alloc=lambda: Value(lcmt_robot_state()),
            calc=self.calc_output,
        )

    def calc_output(self, context: Context, output: Value) -> None:
        msg = lcmt_robot_state()
        msg.utime = time.monotonic_ns()
        msg.num_joints = 7
        joint_position_quat_pos = self.position_input_port.Eval(context)
        joint_position_pos_quat = np.concatenate(
            (joint_position_quat_pos[4:], joint_position_quat_pos[:4])
        )
        msg.joint_position = joint_position_pos_quat
        msg.joint_name = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        output.set_value(msg)


class PoseCommandReceiver(LeafSystem):
    """
    Convert LCM Message to Robot command.
    -----------------------------------------
    Input Ports:
        - lcm_message: LCM message of lcmt_iiwa_command

    Output Ports:
        - position_commanded: joint positions
        - torque_commanded: feedforward commanded torques.
    """

    def __init__(self, default_pose: npt.NDArray) -> None:
        super().__init__()
        assert len(default_pose) == 7

        self.message_input_port = self.DeclareAbstractInputPort(
            name="command", model_value=Value(lcmt_robot_state())
        )

        self.position_output_port = self.DeclareVectorOutputPort(
            name="position",
            size=7,
            calc=self.calc_output_position,
        )
        # Default Joint positions as discrete state.
        self.default_pose_pos_quat = np.concatenate(
            (
                default_pose[4:],  # x, y, z
                default_pose[:4],  # qw, qx, qy, qz
            )
        )
        self.default_pose = self.DeclareDiscreteState(self.default_pose_pos_quat)
        self.command_msg_cache = self.DeclareCacheEntry(
            "command_msg_cache",
            ValueProducer(
                allocate=Value(lcmt_robot_state()).Clone,
                calc=self.calc_command_msg,
            ),
            {
                self.input_port_ticket(self.message_input_port.get_index()),
                self.discrete_state_ticket(self.default_pose),
            },
        )

    def calc_command_msg(self, context: Context, cache_value: Value) -> None:
        new_command_msg = self.message_input_port.Eval(context)
        # Fallback in case of no message.
        if new_command_msg.encode() == lcmt_robot_state().encode():
            new_command_msg = lcmt_robot_state()
            new_command_msg.utime = 0  # time.monotonic_ns()
            new_command_msg.num_joints = 7
            new_command_msg.joint_position = self.default_pose_pos_quat
            new_command_msg.joint_name = [
                "x",
                "y",
                "z",
                "qw",
                "qx",
                "qy",
                "qz",
            ]
        cache_value.set_value(new_command_msg)

    def calc_output_position(self, context: Context, output: BasicVector) -> None:
        # Retrieve command message from cache.
        msg = self.get_cache_entry(self.command_msg_cache.cache_index()).Eval(context)
        joint_position_quat_pos = np.concatenate(
            (
                msg.joint_position[3:],  # x, y, z
                msg.joint_position[:3],  # qw, qx, qy, qz
            )
        )
        output.SetFromVector(joint_position_quat_pos)


class HandStatusSender(LeafSystem):
    """
    Convert Hand status to LCM message.

    Input Ports:
        - actual_position:      current gripper opening [0.0, 1.0]=[open, closed]
        - actual_speed:         current speed in mm/s
        - actual_torque:        measured gripping force (N)
        - target_position:      target gripper opening [0.0, 1.0]=[open, closed]

    Output Ports:
        - lcm_message: LCM message of lcmt_drake_signal
    """

    def __init__(self) -> None:
        super().__init__()
        self.position_input_port = self.DeclareVectorInputPort("actual_position", 1)
        self.speed_input_port = self.DeclareVectorInputPort("actual_speed", 1)
        self.force_input_port = self.DeclareVectorInputPort("actual_torque", 1)
        self.target_position_input_port = self.DeclareVectorInputPort(
            "target_position", 1
        )

        # Output LCM message
        self.DeclareAbstractOutputPort(
            name="status",
            alloc=lambda: Value(lcmt_drake_signal()),
            calc=self.calc_output,
        )

    def calc_output(self, context, output):
        msg = lcmt_drake_signal()

        msg.dim = 4
        msg.coord = [
            "actual_position",
            "actual_speed",
            "actual_torque",
            "target_position",
        ]
        msg.val = [0.0] * msg.dim
        msg.timestamp = time.monotonic_ns()

        msg.val[0] = float(self.position_input_port.Eval(context)[0])
        msg.val[1] = float(self.speed_input_port.Eval(context)[0])
        msg.val[2] = float(self.force_input_port.Eval(context)[0])
        msg.val[3] = float(self.target_position_input_port.Eval(context)[0])

        output.set_value(msg)


class HandStatusReceiver(LeafSystem):
    """
    Convert LCM message to Hand status

    Input Ports:
        - lcm_message: LCM message of lcmt_drake_signal

    Output Ports:
        - actual_position:          actual position
        - actual_speed:             actual speed
        - actual_torque:            actual torque
        - target_position:          target position
    """

    def __init__(self) -> None:
        super().__init__()

        # Input LCM Message
        self.message_input_port = self.DeclareAbstractInputPort(
            name="hand_status",
            model_value=Value(lcmt_drake_signal()),
        )

        # Output ports
        self.position_output_port = self.DeclareVectorOutputPort(
            name="actual_position",
            size=1,
            calc=self.calc_output_position,
        )

        self.speed_output_port = self.DeclareVectorOutputPort(
            name="actual_speed",
            size=1,
            calc=self.calc_output_speed,
        )

        self.force_output_port = self.DeclareVectorOutputPort(
            name="actual_torque",
            size=1,
            calc=self.calc_output_force,
        )

        self.target_position_output_port = self.DeclareVectorOutputPort(
            name="target_position",
            size=1,
            calc=self.calc_output_target_position,
        )

        # Cache for storing last msg (fallback if no msg received)
        self.status_msg_cache = self.DeclareCacheEntry(
            "status_msg_cache",
            ValueProducer(
                allocate=lambda: Value(lcmt_drake_signal()),
                calc=self.calc_status_msg,
            ),
            {self.input_port_ticket(self.message_input_port.get_index())},
        )

    def calc_status_msg(self, context, cache_value: Value):
        msg = self.message_input_port.Eval(context)

        # Fallback: if msg is empty/uninitialized
        empty_msg = lcmt_drake_signal()
        if msg.encode() == empty_msg.encode():
            msg = lcmt_drake_signal()
            msg.dim = 4
            msg.coord = [
                "actual_position",
                "actual_speed",
                "actual_torque",
                "target_position",
            ]
            msg.val = [0.0] * msg.dim

        cache_value.set_value(msg)

    # ---- Output Calculations ----
    def calc_output_position(self, context, output: BasicVector):
        msg = self.status_msg_cache.Eval(context)
        output.SetAtIndex(0, msg.val[0])

    def calc_output_speed(self, context, output: BasicVector):
        msg = self.status_msg_cache.Eval(context)
        output.SetAtIndex(0, msg.val[1])

    def calc_output_force(self, context, output: BasicVector):
        msg = self.status_msg_cache.Eval(context)
        output.SetAtIndex(0, msg.val[2])

    def calc_output_target_position(self, context, output: BasicVector):
        msg = self.status_msg_cache.Eval(context)
        output.SetAtIndex(0, msg.val[3])


class HandCommandSender(LeafSystem):
    """
    Convert Hand command to LCM message.

    Input Ports:
        - position_mm: commanded gripper opening (mm)
        - force:       commanded gripping force (N)

    Output Ports:
        - lcm_message: LCM message of lcmt_schunk_wsg_command
    """

    def __init__(self) -> None:
        super().__init__()
        self.position_input_port = self.DeclareVectorInputPort(
            name="position_mm", size=1
        )
        self.force_input_port = self.DeclareVectorInputPort(name="force", size=1)

        self.DeclareAbstractOutputPort(
            name="hand_command",
            alloc=lambda: Value(lcmt_schunk_wsg_command()),
            calc=self.calc_output,
        )

    def calc_output(self, context: Context, output: Value) -> None:
        msg = lcmt_schunk_wsg_command()

        msg.utime = time.monotonic_ns()

        # Extract scalar inputs
        msg.target_position_mm = float(self.position_input_port.Eval(context)[0])
        msg.force = float(self.force_input_port.Eval(context)[0])

        output.set_value(msg)


class HandCommandReceiver(LeafSystem):
    """
    Convert LCM Message to Hand command.

    Input Ports:
        - lcm_message: LCM message of lcmt_schunk_wsg_command

    Output Ports:
        - target_position_mm:   target gripper opening (mm)
        - force:                gripping force (N)
    """

    def __init__(self, default_position_mm: float, default_force: float) -> None:
        super().__init__()
        # Store defaults
        self.default_position_mm = default_position_mm
        self.default_force = default_force

        # Input port: LCM message
        self.message_input_port = self.DeclareAbstractInputPort(
            name="hand_command",
            model_value=Value(lcmt_schunk_wsg_command()),
        )

        # Output: target_position_mm (1 DOF)
        self.position_output_port = self.DeclareVectorOutputPort(
            name="target_position_mm",
            size=1,
            calc=self.calc_output_position,
        )

        # Output: force (1 DOF)
        self.force_output_port = self.DeclareVectorOutputPort(
            name="force",
            size=1,
            calc=self.calc_output_force,
        )

        # Cache entry
        self.command_msg_cache = self.DeclareCacheEntry(
            "command_msg_cache",
            ValueProducer(
                allocate=lambda: Value(lcmt_schunk_wsg_command()),
                calc=self.calc_command_msg,
            ),
            {self.input_port_ticket(self.message_input_port.get_index())},
        )

    def calc_command_msg(self, context: Context, cache_value: Value) -> None:
        msg = self.message_input_port.Eval(context)

        # If no message → fallback to defaults
        empty = lcmt_schunk_wsg_command()
        if msg.encode() == empty.encode():
            msg = lcmt_schunk_wsg_command()
            msg.target_position_mm = self.default_position_mm
            msg.force = self.default_force
        cache_value.set_value(msg)

    # ---- Output Ports ----
    def calc_output_position(self, context: Context, output: BasicVector):
        msg = self.command_msg_cache.Eval(context)
        output.SetAtIndex(0, msg.target_position_mm)

    def calc_output_force(self, context: Context, output: BasicVector):
        msg = self.command_msg_cache.Eval(context)
        output.SetAtIndex(0, msg.force)
