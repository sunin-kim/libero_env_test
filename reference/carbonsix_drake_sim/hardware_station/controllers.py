from typing import List

import numpy as np
import numpy.typing as npt
from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    Context,
    Value,
    RigidTransform,
    Quaternion,
)


class GravityCompensator(LeafSystem):
    """
    Computes gravity for the given system and applies it to the output port.
    """

    def __init__(self, plant: MultibodyPlant, model_instance_name):
        super().__init__()
        # Create a copy of the plant.
        self.ctrl_plant = plant.Clone()
        self.set_name(f"{model_instance_name}_GravityCompensator")

        self.state_input_port = self.DeclareVectorInputPort(
            name="plant_state",
            size=self.ctrl_plant.num_positions() + self.ctrl_plant.num_velocities(),
        )
        self.model_idx = self.ctrl_plant.GetModelInstanceByName(model_instance_name)

        self.gravity_torque_output_port = self.DeclareVectorOutputPort(
            name="gravity_torque",
            size=plant.num_velocities(self.model_idx),
            calc=self.calc_output,
        )
        self.plant_context = self.ctrl_plant.CreateDefaultContext()

    def calc_output(self, context: Context, output: Value) -> None:
        state = self.state_input_port.Eval(context)
        self.ctrl_plant.SetPositionsAndVelocities(self.plant_context, state)
        # Offset gravity torque by negation.
        gravity_torque = -self.ctrl_plant.CalcGravityGeneralizedForces(
            self.plant_context
        )
        gravity_torque = self.ctrl_plant.GetVelocitiesFromArray(
            model_instance=self.model_idx, v=gravity_torque
        )
        output.SetFromVector(gravity_torque)


class PoseController(LeafSystem):
    """
    Computes gravity for the given system and applies it to the output port.
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        model_instance_name: str,
        p_gains_rotation: List[float],
        d_gains_rotation: List[float],
        p_gains_translation: List[float],
        d_gains_translation: List[float],
    ):
        super().__init__()
        # Create a copy of the plant.
        self.ctrl_plant = plant.Clone()
        self.set_name(f"{model_instance_name}_GravityCompensator")
        self.model_idx = self.ctrl_plant.GetModelInstanceByName(model_instance_name)

        self.p_gains_rotation = np.array(p_gains_rotation, dtype=float)
        self.d_gains_rotation = np.array(d_gains_rotation, dtype=float)
        self.p_gains_translation = np.array(p_gains_translation, dtype=float)
        self.d_gains_translation = np.array(d_gains_translation, dtype=float)

        self.feedforward_input_port = self.DeclareVectorInputPort(
            name="feedforward_spatial_force",
            size=6,
        )
        self.current_state_input_port = self.DeclareVectorInputPort(
            name="current_state",
            size=13,
        )
        self.desired_pose_input_port = self.DeclareVectorInputPort(
            name="desired_pose",
            size=7,
        )

        self.applied_generalized_force_output_port = self.DeclareVectorOutputPort(
            name="applied_generalized_torque",
            size=plant.num_velocities(),
            calc=self.calc_output,
        )
        self.plant_context = self.ctrl_plant.CreateDefaultContext()

    def calc_output(self, context: Context, output: Value) -> None:
        state = self.current_state_input_port.Eval(context)
        current_pose = state[:7]
        current_velocity = state[7:]
        desired_pose = self.desired_pose_input_port.Eval(context)
        feedforward_force = self.feedforward_input_port.Eval(context)

        # Compute the error in position and orientation
        e_AB_W = self.calc_pose_error(
            X_WA=self.pos_quat_to_rigid_transform(desired_pose),
            X_WB=self.pos_quat_to_rigid_transform(current_pose),
        )

        tau_AB_W = (
            -self.p_gains_rotation * e_AB_W[:3]
            - self.d_gains_rotation * current_velocity[:3]
        )
        f_AB_W = (
            -self.p_gains_translation * e_AB_W[3:]
            - self.d_gains_translation * current_velocity[3:]
        )

        recovery_force = np.concatenate((tau_AB_W, f_AB_W))
        model_force = feedforward_force + recovery_force

        output_vector = np.zeros(self.ctrl_plant.num_velocities(), dtype=float)
        self.ctrl_plant.SetVelocitiesInArray(self.model_idx, model_force, output_vector)

        output.SetFromVector(output_vector)

    def pos_quat_to_rigid_transform(self, pos_quat: npt.NDArray):
        quat_norm = np.linalg.norm(pos_quat[:4])
        if np.abs(quat_norm - 1.0) > 1e-1:
            raise ValueError(
                f"Quaternion norm is not close to 1: {quat_norm} with error > 0.1"
            )
        return RigidTransform(
            Quaternion(pos_quat[:4] / np.linalg.norm(pos_quat[:4])),
            np.array(pos_quat[4:7], dtype=float),
        )

    def calc_pose_error(self, X_WA: RigidTransform, X_WB: RigidTransform):
        """
        Computes a pseudo-velocity like error between two poses X_WA and X_WB in the world frame W.
        """
        # rotation matrix R_AB
        R_WA = X_WA.rotation()
        R_WB = X_WB.rotation()
        AA_AB = R_WA.inverse().multiply(R_WB).ToAngleAxis()
        w_AB_A = AA_AB.axis() * AA_AB.angle()
        w_AB_W = R_WA.multiply(w_AB_A)
        # translation
        v_AB_W = X_WB.translation() - X_WA.translation()

        return np.concatenate((w_AB_W, v_AB_W))


class SigmaHandController(LeafSystem):
    """
    Computes torque for the Sigma hand.

    Joint angle definitions:
        - Q_CLOSE: Joint angle [rad] corresponding to the **fully closed** state
                   of the gripper. At this position the fingertips make contact
                   and cannot close further.

        - Q_OPEN: Joint angle [rad] corresponding to the **fully open** state
                  of the gripper, where the mechanism has reached its physical
                  open limit and the hardware linkage contacts its stop.
    Input:
        - `pos_cmd` ∈ [0, 1] : Normalized desired finger position
            * pos_cmd = 0 → fully closed  (q_des = Q_CLOSE)
            * pos_cmd = 1 → fully open    (q_des = Q_OPEN)
    Output:
        - tau: Commanded motor torque
        - Send normalized position ∈ [0, 1] and the joint velocity (v), matching the
          same state values that a real dynamixel-driver reports
    """

    Q_CLOSE = -0.74734999  # rad (fully closed)
    Q_OPEN = 0.445058959  # rad (fully opend)

    def __init__(
        self,
        plant: MultibodyPlant,
        model_instance_name: str,
        kp: float,
        kd: float,
        discrete_cmd: bool = False,
        threshold: float = 0.2,
    ):
        super().__init__()

        self._plant = plant
        self._model_idx = plant.GetModelInstanceByName(model_instance_name)
        self._nq = plant.num_positions(self._model_idx)  # sigma_hand: 6
        self._nv = plant.num_velocities(self._model_idx)  # sigma_hand: 6

        # Gripper command parameters
        self.discrete_cmd = discrete_cmd
        self.threshold = threshold

        assert self._nq == 6 and self._nv == 6, (
            "Current SigmaHandController only supports sigma_hand with 6 DOF."
        )

        self._kp = float(kp)
        self._kd = float(kd)

        # --- Input ports ---
        # 0.0 (fully open) ~ 1.0 (fully closed) command
        self.position_cmd_input_port = self.DeclareVectorInputPort("position_cmd", 1)
        self.state_input_port = self.DeclareVectorInputPort(
            "state", self._nq + self._nv
        )
        # feedforward torque
        self.feedforward_input_port = self.DeclareVectorInputPort(
            "feedforward_torque", 1
        )

        # --- Output port ---
        # actuation for sigma_hand: 1 DOF (JL1_motor)
        self.tau_output_port = self.DeclareVectorOutputPort("tau", 1, self._calc_output)

        self.pos_mm_output_port = self.DeclareVectorOutputPort(
            "position_mm_out",
            1,
            self._cal_pos_mm,
        )

        self.vel_mm_output_port = self.DeclareVectorOutputPort(
            "velocity_mm_out",
            1,
            self._cal_vel_mm,
        )

    def _calc_output(self, context, output):
        state = self.state_input_port.Eval(context)
        q, v = state[: self._nq], state[self._nq :]
        # only JL1_motor is controlled, others are coupled
        q0, v0 = float(q[0]), float(v[0])

        force_cmd = float(self.feedforward_input_port.Eval(context)[0])
        pos_cmd = float(self.position_cmd_input_port.Eval(context)[0])
        pos_cmd = np.clip(pos_cmd, 0.0, 1.0)  # normalize to [0.0, 1.0]

        if self.discrete_cmd:
            if pos_cmd > self.threshold:
                # fully closed
                q_des = self.Q_CLOSE
            else:
                # the mid-range between fully open and fully closed
                q_des = 0.5 * (self.Q_OPEN + self.Q_CLOSE)
        else:
            q_des = (self.Q_CLOSE - self.Q_OPEN) * pos_cmd + self.Q_OPEN

        pose_trq = self._kp * (q_des - q0)
        vel_trq = self._kd * (-v0)
        tau = pose_trq + vel_trq + force_cmd
        output.SetAtIndex(0, tau)

    def _cal_pos_mm(self, context, output):
        state = self.state_input_port.Eval(context)
        q = state[: self._nq]
        q0 = float(q[0])
        mm = self._rad_to_normalized_position(q0)
        output.SetFromVector([mm])

    def _cal_vel_mm(self, context, output):
        state = self.state_input_port.Eval(context)
        v = state[self._nq :]
        v0 = float(v[0])
        # Current dynamixel also returns joint velocity directly
        output.SetFromVector([v0])

    def _rad_to_normalized_position(self, q: float) -> float:
        return (q - self.Q_OPEN) / (self.Q_CLOSE - self.Q_OPEN)
