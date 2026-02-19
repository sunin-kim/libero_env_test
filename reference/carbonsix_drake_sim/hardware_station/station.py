import dataclasses as dc
import typing
from pathlib import Path
import cv2
import lcm

import numpy as np
from pydrake.all import (
    Adder,
    AddFrameTriadIllustration,
    AddMultibodyPlant,
    ApplyLcmBusConfig,
    ApplyVisualizationConfig,
    Diagram,
    DiagramBuilder,
    DrakeLcmParams,
    JointActuatorIndex,
    LcmBuses,
    ModelDirective,
    ModelDirectives,
    ModelInstanceInfo,
    MultibodyPlant,
    MultibodyPlantConfig,
    Parser,
    PdControllerGains,
    ProcessModelDirectives,
    RigidTransform,
    RotationMatrix,
    Quaternion,
    RollPitchYaw,
    SceneGraph,
    SimulatorConfig,
    VisualizationConfig,
    StartMeshcat,
    Simulator,
    StateInterpolatorWithDiscreteDerivative,
    ApplySimulatorConfig,
    LightParameter,
    EquirectangularMap,
    EnvironmentMap,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    PassThrough,
    Demultiplexer,
    MatrixGain,
)

from drake import (
    lcmt_iiwa_command,
    lcmt_iiwa_status,
    lcmt_robot_state,
    lcmt_schunk_wsg_command,
    lcmt_drake_signal,
)
from pydrake.common.yaml import yaml_load_typed
from pydrake.systems.sensors import ApplyCameraConfig, CameraConfig
from carbonsix_drake_sim.hardware_station.lcm_robot import (
    RobotCommandReceiver,
    RobotStatusSender,
    PoseCommandReceiver,
    PoseStatusSender,
    HandCommandReceiver,
    HandStatusSender,
)
from carbonsix_drake_sim.hardware_station.controllers import (
    GravityCompensator,
    PoseController,
    SigmaHandController,
)


@dc.dataclass
class JointPdControlledDriver:
    p_gains: list[float] = dc.field(default_factory=list)
    d_gains: list[float] = dc.field(default_factory=list)
    lcm_publish_bus: str = "default"
    lcm_subscribe_bus: str = "default"
    lcm_publish_channel: str = "STATUS"
    lcm_subscribe_channel: str = "COMMAND"
    lcm_publish_hz: float = 1000.0


@dc.dataclass
class PosePdControlledDriver:
    p_gains_rotation: list[float] = dc.field(default_factory=list)
    d_gains_rotation: list[float] = dc.field(default_factory=list)
    p_gains_translation: list[float] = dc.field(default_factory=list)
    d_gains_translation: list[float] = dc.field(default_factory=list)
    lcm_publish_bus: str = "default"
    lcm_subscribe_bus: str = "default"
    lcm_publish_channel: str = "STATUS"
    lcm_subscribe_channel: str = "COMMAND"
    lcm_publish_hz: float = 1000.0


@dc.dataclass
class HandControlledDriver:
    p_gains: list[float] = dc.field(default_factory=list)
    d_gains: list[float] = dc.field(default_factory=list)
    discrete_cmd: bool = False
    threshold: float = 0.2
    lcm_publish_bus: str = "default"
    lcm_subscribe_bus: str = "default"
    lcm_publish_channel: str = "STATUS"
    lcm_subscribe_channel: str = "COMMAND"
    lcm_publish_hz: float = 100.0


@dc.dataclass
class RandomPose:
    body: str = ""
    max_translation: list[float] = dc.field(default_factory=list)
    min_translation: list[float] = dc.field(default_factory=list)
    max_rotation_deg: list[float] = dc.field(default_factory=list)
    min_rotation_deg: list[float] = dc.field(default_factory=list)


@dc.dataclass
class RandomJoint:
    max_value: list[float] = dc.field(default_factory=list)
    min_value: list[float] = dc.field(default_factory=list)


@dc.dataclass
class Scenario:
    """
    Scenario class.

    Defines the YAML format for a (possibly stochastic) scenario to be
    simulated.
    """

    # Random seed for any random elements in the scenario. The seed is always
    # deterministic in the `Scenario`; a caller who wants randomness must
    # populate this value from their own randomness.
    random_seed: int = 0

    # The maximum simulation time (in seconds). The simulator will attempt to
    # run until this time and then terminate.
    simulation_duration: float = np.inf

    # Simulator configuration (integrator and publisher parameters).
    simulator_config: SimulatorConfig = dc.field(
        default_factory=lambda: SimulatorConfig()
    )

    # Plant configuration (time step and contact parameters).
    plant_config: MultibodyPlantConfig = MultibodyPlantConfig(
        discrete_contact_approximation="lagged"
    )

    # All the fully deterministic elements of the simulation.
    directives: list[ModelDirective] = dc.field(default_factory=list)

    # Randomized initial conditions.
    initial_conditions: typing.Mapping[
        str, list[typing.Union[RandomJoint, RandomPose]]
    ] = dc.field(default_factory=dict)

    lights: typing.Mapping[str, LightParameter] = dc.field(default_factory=dict)

    # Cameras to add to the scene (and broadcast over LCM). The key for each
    # camera is a helpful mnemonic, but does not serve a technical role. The
    # CameraConfig::name field is still the name that will appear in the
    # Diagram artifacts.
    cameras: typing.Mapping[str, CameraConfig] = dc.field(default_factory=dict)
    environment_map: str = dc.field(
        default_factory=lambda: (
            "package://carbonsix-models/environment_maps/empty_play_room_4k.hdr"
        )
    )

    visualization: VisualizationConfig = VisualizationConfig()

    publish_lcm: bool = True
    # A map of {bus_name: lcm_params} for LCM transceivers to be used by
    # drivers, sensors, etc.
    lcm_buses: typing.Mapping[str, DrakeLcmParams] = dc.field(
        default_factory=lambda: dict(default=DrakeLcmParams())
    )

    # For actuated models, specifies where each model's actuation inputs come
    # from, keyed on the ModelInstance name.
    # More than one item is needed in typing.Union. Otherwise,
    # typing.Union[foo] is collapsed into foo.
    model_drivers: typing.Mapping[
        str,
        list[
            typing.Union[  # noqa: UP007
                JointPdControlledDriver, PosePdControlledDriver, HandControlledDriver
            ]
        ],
    ] = dc.field(default_factory=dict)


class Station:
    def __init__(
        self,
        scenario_file_name: str | Path,
        package_xmls: list[str | Path] | None = None,
        visualize: bool = True,
    ):
        self.scenario_file_name = scenario_file_name
        self.scenario = self.load_scenario(self.scenario_file_name)
        self.package_xmls = package_xmls if package_xmls is not None else []
        self.visualize = visualize

        # Random number generator
        self.rng = np.random.default_rng(self.scenario.random_seed)

        if self.visualize:
            self.meshcat = StartMeshcat()

        self.reset_simulation = False
        self.target_obj_pose = None

        self.station_diagram = None
        self.make_simulation()
        self.set_initial_conditions()

        # LCM Reset settings
        self.lcm = lcm.LCM()
        self.lcm.subscribe("RESET_SIMULATION", self.lcm_reset_callback)

        # Set target object pose for next episode
        self.lcm.subscribe("SET_OBJ_TARGET_POSE", self.lcm_set_obj_target_pose_callback)

    def lcm_reset_callback(self, channel, data):
        print("Received reset simulation command over LCM.")
        self.reset_simulation = True

    def lcm_set_obj_target_pose_callback(self, channel, data):
        print("Received target object pose for next episode.")

        msg = lcmt_robot_state.decode(data)
        self.target_obj_pose = msg.joint_position

        print(f"Target pose: {self.target_obj_pose}")

    def set_initial_conditions(self):
        for model_name, random_value in self.scenario.initial_conditions.items():
            random_value = random_value[0]  # Unpack the list
            if isinstance(random_value, RandomPose):
                model_instance = self.plant.GetModelInstanceByName(model_name)
                body = self.plant.GetBodyByName(random_value.body, model_instance)

                if self.target_obj_pose is not None:
                    target = np.asarray(self.target_obj_pose, dtype=float)
                    translation = target[0:3]
                    quat = target[3:7]
                    R_WF = RotationMatrix(Quaternion(wxyz=quat))
                    X_WF = RigidTransform(R_WF, translation)
                    self.target_obj_pose = None
                else:
                    translation = np.array(
                        [
                            self.rng.uniform(
                                random_value.min_translation[i],
                                random_value.max_translation[i],
                            )
                            for i in range(3)
                        ]
                    )
                    rotation = RollPitchYaw(
                        np.deg2rad(
                            [
                                self.rng.uniform(
                                    random_value.min_rotation_deg[i],
                                    random_value.max_rotation_deg[i],
                                )
                                for i in range(3)
                            ]
                        )
                    )
                    X_WF = RigidTransform(rotation, translation)
                self.plant.SetFreeBodyPose(self.plant_context, body, X_WF)
            elif isinstance(random_value, RandomJoint):
                model_idx = self.plant.GetModelInstanceByName(model_name)
                random_value = self.rng.uniform(
                    random_value.min_value, random_value.max_value
                )
                self.plant.SetPositions(self.plant_context, model_idx, random_value)

    def make_simulation(self):
        if self.scenario.publish_lcm:
            self.diagram = self.build_lcm_diagram()
            self.hardware_station = self.diagram.GetSubsystemByName("HardwareStation")
        else:
            self.diagram = self.make_hardware_station()
            self.hardware_station = self.diagram

        self.simulator = Simulator(self.diagram)
        ApplySimulatorConfig(self.scenario.simulator_config, self.simulator)
        self.simulator.Initialize()
        self.context = self.simulator.get_mutable_context()
        self.plant = self.hardware_station.GetSubsystemByName("plant")
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)

    def get_plant(self):
        return self.plant

    def get_mutable_context(self):
        return self.simulator.get_mutable_context()

    def get_plant_context(self):
        return self.plant_context

    def run(self):
        while True:
            self.lcm.handle_timeout(0)
            if not self.reset_simulation:
                time_now = self.simulator.get_context().get_time()
                self.simulator.AdvanceTo(time_now + 0.1)
            else:
                self.reset()
                self.set_initial_conditions()
                self.reset_simulation = False

    def load_scenario(
        self,
        filename: str | Path | None = None,
        data: str | None = None,
        scenario_name: str | None = None,
    ) -> Scenario:
        """
        Implement the command-line handling logic for scenario data.

        Return a Scenario object loaded from the given input arguments.

        Args:
        ----
            filename: A yaml filename to load the scenario from.

            data: A yaml string to load the scenario from. If both
                filename and string are specified, then the filename is parsed
                first, and then the string is _also_ parsed, potentially
                overwriting defaults from the filename.

            scenario_name: The name of the scenario/child to load from
                the yaml file. If None, then the entire file is loaded.

        """
        result = Scenario()
        if filename:
            result = yaml_load_typed(
                schema=Scenario,
                filename=filename,
                child_name=scenario_name,
                defaults=result,
            )
        if data:
            result = yaml_load_typed(schema=Scenario, data=data, defaults=result)
        return result

    def add_plant_from_scenario(
        self,
        builder: DiagramBuilder,
    ) -> tuple[MultibodyPlant, SceneGraph, list[ModelInstanceInfo]]:
        scenario = self.scenario
        package_xmls = self.package_xmls

        # Create the multibody plant and scene graph.
        sim_plant: MultibodyPlant
        scene_graph: SceneGraph
        sim_plant, scene_graph = AddMultibodyPlant(
            config=scenario.plant_config, builder=builder
        )

        self.parser = Parser(sim_plant)
        if package_xmls is not None:
            for p in package_xmls:
                self.parser.package_map().AddPackageXml(p)

        # Add model directives.
        added_models = ProcessModelDirectives(
            directives=ModelDirectives(directives=scenario.directives),
            parser=self.parser,
        )

        for (
            model_instance_name,
            driver_configs,
        ) in scenario.model_drivers.items():
            for driver_config in driver_configs:
                if isinstance(driver_config, JointPdControlledDriver):
                    robot_model = sim_plant.GetModelInstanceByName(model_instance_name)
                    joint_stiffness = driver_config.p_gains
                    joint_damping = driver_config.d_gains

                    actuator_indices = [
                        JointActuatorIndex(i)
                        for i in range(sim_plant.num_actuators())
                        if sim_plant.get_joint_actuator(
                            JointActuatorIndex(i)
                        ).model_instance()
                        == robot_model
                    ]

                    for actuator_index, Kp, Kd in zip(
                        actuator_indices, joint_stiffness, joint_damping
                    ):
                        sim_plant.get_joint_actuator(
                            actuator_index
                        ).set_controller_gains(PdControllerGains(p=Kp, d=Kd))

                    # Turn gravity off for arms and robots.
                    sim_plant.set_gravity_enabled(robot_model, False)
                if isinstance(driver_config, PosePdControlledDriver):
                    robot_model = sim_plant.GetModelInstanceByName(model_instance_name)
                    sim_plant.set_gravity_enabled(robot_model, False)
                if isinstance(driver_config, HandControlledDriver):
                    robot_model = sim_plant.GetModelInstanceByName(model_instance_name)

                    base = sim_plant.GetJointByName("JL1", robot_model)
                    mimic_joints = ["JL2", "JLF", "JR1", "JR2", "JRF"]
                    multiplier = [1.0, -1.0, -1.0, -1.0, -1.0]
                    for name, mult in zip(mimic_joints, multiplier):
                        j = sim_plant.GetJointByName(name, robot_model)
                        sim_plant.AddCouplerConstraint(
                            joint0=base, joint1=j, gear_ratio=mult, offset=0.0
                        )

                    # Turn gravity off for arms and robots.
                    sim_plant.set_gravity_enabled(robot_model, False)

        # Now the plant is complete. Add camera models to visualize configuration.
        for _, camera_config in self.scenario.cameras.items():
            camera_model_idx = self.parser.AddModelsFromUrl(
                "package://carbonsix-models/objects/camera_box/descriptor/camera_box.sdf"
            )[0]
            sim_plant.RenameModelInstance(camera_model_idx, camera_config.name)
            if camera_config.X_PB.base_frame is not None:
                model_instance_name, frame_name = camera_config.X_PB.base_frame.split(
                    "::"
                )
                parent_frame = sim_plant.GetFrameByName(
                    frame_name,
                    sim_plant.GetModelInstanceByName(model_instance_name),
                )

            else:
                parent_frame = sim_plant.world_frame()

            sim_plant.WeldFrames(
                frame_on_parent_F=parent_frame,
                frame_on_child_M=sim_plant.GetFrameByName("center", camera_model_idx),
                X_FM=RigidTransform(
                    RollPitchYaw(np.deg2rad(camera_config.X_PB.rotation.value.deg)),
                    camera_config.X_PB.translation,
                ),
            )

        sim_plant.Finalize()

        return sim_plant, scene_graph, added_models

    def make_hardware_station(self) -> Diagram:
        builder = DiagramBuilder()

        sim_plant, scene_graph, added_models = self.add_plant_from_scenario(
            builder=builder
        )

        # Add visualization.
        if self.visualize:
            ApplyVisualizationConfig(
                self.scenario.visualization, builder, meshcat=self.meshcat
            )
            self.meshcat.SetEnvironmentMap(
                self.parser.package_map().ResolveUrl(f"{self.scenario.environment_map}")
            )

        # Define lights
        lights_list = []
        lights_cfgs = self.scenario.lights
        if len(lights_cfgs) > 0:
            for light_cfg in lights_cfgs.values():
                lights = LightParameter()
                lights.type = light_cfg.type
                lights.position = np.array(light_cfg.position)
                lights.direction = np.array(light_cfg.direction)
                lights.cone_angle = light_cfg.cone_angle
                lights.frame = light_cfg.frame
                lights.color = light_cfg.color
                lights.intensity = light_cfg.intensity
                lights_list.append(lights)
        else:
            lights = LightParameter()
            lights.type = "directional"
            lights.position = np.array([0, 0, 1])
            lights.direction = np.array([0, 0, -1])
            lights.cone_angle = 45
            lights.frame = "world"
            lights.intensity = 1.0
            lights_list.append(lights)

        # Add renderer
        renderer_name = "renderer"
        rendering_params = RenderEngineVtkParams()
        rendering_params.environment_map = EnvironmentMap(
            texture=EquirectangularMap(
                path=self.parser.package_map().ResolveUrl(
                    f"{self.scenario.environment_map}"
                )
            )
        )
        rendering_params.lights = lights_list
        rendering_params.exposure = 2.0
        rendering_params.backend = ""

        scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(rendering_params))

        # Add scene cameras.
        for i, camera_config in enumerate(self.scenario.cameras.values()):
            # if i == 0:
            #    camera_config.renderer_class = rendering_params

            ApplyCameraConfig(config=camera_config, builder=builder)
            camera_sensor = builder.GetSubsystemByName(
                f"rgbd_sensor_{camera_config.name}"
            )
            for port_name in ["color_image", "depth_image_32f", "label_image"]:
                builder.ExportOutput(
                    camera_sensor.GetOutputPort(port_name),
                    f"{port_name}_{camera_config.name}",
                )

            AddFrameTriadIllustration(
                scene_graph=scene_graph,
                frame=sim_plant.GetFrameByName(
                    "center",
                    sim_plant.GetModelInstanceByName(camera_config.name),
                ),
                length=0.15,
                radius=0.003,
            )

        # Export "cheat" ports.
        builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
        builder.ExportOutput(
            sim_plant.get_contact_results_output_port(), "contact_results"
        )
        builder.ExportOutput(
            sim_plant.get_state_output_port(), "plant_continuous_state"
        )
        builder.ExportOutput(sim_plant.get_body_poses_output_port(), "body_poses")

        # Export required status and command ports for LCM communication.
        for (
            model_instance_name,
            model_driver_config,
        ) in self.scenario.model_drivers.items():
            model_driver_config = model_driver_config[0]
            if isinstance(model_driver_config, JointPdControlledDriver):
                self.export_joint_pd_driver_ports(
                    builder, sim_plant, model_instance_name
                )
            elif isinstance(model_driver_config, PosePdControlledDriver):
                self.export_pose_pd_driver_ports(
                    builder, sim_plant, model_instance_name
                )
            elif isinstance(model_driver_config, HandControlledDriver):
                self.export_hand_driver_ports(builder, sim_plant, model_instance_name)
            else:
                raise ValueError(
                    "Only JointPdControlledDriver, PosePdControlledDriver, and HandControlledDriver are supported."
                )

        diagram = builder.Build()
        diagram.set_name("HardwareStation")
        return diagram

    def export_joint_pd_driver_ports(self, builder, sim_plant, model_instance_name):
        """
        Export diagram ports for connecting to LCM publisher and subscriber systems.
        Required ports for command receiver:
        - position
        - feedforward torque.

        Required ports for status sender:
         - position_measured
         - velocity_estimated
         - position_commanded
         - torque_commanded
        """
        model_idx = sim_plant.GetModelInstanceByName(model_instance_name)
        n_dofs = sim_plant.num_actuated_dofs(model_idx)

        # LCM command receiver ports.
        # Add state interpolator from position to desired state.
        desired_state_from_position = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                n_dofs, sim_plant.time_step(), suppress_initial_transient=True
            )
        )
        desired_state_from_position.set_name(
            f"{model_instance_name}.desired_state_from_position"
        )
        builder.Connect(
            desired_state_from_position.get_output_port(),
            sim_plant.get_desired_state_input_port(model_idx),
        )
        position_passthrough = builder.AddSystem(PassThrough([0] * n_dofs))
        builder.Connect(
            position_passthrough.get_output_port(),
            desired_state_from_position.get_input_port(),
        )
        builder.ExportInput(
            position_passthrough.get_input_port(),
            f"{model_instance_name}.position",
        )

        # Add torque passthrough.
        torque_passthrough = builder.AddSystem(PassThrough([0] * n_dofs))
        gravity_compensator = builder.AddSystem(
            GravityCompensator(sim_plant, model_instance_name)
        )
        adder = builder.AddSystem(Adder(2, n_dofs))
        builder.Connect(
            torque_passthrough.get_output_port(),
            adder.get_input_port(0),
        )
        builder.Connect(
            gravity_compensator.get_output_port(),
            adder.get_input_port(1),
        )
        builder.Connect(
            sim_plant.get_state_output_port(),
            gravity_compensator.get_input_port(),
        )
        builder.Connect(
            adder.get_output_port(),
            sim_plant.get_actuation_input_port(model_idx),
        )
        builder.ExportInput(
            torque_passthrough.get_input_port(),
            f"{model_instance_name}.feedforward_torque",
        )
        builder.ExportOutput(
            sim_plant.get_net_actuation_output_port(model_idx),
            f"{model_instance_name}.torque_measured",
        )

        # LCM status sender ports.
        # Add demux to export position and velocity.
        demux = builder.AddSystem(Demultiplexer(2 * n_dofs, n_dofs))
        builder.Connect(
            sim_plant.get_state_output_port(model_idx),
            demux.get_input_port(),
        )
        builder.ExportOutput(
            demux.get_output_port(0),
            f"{model_instance_name}.position_measured",
        )
        builder.ExportOutput(
            demux.get_output_port(1),
            f"{model_instance_name}.velocity_estimated",
        )

        # Passthrough position input to position_commanded.
        builder.ExportOutput(
            position_passthrough.get_output_port(),
            f"{model_instance_name}.position_commanded",
        )

        # Passthrough torque input to torque_commanded.
        builder.ExportOutput(
            torque_passthrough.get_output_port(),
            f"{model_instance_name}.torque_commanded",
        )

        # Export external torque input.
        builder.ExportOutput(
            sim_plant.get_generalized_contact_forces_output_port(model_idx),
            f"{model_instance_name}.torque_external",
        )

    def export_hand_driver_ports(self, builder, sim_plant, model_instance_name):
        """
        Export Sigma hand LCM command/status ports and connect the external PD controller.

        Command ports:
        - position_mm
        - force  (feedforward torque)

        Status ports:
        - actual_position_mm
        - actual_speed_mm_per_s
        - actual_force
        """

        model_idx = sim_plant.GetModelInstanceByName(model_instance_name)
        driver_cfg = self.scenario.model_drivers[model_instance_name][0]
        kp = driver_cfg.p_gains[0]
        kd = driver_cfg.d_gains[0]
        discrete_cmd = driver_cfg.discrete_cmd
        threshold = driver_cfg.threshold

        # LCM command receiver ports.
        # position command
        position_cmd = builder.AddSystem(PassThrough([0.0]))
        builder.ExportInput(
            position_cmd.get_input_port(),
            f"{model_instance_name}.position_mm",
        )

        # feedforward torque
        force_cmd = builder.AddSystem(PassThrough([0.0]))
        builder.ExportInput(
            force_cmd.get_input_port(),
            f"{model_instance_name}.force",
        )

        # Sigma hand controller
        controller = builder.AddSystem(
            SigmaHandController(
                sim_plant, model_instance_name, kp, kd, discrete_cmd, threshold
            )
        )

        # Connect command, state, feedforward to controller
        builder.Connect(
            position_cmd.get_output_port(),
            controller.position_cmd_input_port,
        )
        builder.Connect(
            sim_plant.get_state_output_port(model_idx),
            controller.state_input_port,
        )
        builder.Connect(
            force_cmd.get_output_port(),
            controller.feedforward_input_port,
        )

        builder.Connect(
            controller.tau_output_port,
            sim_plant.get_actuation_input_port(model_idx),
        )

        # LCM status sender ports.
        builder.ExportOutput(
            controller.pos_mm_output_port,
            f"{model_instance_name}.actual_position",
        )
        builder.ExportOutput(
            controller.vel_mm_output_port,
            f"{model_instance_name}.actual_speed",
        )
        force_sum = builder.AddSystem(MatrixGain(np.ones((1, 6))))
        builder.Connect(
            sim_plant.get_generalized_contact_forces_output_port(model_idx),
            force_sum.get_input_port(),
        )

        builder.ExportOutput(
            force_sum.get_output_port(),
            f"{model_instance_name}.actual_torque",
        )

        builder.ExportOutput(
            position_cmd.get_output_port(),
            f"{model_instance_name}.target_position",
        )

    def export_pose_pd_driver_ports(self, builder, sim_plant, model_instance_name):
        """
        Export diagram ports for connecting to LCM publisher and subscriber systems.
        Required ports for command receiver:
        - position
        - feedforward torque.

        Required ports for status sender:
         - position_measured
         - velocity_estimated
         - position_commanded
         - torque_commanded
        """
        model_idx = sim_plant.GetModelInstanceByName(model_instance_name)
        assert sim_plant.num_positions(model_idx) == 7
        assert sim_plant.num_velocities(model_idx) == 6

        gravity_compensator = builder.AddSystem(
            GravityCompensator(sim_plant, model_instance_name)
        )
        pose_controller = builder.AddSystem(
            PoseController(
                sim_plant,
                model_instance_name=model_instance_name,
                p_gains_rotation=self.scenario.model_drivers[model_instance_name][
                    0
                ].p_gains_rotation,
                d_gains_rotation=self.scenario.model_drivers[model_instance_name][
                    0
                ].d_gains_rotation,
                p_gains_translation=self.scenario.model_drivers[model_instance_name][
                    0
                ].p_gains_translation,
                d_gains_translation=self.scenario.model_drivers[model_instance_name][
                    0
                ].d_gains_translation,
            )
        )
        pose_controller.set_name(f"{model_instance_name}.pose_controller")

        builder.Connect(
            sim_plant.get_state_output_port(),
            gravity_compensator.get_input_port(),
        )
        builder.Connect(
            gravity_compensator.get_output_port(),
            pose_controller.GetInputPort("feedforward_spatial_force"),
        )
        builder.Connect(
            sim_plant.GetOutputPort(f"{model_instance_name}_state"),
            pose_controller.GetInputPort("current_state"),
        )
        builder.ExportInput(
            pose_controller.GetInputPort("desired_pose"),
            f"{model_instance_name}.position",
        )
        builder.Connect(
            pose_controller.GetOutputPort("applied_generalized_torque"),
            sim_plant.get_applied_generalized_force_input_port(),
        )

        demux = builder.AddSystem(Demultiplexer([7, 6]))
        builder.Connect(
            sim_plant.GetOutputPort(f"{model_instance_name}_state"),
            demux.get_input_port(),
        )
        builder.ExportOutput(
            demux.get_output_port(0), f"{model_instance_name}.position"
        )
        builder.ExportOutput(
            demux.get_output_port(1), f"{model_instance_name}.velocity"
        )

    def build_lcm_diagram(self):
        builder = DiagramBuilder()
        scenario = self.scenario

        # build
        station = builder.AddSystem(self.make_hardware_station())
        sim_plant = station.GetSubsystemByName("plant")

        # Apply LCM bus configuration.
        lcm_buses = ApplyLcmBusConfig(
            lcm_buses=self.scenario.lcm_buses,
            builder=builder,  # type: ignore
        )

        for (
            model_instance_name,
            driver_configs,
        ) in scenario.model_drivers.items():
            for driver_config in driver_configs:
                if isinstance(driver_config, JointPdControlledDriver):
                    self.apply_joint_pd_controlled_driver_config_mock(
                        builder=builder,
                        driver_config=driver_config,
                        lcm_buses=lcm_buses,
                        model_instance_name=model_instance_name,
                        sim_plant=sim_plant,
                        station=station,
                        status_channel_name=driver_config.lcm_publish_channel,
                        command_channel_name=driver_config.lcm_subscribe_channel,
                    )
                elif isinstance(driver_config, PosePdControlledDriver):
                    self.apply_pose_pd_controlled_driver_config_mock(
                        builder=builder,
                        driver_config=driver_config,
                        lcm_buses=lcm_buses,
                        model_instance_name=model_instance_name,
                        sim_plant=sim_plant,
                        station=station,
                        status_channel_name=driver_config.lcm_publish_channel,
                        command_channel_name=driver_config.lcm_subscribe_channel,
                    )
                elif isinstance(driver_config, HandControlledDriver):
                    self.apply_hand_controlled_driver_config_mock(
                        builder=builder,
                        driver_config=driver_config,
                        lcm_buses=lcm_buses,
                        model_instance_name=model_instance_name,
                        sim_plant=sim_plant,
                        station=station,
                        status_channel_name=driver_config.lcm_publish_channel,
                        command_channel_name=driver_config.lcm_subscribe_channel,
                    )
                else:
                    raise ValueError(
                        f"{type(driver_config)} not yet supported in MockStation"
                    )

        diagram = builder.Build()
        diagram.set_name("LcmHardwareStation")
        return diagram

    def apply_joint_pd_controlled_driver_config_mock(
        self,
        builder: DiagramBuilder,
        driver_config: JointPdControlledDriver,
        lcm_buses: LcmBuses,
        model_instance_name: str,
        sim_plant: MultibodyPlant,
        station: Diagram,
        status_channel_name: str = "STATUS",
        command_channel_name: str = "COMMAND",
    ) -> None:
        n_joints = sim_plant.num_actuated_dofs(
            sim_plant.GetModelInstanceByName(model_instance_name)
        )
        status_sender = command_receiver = None
        status_sender = RobotStatusSender(n_joints=n_joints)
        command_receiver = RobotCommandReceiver(
            n_joints=n_joints,
            default_joint_positions=sim_plant.GetDefaultPositions(
                sim_plant.GetModelInstanceByName(model_instance_name)
            ),
        )

        return self.apply_joint_pd_controlled_driver_config_mock_impl(
            builder=builder,
            driver_config=driver_config,
            lcm_buses=lcm_buses,
            model_instance_name=model_instance_name,
            sim_plant=sim_plant,
            station=station,
            status_sender=status_sender,
            status_channel_name=status_channel_name,
            command_receiver=command_receiver,
            command_channel_name=command_channel_name,
        )

    def apply_hand_controlled_driver_config_mock(
        self,
        builder: DiagramBuilder,
        driver_config: HandControlledDriver,
        lcm_buses: LcmBuses,
        model_instance_name: str,
        sim_plant: MultibodyPlant,
        station: Diagram,
        status_channel_name: str = "STATUS",
        command_channel_name: str = "COMMAND",
    ) -> None:
        status_sender = command_receiver = None
        status_sender = HandStatusSender()
        command_receiver = HandCommandReceiver(
            default_position_mm=sim_plant.GetDefaultPositions(
                sim_plant.GetModelInstanceByName(model_instance_name)
            )[0],
            default_force=0.0,
        )

        return self.apply_hand_controlled_driver_config_mock_impl(
            builder=builder,
            driver_config=driver_config,
            lcm_buses=lcm_buses,
            model_instance_name=model_instance_name,
            sim_plant=sim_plant,
            station=station,
            status_sender=status_sender,
            status_channel_name=status_channel_name,
            command_receiver=command_receiver,
            command_channel_name=command_channel_name,
        )

    def apply_pose_pd_controlled_driver_config_mock(
        self,
        builder: DiagramBuilder,
        driver_config: JointPdControlledDriver,
        lcm_buses: LcmBuses,
        model_instance_name: str,
        sim_plant: MultibodyPlant,
        station: Diagram,
        status_channel_name: str = "STATUS",
        command_channel_name: str = "COMMAND",
    ) -> None:
        status_sender = command_receiver = None
        status_sender = PoseStatusSender()
        command_receiver = PoseCommandReceiver(
            default_pose=sim_plant.GetDefaultPositions(
                sim_plant.GetModelInstanceByName(model_instance_name)
            ),
        )

        return self.apply_pose_pd_controlled_driver_config_mock_impl(
            builder=builder,
            driver_config=driver_config,
            lcm_buses=lcm_buses,
            model_instance_name=model_instance_name,
            sim_plant=sim_plant,
            station=station,
            status_sender=status_sender,
            status_channel_name=status_channel_name,
            command_receiver=command_receiver,
            command_channel_name=command_channel_name,
        )

    def apply_hand_controlled_driver_config_mock_impl(
        self,
        builder: DiagramBuilder,
        driver_config: HandControlledDriver,
        lcm_buses: LcmBuses,
        model_instance_name: str,
        sim_plant: MultibodyPlant,
        station: Diagram,
        status_sender: HandStatusSender,
        status_channel_name: str,
        command_receiver: HandCommandReceiver,
        command_channel_name: str,
    ) -> None:
        lcm_subscribe = lcm_buses.Find(
            "Driver for " + model_instance_name,
            driver_config.lcm_subscribe_bus,
        )
        lcm_publish = lcm_buses.Find(
            "Driver for " + model_instance_name, driver_config.lcm_publish_bus
        )

        self.setup_hand_status_sender(
            builder,
            station,
            model_instance_name,
            status_sender,
            lcm_publish,
            status_channel_name,
            driver_config,
        )

        self.setup_hand_command_receiver(
            builder,
            station,
            sim_plant,
            model_instance_name,
            command_receiver,
            lcm_subscribe,
            command_channel_name,
        )

    def apply_joint_pd_controlled_driver_config_mock_impl(
        self,
        builder: DiagramBuilder,
        driver_config: JointPdControlledDriver,
        lcm_buses: LcmBuses,
        model_instance_name: str,
        sim_plant: MultibodyPlant,
        station: Diagram,
        status_sender: RobotStatusSender,
        status_channel_name: str,
        command_receiver: RobotCommandReceiver,
        command_channel_name: str,
    ) -> None:
        lcm_subscribe = lcm_buses.Find(
            "Driver for " + model_instance_name,
            driver_config.lcm_subscribe_bus,
        )
        lcm_publish = lcm_buses.Find(
            "Driver for " + model_instance_name, driver_config.lcm_publish_bus
        )

        self.setup_joint_status_sender(
            builder,
            station,
            model_instance_name,
            status_sender,
            lcm_publish,
            status_channel_name,
            driver_config,
        )

        self.setup_joint_command_receiver(
            builder,
            station,
            sim_plant,
            model_instance_name,
            command_receiver,
            lcm_subscribe,
            command_channel_name,
        )

    def apply_pose_pd_controlled_driver_config_mock_impl(
        self,
        builder: DiagramBuilder,
        driver_config: JointPdControlledDriver,
        lcm_buses: LcmBuses,
        model_instance_name: str,
        sim_plant: MultibodyPlant,
        station: Diagram,
        status_sender: PoseStatusSender,
        status_channel_name: str,
        command_receiver: RobotCommandReceiver,
        command_channel_name: str,
    ) -> None:
        lcm_subscribe = lcm_buses.Find(
            "Driver for " + model_instance_name,
            driver_config.lcm_subscribe_bus,
        )
        lcm_publish = lcm_buses.Find(
            "Driver for " + model_instance_name, driver_config.lcm_publish_bus
        )

        self.setup_pose_status_sender(
            builder,
            station,
            model_instance_name,
            status_sender,
            lcm_publish,
            status_channel_name,
            driver_config,
        )

        self.setup_pose_command_receiver(
            builder,
            station,
            sim_plant,
            model_instance_name,
            command_receiver,
            lcm_subscribe,
            command_channel_name,
        )

    def setup_joint_status_sender(
        self,
        builder: DiagramBuilder,
        station: Diagram,
        model_instance_name: str,
        status_sender: typing.Any,
        lcm_publish: LcmPublisherSystem,
        channel: str,
        driver_config: JointPdControlledDriver,
    ) -> None:
        builder.AddNamedSystem(f"{model_instance_name}.status_sender", status_sender)
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.position_measured"),
            status_sender.joint_position_input_port,
        )
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.velocity_estimated"),
            status_sender.joint_velocity_input_port,
        )
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.position_commanded"),
            status_sender.joint_position_commanded_input_port,
        )
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.torque_commanded"),
            status_sender.joint_torque_commanded_input_port,
        )
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.torque_measured"),
            status_sender.joint_torque_measured_input_port,
        )
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.torque_external"),
            status_sender.joint_torque_external_input_port,
        )

        status_publisher = LcmPublisherSystem.Make(
            channel=channel,
            lcm_type=lcmt_iiwa_status,
            lcm=lcm_publish,
            publish_period=1.0 / driver_config.lcm_publish_hz,
            use_cpp_serializer=False,
        )
        status_publisher.set_name(f"{model_instance_name}.status_publisher")
        builder.AddSystem(status_publisher)
        builder.Connect(
            status_sender.get_output_port(), status_publisher.get_input_port()
        )

    def setup_hand_status_sender(
        self,
        builder: DiagramBuilder,
        station: Diagram,
        model_instance_name: str,
        status_sender: typing.Any,
        lcm_publish: LcmPublisherSystem,
        channel: str,
        driver_config: HandControlledDriver,
    ) -> None:
        builder.AddNamedSystem(f"{model_instance_name}.status_sender", status_sender)
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.actual_position"),
            status_sender.position_input_port,
        )
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.actual_speed"),
            status_sender.speed_input_port,
        )
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.actual_torque"),
            status_sender.force_input_port,
        )

        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.target_position"),
            status_sender.target_position_input_port,
        )

        status_publisher = LcmPublisherSystem.Make(
            channel=channel,
            lcm_type=lcmt_drake_signal,
            lcm=lcm_publish,
            publish_period=1.0 / driver_config.lcm_publish_hz,
            use_cpp_serializer=False,
        )
        status_publisher.set_name(f"{model_instance_name}.status_publisher")
        builder.AddSystem(status_publisher)
        builder.Connect(
            status_sender.get_output_port(), status_publisher.get_input_port()
        )

    def setup_pose_status_sender(
        self,
        builder: DiagramBuilder,
        station: Diagram,
        model_instance_name: str,
        status_sender: PoseStatusSender,
        lcm_publish: LcmPublisherSystem,
        channel: str,
        driver_config: PosePdControlledDriver,
    ) -> None:
        builder.AddNamedSystem(f"{model_instance_name}.status_sender", status_sender)
        builder.Connect(
            station.GetOutputPort(f"{model_instance_name}.position"),
            status_sender.position_input_port,
        )
        status_publisher = LcmPublisherSystem.Make(
            channel=channel,
            lcm_type=lcmt_robot_state,
            lcm=lcm_publish,
            publish_period=1.0 / driver_config.lcm_publish_hz,
            use_cpp_serializer=False,
        )
        status_publisher.set_name(f"{model_instance_name}.status_publisher")
        builder.AddSystem(status_publisher)
        builder.Connect(
            status_sender.get_output_port(), status_publisher.get_input_port()
        )

    def setup_joint_command_receiver(
        self,
        builder: DiagramBuilder,
        station: Diagram,
        sim_plant: MultibodyPlant,
        model_instance_name: str,
        command_receiver: typing.Any,
        lcm_subscribe: LcmSubscriberSystem,
        channel: str,
    ) -> None:
        command_receiver.set_name(f"{model_instance_name}.command_receiver")
        builder.AddSystem(command_receiver)
        builder.Connect(
            command_receiver.joint_position_output_port,
            station.GetInputPort(f"{model_instance_name}.position"),
        )
        builder.Connect(
            command_receiver.joint_torque_output_port,
            station.GetInputPort(f"{model_instance_name}.feedforward_torque"),
        )

        command_subscriber = builder.AddNamedSystem(
            f"{model_instance_name}.command_subscriber",
            LcmSubscriberSystem.Make(
                channel=channel,
                lcm_type=lcmt_iiwa_command,
                lcm=lcm_subscribe,
                use_cpp_serializer=False,
            ),
        )
        builder.Connect(
            command_subscriber.get_output_port(),
            command_receiver.message_input_port,
        )

    def setup_hand_command_receiver(
        self,
        builder: DiagramBuilder,
        station: Diagram,
        sim_plant: MultibodyPlant,
        model_instance_name: str,
        command_receiver: typing.Any,
        lcm_subscribe: LcmSubscriberSystem,
        channel: str,
    ) -> None:
        command_receiver.set_name(f"{model_instance_name}.command_receiver")
        builder.AddSystem(command_receiver)
        builder.Connect(
            command_receiver.position_output_port,
            station.GetInputPort(f"{model_instance_name}.position_mm"),
        )
        builder.Connect(
            command_receiver.force_output_port,
            station.GetInputPort(f"{model_instance_name}.force"),
        )
        command_subscriber = builder.AddNamedSystem(
            f"{model_instance_name}.command_subscriber",
            LcmSubscriberSystem.Make(
                channel=channel,
                lcm_type=lcmt_schunk_wsg_command,
                lcm=lcm_subscribe,
                use_cpp_serializer=False,
            ),
        )
        builder.Connect(
            command_subscriber.get_output_port(),
            command_receiver.message_input_port,
        )

    def setup_pose_command_receiver(
        self,
        builder: DiagramBuilder,
        station: Diagram,
        sim_plant: MultibodyPlant,
        model_instance_name: str,
        command_receiver: PoseCommandReceiver,
        lcm_subscribe: LcmSubscriberSystem,
        channel: str,
    ) -> None:
        command_receiver.set_name(f"{model_instance_name}.command_receiver")
        builder.AddSystem(command_receiver)
        builder.Connect(
            command_receiver.position_output_port,
            station.GetInputPort(f"{model_instance_name}.position"),
        )

        command_subscriber = builder.AddNamedSystem(
            f"{model_instance_name}.command_subscriber",
            LcmSubscriberSystem.Make(
                channel=channel,
                lcm_type=lcmt_robot_state,
                lcm=lcm_subscribe,
                use_cpp_serializer=False,
            ),
        )
        builder.Connect(
            command_subscriber.get_output_port(),
            command_receiver.message_input_port,
        )

    def reset(self):
        """
        Reset the simulation to the initial state.
        """
        self.simulator.reset_context(self.context)
        self.simulator.Initialize()
        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.diagram.ForcedPublish(self.context)

    def step(self, dt: float = 0.02):
        """
        Step the simulation by dt seconds.
        """
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + dt)

    def get_positions(self, model_instance_names: list[str]) -> dict[str, np.ndarray]:
        """
        Get the positions of all models in the plant.
        Returns a dictionary mapping model instance names to their positions.
        """
        positions = {}
        for model_instance_name in model_instance_names:
            model_idx = self.plant.GetModelInstanceByName(model_instance_name)
            positions[model_instance_name] = self.plant.GetPositions(
                self.plant_context, model_idx
            )
        return positions

    def set_positions(self, positions: dict[str, np.ndarray]):
        """
        Set the positions of all models in the plant.
        Expects a dictionary mapping model instance names to their positions.
        """
        for model_instance, pos in positions.items():
            model_idx = self.plant.GetModelInstanceByName(model_instance)
            self.plant.SetPositions(self.plant_context, model_idx, pos)
        self.diagram.ForcedPublish(self.context)

    def set_command(
        self,
        position_commands: dict[str, dict[str, np.ndarray | RigidTransform]],
    ):
        for model_name, value in position_commands.items():
            input_port = self.hardware_station.GetInputPort(f"{model_name}.position")
            input_port.FixValue(self.context, value)

    def get_images(self):
        images = {}
        for key, item in self.scenario.cameras.items():
            images[item.name] = (
                self.hardware_station.GetOutputPort(f"color_image_{item.name}")
                .Eval(self.context)
                .data
            )
            images[item.name] = cv2.cvtColor(images[item.name], cv2.COLOR_RGB2BGR)
        return images
