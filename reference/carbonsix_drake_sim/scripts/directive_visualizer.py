from pathlib import Path
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    MeshcatVisualizer,
    Parser,
    ProcessModelDirectives,
    Simulator,
    StartMeshcat,
)


def run_directive_visualizer(directive: Path):
    # Boilerplate for setting up meshcat, plant, scene_graph.
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    parser = Parser(plant, scene_graph)
    package_map = parser.package_map()
    package_map.AddPackageXml("carbonsix-models/package.xml")

    # Load model directives and finalize.
    ProcessModelDirectives(
        LoadModelDirectives(str(directive)),
        plant,
        parser,
    )
    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)

    meshcat.StartRecording()
    simulator.AdvanceTo(1.0)
    meshcat.PublishRecording()
    input()
