from pathlib import Path
from pydrake.all import ModelVisualizer, StartMeshcat


def run_model_visualizer(descriptor: Path):
    meshcat = StartMeshcat()
    visualizer = ModelVisualizer(
        meshcat=meshcat,
        visualize_frames=True,
        triad_length=0.05,
        triad_radius=0.001,
    )
    package_map = visualizer.parser().package_map()
    package_map.AddPackageXml("carbonsix-models/package.xml")
    """
    Some examples
       - package://carbonsix-models/robots/m0609/descriptor/m0609.urdf
       - package://carbonsix-models/objects/004_sugar_box/descriptor/004_sugar_box.urdf
       - package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf
    """
    visualizer.AddModels(url=f"package://{descriptor}")
    visualizer.Run()
    input()
