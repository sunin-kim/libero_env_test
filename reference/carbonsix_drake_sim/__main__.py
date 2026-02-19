import argparse
import sys
from pathlib import Path
from carbonsix_drake_sim.scripts import (
    directive_visualizer,
    model_visualizer,
    simulation,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run CarbonSix Drake Simulation utilities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    $ carbonsix-drake-sim --help
    $ carbonsix-drake-sim model_visualizer --descriptor <descriptor_path>
    $ carbonsix-drake-sim directive_visualizer --directive <directive_path>
    $ carbonsix-drake-sim simulation --scenario <scenario_file_path>
        """,
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    model_vis_parser = subparsers.add_parser(
        "model_visualizer",
        help="Visualize a model in Meshcat.",
    )
    model_vis_parser.add_argument(
        "--descriptor",
        type=Path,
        required=True,
        help="Path to the model descriptor file (e.g., URDF, SDF).",
    )

    directive_vis_parser = subparsers.add_parser(
        "directive_visualizer",
        help="Visualize a directive in Meshcat.",
    )
    directive_vis_parser.add_argument(
        "--directive",
        type=Path,
        required=True,
        help="Path to the directive file (.yaml)",
    )

    simulation_parser = subparsers.add_parser(
        "simulation",
        help="Run simulation provided a scenario file.",
    )
    simulation_parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to the scenario file (.yaml)",
    )

    args = parser.parse_args()

    match args.subcommand:
        case "model_visualizer":
            model_visualizer.run_model_visualizer(descriptor=args.descriptor)
        case "directive_visualizer":
            directive_visualizer.run_directive_visualizer(directive=args.directive)
        case "simulation":
            simulation.run_simulation(scenario=args.scenario)


if __name__ == "__main__":
    sys.exit(main())
