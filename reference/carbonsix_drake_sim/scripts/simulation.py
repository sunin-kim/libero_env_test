from pathlib import Path
from carbonsix_drake_sim.hardware_station.station import Station


def run_simulation(scenario: Path):
    station = Station(
        scenario_file_name=scenario,
        package_xmls=["carbonsix-models/package.xml"],
        visualize=True,
    )
    station.run()
