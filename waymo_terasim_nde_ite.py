import faulthandler
from pathlib import Path
from typing import List, Optional
import typer
from loguru import logger
from terasim.simulator import Simulator
from terasim.logger.infoextractor import InfoExtractor
from terasim_nde_nade.envs.safetest_nade import SafeTestNADE
from waymo_terasim_vehicle_factory import WaymoTeraSimVehicleFactory
from sumo_baseline.model.waymo_terasim_plugin import WaymoTeraSimPlugin
from scenparse.utils import read_data


# Enable faulthandler
# faulthandler.enable()

app = typer.Typer()


@app.command()
def run_simulation(
    tfrecord_path: Path = typer.Argument(
        "/home/haoweis/waymo_parse/TeraSim-Data-Zoo/uncompressed_scenario_validation_validation.tfrecord-00000-of-00150",
        help="Path to Waymo tfrecord file",
        exists=True,
    ),
    map_base_dir: Path = typer.Argument(
        "/home/haoweis/waymo_parse/TeraSim-Data-Zoo/womd_map",
        help="Base directory containing SUMO map files",
        exists=True,
    ),
    scenario_id: Optional[str] = typer.Option(
        "19a486cd29abd7a7",
        "--scenario-id",
        help="Specific scenario ID to simulate. If not provided, will process all scenarios.",
    ),
    experiment_name: str = typer.Option(
        "test_experiment",
        "--name",
        help="Experiment name",
    ),
    output_dir: Path = typer.Option(
        "output",
        "--dir",
        "-d",
        help="Output directory",
    ),
    epoch: str = typer.Option(
        "0_0",
        "--nth",
        help="The nth epoch",
    ),
    aggregated_dir: Path = typer.Option(
        "aggregated",
        "--aggregated-dir",
        "-a",
        help="Aggregated directory for logs",
    ),
    gui: bool = typer.Option(
        True,
        "--gui/--no-gui",
        help="Enable/disable SUMO GUI",
    ),
):
    """
    Run TeraSimulation using Waymo scenario data and converted SUMO maps.
    """
    # Create output directories
    base_dir = output_dir / experiment_name / "raw_data" / epoch
    base_dir.mkdir(parents=True, exist_ok=True)

    aggregated_dir.mkdir(parents=True, exist_ok=True)
    aggregated_log_dir = aggregated_dir / "loguru_run.log"

    # Setup logging
    log_files = [base_dir / "loguru_run.log", aggregated_log_dir]
    log_levels = ["TRACE", "INFO"]

    for log_file, log_level in zip(log_files, log_levels):
        logger.add(
            log_file,
            level=log_level,
            enqueue=True,
            backtrace=True,
            serialize=True,
        )

    sumo_cfg_path = map_base_dir / f"{scenario_id}/{scenario_id}.sumocfg"
    sumo_net_path = map_base_dir / f"{scenario_id}/{scenario_id}.net.xml"

    if not sumo_cfg_path.exists() or not sumo_net_path.exists():
        logger.error(f"Required SUMO files not found for scenario {scenario_id}")
        raise typer.Exit(1)

    logger.info(f"Processing scenario: {scenario_id}")

    # Create SafeTestNADE environment with NDEVehicleFactory
    env = SafeTestNADE(
        vehicle_factory=WaymoTeraSimVehicleFactory(),
        info_extractor=InfoExtractor,
        log_flag=True,
        log_dir=output_dir / experiment_name / "raw_data" / epoch,
        warmup_time_lb=0,
        warmup_time_ub=1,
        run_time=80,
    )

    sim = Simulator(
        sumo_net_file_path=str(sumo_net_path),
        sumo_config_file_path=str(sumo_cfg_path),
        num_tries=10,
        gui_flag=gui,
        output_path=map_base_dir,
        sumo_output_file_types=["fcd_all", "collision", "tripinfo"],
        additional_sumo_args=["--lateral-resolution", "0.2"],
    )
    sim.bind_env(env)

    scenario_ids = [scenario_id] if scenario_id else None

    # Load scenarios
    scenario_list = read_data(
        str(tfrecord_path),
        scenario_ids,  # If None, read_data should process all scenarios
    )

    # Note: You'll need to handle scenario data loading here
    # This might require adjusting the WaymoTeraSimPlugin initialization
    waymo_terasim_plugin = WaymoTeraSimPlugin(
        scenario=scenario_list[0],
        sumo_cfg_path=str(sumo_cfg_path),
        sumo_net_path=str(sumo_net_path),
    )
    waymo_terasim_plugin.inject(sim, {})

    sim.run()


if __name__ == "__main__":
    app()
