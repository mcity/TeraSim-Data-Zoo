import os
from pathlib import Path
import typer
from typing import Optional
from scenparse import ScenarioProcessor
from scenparse.utils import read_data
from scenparse.utils.gen_sumocfg import gen_sumocfg

app = typer.Typer()


@app.command()
def convert(
    input_path: str = typer.Argument(
        ...,
        help="Path to Waymo data - can be a single tfrecord file or directory containing multiple tfrecord files",
    ),
    output_dir: str = typer.Option(
        "sumo_maps",
        "--output-dir",
        "-o",
        help="Root directory for output SUMO map files",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="If input is a directory, whether to recursively process tfrecord files in all subdirectories",
    ),
    scenario_id: Optional[str] = typer.Option(
        None,
        "--scenario-id",
        help="Optional: Process only this specific scenario ID",
    ),
):
    """
    Convert Waymo scenario data to SUMO map files and generate corresponding configuration files.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all tfrecord files to process
    tfrecord_files = []
    if input_path.is_file():
        if input_path.suffix == ".tfrecord" or "tfrecord" in input_path.name:
            tfrecord_files.append(input_path)
    else:
        pattern = "**/*.tfrecord" if recursive else "*.tfrecord"
        tfrecord_files.extend(input_path.glob(pattern))

    if not tfrecord_files:
        typer.echo(f"No tfrecord files found in {input_path}")
        raise typer.Exit(1)

    # Process each tfrecord file
    for tfrecord_file in tfrecord_files:
        typer.echo(f"Processing file: {tfrecord_file}")

        try:
            # Read scenario data
            scenarios = read_data(str(tfrecord_file))

            # Process each scenario
            for scenario in scenarios:
                current_scenario_id = scenario.scenario_id

                # Skip if scenario_id is specified and doesn't match
                if scenario_id and current_scenario_id != scenario_id:
                    continue

                # Create subfolder for each scenario using its ID
                scenario_dir = output_dir / current_scenario_id
                scenario_dir.mkdir(parents=True, exist_ok=True)

                # Generate SUMO map files
                sp = ScenarioProcessor(scenario)
                sp.generate_sumonic_tls()
                sp.save_SUMO_netfile(
                    base_dir=str(scenario_dir),
                    filename=f"{current_scenario_id}.net.xml",
                )

                # Generate sumocfg file for each scenario
                gen_sumocfg(str(output_dir), current_scenario_id, empty_route_file=True)

                # If processing specific scenario_id and found it, we can break
                if scenario_id and current_scenario_id == scenario_id:
                    typer.echo(
                        f"Successfully processed requested scenario {scenario_id}"
                    )
                    return

            if not scenario_id:
                typer.echo(f"Successfully processed {len(scenarios)} scenarios")

        except Exception as e:
            typer.echo(f"Error processing file {tfrecord_file}: {str(e)}", err=True)
            continue

    if scenario_id:
        typer.echo(
            f"Warning: Requested scenario {scenario_id} not found in any processed files"
        )


if __name__ == "__main__":
    app()
