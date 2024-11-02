# TeraSim Data Zoo

A toolkit for integrating Waymo Open Motion Dataset with the TeraSim simulation platform.

## Overview

TeraSim Data Zoo provides tools to:
1. Convert Waymo traffic scenarios into SUMO-compatible formats
2. Run TeraSim simulations using the converted Waymo data
3. Generate detailed simulation logs and analysis

## Features

- Waymo to SUMO conversion pipeline
- Scenario-specific simulation configuration
- GUI/headless simulation modes
- Detailed logging and data extraction
- Batch processing support for multiple scenarios

## Prerequisites

- Python 3.7+
- terasim
- scenparse
- SUMO (Simulation of Urban MObility)
- Additional Python packages:
  - typer (CLI interface)
  - loguru (logging)

## Installation

1. Clone the repository:
    ```bash
    git clone [repository-url]
    cd terasim-data-zoo
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Converting Waymo Data

Convert Waymo scenarios to SUMO format using:
```bash
python waymo_to_sumo.py INPUT_PATH [OPTIONS]
```

Options:
  -o, --output-dir TEXT     Root directory for output SUMO map files  
  -r, --recursive           Process tfrecord files recursively  
  --scenario-id TEXT        Process specific scenario ID  

### Running Simulations

Execute TeraSim simulation with converted data:
```bash
python waymo_terasim_main.py TFRECORD_PATH MAP_BASE_DIR [OPTIONS]
```

Options:
  --scenario-id TEXT        Specific scenario to simulate  
  --name TEXT               Experiment name  
  -d, --dir TEXT           Output directory  
  --gui / --no-gui         Enable/disable SUMO GUI  

### Example Usage

See the provided example script for a complete workflow:
```bash
./examples/waymo_example.sh
```

## Project Structure

```
terasim-data-zoo/
├── waymo_to_sumo.py        # Waymo to SUMO converter
├── waymo_terasim_main.py   # Main simulation runner
├── examples/
│   └── waymo_example.sh    # Example usage script
└── output/                 # Simulation outputs
```

## License

This project is closed source and not available for public use or distribution.

## Contact

For bug reports and feature requests, please open an issue in the repository.

---

Part of the TeraSim ecosystem. For more information about TeraSim, visit [TeraSim Documentation](https://github.com/michigan-traffic-lab/TeraSim).
