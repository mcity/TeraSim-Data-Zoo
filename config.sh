#!/bin/bash

# Path configurations
WAYMO_TFRECORD="uncompressed_scenario_validation_validation.tfrecord-00000-of-00150"
MAP_OUTPUT_DIR="./womd_map_new"

# Default values
DEFAULT_SCENARIO_ID="19a486cd29abd7a7"
DEFAULT_EXPERIMENT_NAME="test_experiment"

# Logging configuration
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/conversion_$(date +%Y%m%d_%H%M%S).log" 