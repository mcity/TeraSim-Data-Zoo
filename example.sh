#!/bin/bash

# Configuration
SCENARIO_ID="19a486cd29abd7a7"
WAYMO_TFRECORD="uncompressed_scenario_validation_validation.tfrecord-00000-of-00150"
MAP_OUTPUT_DIR="womd_map_new"
EXPERIMENT_NAME="test_experiment"

echo "Starting conversion and simulation for scenario: ${SCENARIO_ID}"

# Step 1: Convert specific Waymo scenario to SUMO map
echo "Converting scenario ${SCENARIO_ID} to SUMO map..."
python waymo_to_sumo.py \
    ${WAYMO_TFRECORD} \
    --scenario-id ${SCENARIO_ID} \
    -o ${MAP_OUTPUT_DIR}

# Check if conversion was successful
if [ ! -f "${MAP_OUTPUT_DIR}/${SCENARIO_ID}/${SCENARIO_ID}.sumocfg" ]; then
    echo "Error: Map conversion failed! SUMO configuration file not found."
    exit 1
fi

# Step 2: Run TeraSimulation for the specific scenario
echo "Starting TeraSimulation for scenario ${SCENARIO_ID}..."
python waymo_terasim_main.py \
    ${WAYMO_TFRECORD} \
    ${MAP_OUTPUT_DIR} \
    --scenario-id ${SCENARIO_ID} \
    --name ${EXPERIMENT_NAME} \
    --gui

echo "Pipeline completed for scenario: ${SCENARIO_ID}!" 