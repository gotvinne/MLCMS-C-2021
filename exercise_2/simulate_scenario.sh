#!/bin/bash
# Bash script calling the vadere cli 
cd vadere.master.windows

# Simulate 
java -jar vadere-console.jar scenario-run --scenario-file "Scenarios/ModelTests/TestOSM/scenarios/rimea_06_corner.scenario" --output-dir="vadere_cli"