#!/bin/bash
# Bash script calling the vadere cli with a given scenario
cd vadere.master.windows

# Generate scenario
python3 vadere_cli/modify_scenario.py

# Simulate 
java -jar vadere-console.jar scenario-run --scenario-file "vadere_cli/rimea_06_updated.scenario" --output-dir="vadere_cli"