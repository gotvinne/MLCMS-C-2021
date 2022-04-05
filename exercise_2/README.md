
# Exercise 2: Using and analysing simulation software 

## Task 1 & 2
### Run the VADERE simualtion software to test RiMEA scenarios and use different models

These task are all solvable within the VADERE software. The only thing to do where is therefore just to download the build software on the follwing link.

http://www.vadere.org/releases/

## Task 3
### Run VADERE CLI

Run following command in terminal to generate the updated scenario-file and simulate it using CLI
```console
chmod 755 simulate_updated_scenario.sh
./simulate_updated_scenario.sh
```

*Run standard scenario from CLI:*
```console
java -jar vadere-console.jar scenario-run --scenario-file "Scenarios/ModelTests/TestOSM/scenarios/rimea_06_corner.scenario" --output-dir="vadere_cli"
```

*Run updated scenario from CLI:*
```console
java -jar vadere-console.jar scenario-run --scenario-file "vadere_cli/rimea_06_updated.scenario" --output-dir="vadere_cli"
```

## Task 4
### updating the VADERE simulation with the SI Model

The integration of the new SI Model can be found in the folder varere_SI_src. 
The software can be built by using a IDE and running the entrypoint varere_SI_src/VadereGui/src/org/vadere/gui/projectview/VadereApplication.java
Or by built it using MVN.
1. git clone vadere_SI_src
2. mvn clean
3. mvn -Dmaven.test.skip=true package

## Task 5
### updating the VADERE simulation with the SIR Model

The integration of the new SI Model can be found in the folder varere_SIR_src. 
The software can be built by using a IDE and running the entrypoint varere_SIR_src/VadereGui/src/org/vadere/gui/projectview/VadereApplication.java
Or by built it using MVN.
1. git clone vadere_SIR_src
2. mvn clean
3. mvn -Dmaven.test.skip=true package
