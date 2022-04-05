"""Modifying a vadere scenario file and storing it in folder
"""

from json import load, dump

RIMEA3_FILEPATH = "vadere.master.windows/Scenarios/ModelTests/TestOSM/scenarios/rimea_06_corner.scenario"
PED_FILEPATH = "vadere.master.windows/vadere_cli/corner_ped.json"

# Positions for task 3
X = 11.9
Y = 2.0

def read_scenario(filepath):
    """Reads the base scenario
    Returns:
        [JSONDecoder]: Object representing a json file
    """
    with open(filepath) as json_file:
        old_scenario = load(json_file)

    return old_scenario

def get_pedestrian(filepath):
    """Reading a json file representing the pedestrian used in simulation to a JSONDecoder object
    Returns:
        [JSONDecoder]: Object representing a json file
    """
    with open(filepath) as json_file: 
        ped = load(json_file)
    return ped

def spesify_pedestrian(x,y):
    """Takes a scenario json and returns the scenario 
    Args:
        scenario (JSONDecoder): JSON-object holding a scenario
        x (float): x position
        y (float): y position
    """
    ped = get_pedestrian(PED_FILEPATH)

    # Set position:
    position_dict = {"x" : x, 
                    "y" : y }
    ped["position"] = position_dict

    return ped

def write_new_scenario(filename,x,y):
    """Creating the json file representing the updated scenario and writing it to a file
    """
    old_scenario = read_scenario(RIMEA3_FILEPATH)
    dynamic_element = old_scenario["scenario"]["topography"]["dynamicElements"]
    dynamic_element.append(spesify_pedestrian(x,y))
    old_scenario["scenario"]["topography"]["dynamicElements"] = dynamic_element

    with open("vadere.master.windows/vadere_cli/"+filename,"w") as file:
        dump(old_scenario,file,indent=2)

write_new_scenario("rimea_06_updated.scenario",X,Y)
   

