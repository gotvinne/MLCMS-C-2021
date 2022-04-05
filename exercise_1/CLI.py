"""Command line interface creating determing a simulation configuration

Returns:
    dictionary : Data structure in the correct format for the GUI-object to interpret
"""
import os, sys
from objects.pedestrian import VALID_SPEEDS

def input_simulation_data():
    """Get simulation data from user
    """
    n = input("Insert the size of the grid (i.e input n for nxn grid): ")
    if (n.isalpha()):
        print("\nINVALID DIMENTIONS GIVEN")
        print("Please retype your simulation values: ")
        return None

    peds_str = input("Insert pedestrial info (pos0,pos1,speed) (i.e 1,2,2.0 2,3,1.5 ): ")
    peds_lst = peds_str.split()

    for pos_str in peds_lst:
        if (not valid_ped_pos_str(pos_str,int(n))):
            return None

    obstacles_str = input("Insert the positions of obstacles (i.e 1,2 2,3 ): ")
    if (len(obstacles_str) != 0):
        obstacles_lst = obstacles_str.split()

        for pos_str in obstacles_lst:
            if (not valid_pos_str(pos_str,int(n))):
                return None
    else: 
        obstacles_lst = []

    target = input("Insert the position of the target (i.e 1,2 ): ")
    if (not valid_pos_str(target,int(n))):
        return None
    
    if (not no_collissions(peds_lst,obstacles_lst,target)):
        print("\nDATA IS NOT COLLISSION FREE")
        print("Please retype your simulation values: ")
        return None

    return [n,peds_lst,obstacles_lst,target]

def no_collissions(ped_lst, obstacles_lst, target):
    """Determine if none of the objects are colliding
    Returns:
        bool: Simulation configuration has no overlapping objects
    """
    positions = [target]

    for ped in ped_lst:
        positions.append(ped)
    for obst in obstacles_lst:
        positions.append(obst)
    
    if (len(positions)==len(set(positions))):
        return True
    return False

def valid_ped_pos_str(pos,n):
    if (pos.find(",") == -1):
        print("\nINVALID POSITIONS GIVEN")
        print("Please retype your simulation values: ")
        return False

    if (pos[0].isalpha() or pos[2].isalpha()):
        print("\nDATA IS NOT INTERGER")
        print("Please retype your simulation values: ")
        return False
    
    num_lst = [ float(num) for num in pos.split(",") ]
    if (num_lst[0] < 0 or num_lst[0] >= n or num_lst[1] < 0 or num_lst[1] >= n):
        print("\nCOORDINATE IS OUTSIDE GRID")
        print("Please retype your simulation values: ")
        return False
    if (num_lst[2] < VALID_SPEEDS[0] or VALID_SPEEDS[1] < num_lst[2]):
        print("\nInvalid pedestrian speed")
        print("Please retype your simulation values: ")
        return False 
    return True

def valid_pos_str(pos,n):
    if (pos.find(",") == -1):
        print("\nINVALID POSITIONS GIVEN")
        print("Please retype your simulation values: ")
        return False

    if (pos[0].isalpha() or pos[2].isalpha()):
        print("\nDATA IS NOT INTERGER")
        print("Please retype your simulation values: ")
        return False
    
    num_lst = [ float(num) for num in pos.split(",") ]
    if (num_lst[0] < 0 or num_lst[0] >= n or num_lst[1] < 0 or num_lst[1] >= n):
        print("\nCOORDINATE IS OUTSIDE GRID")
        print("Please retype your simulation values: ")
        return False
    return True

def write_simulation_file(simulation_data):
    """Writes simulation configuration to file
    """
    with open(os.path.join(sys.path[0],"simulations/custom_simulation.txt"),'w+') as file:
        pedestrials = ""
        obstacles = ""

        for pedestrial in simulation_data[1]:
            pedestrials += pedestrial
            pedestrials += " "
        for obstacle in simulation_data[2]:
            obstacles += obstacle
            obstacles += " "
        
        file.write("N | "+simulation_data[0]+"\n")
        file.write("P | "+pedestrials+"\n")
        file.write("O | "+obstacles+"\n")
        file.write("T | "+simulation_data[3])

def read_simulation_file(filepath):
    """Read simulation configuration from file
    """
    simulation = {}
    with open(os.path.join(sys.path[0],filepath)) as file: 
        for line in file:
            key, values = line.split("|")

        # Strip strings for whitespaces
            key = key.strip()
            values = values.strip()

            simulation[key] = values.split()
    return simulation

def cli_menu():
    print("-----------------------------------------------")
    print("WELCOME TO GROUP C's CROWD MODELING SIMULATION ")
    print("-----------------------------------------------")
    
    # Print alternatives:
    print("1: Create own simulation")
    print("2: Run Task 2 simulation")
    print("3: Run Task 3 simulation")
    print("4: Run Task 4 simulation")
    print("5: RiMEA TEST1 simulation")
    print("6: RiMEA TEST2 simulation")
    print("7: RiMEA TEST3 simulation")
    print("8: RiMEA TEST4 simulation")
    
    choose = int(input("Select one on the simulations alternatives: (1, 2, 3, 4, 5, 6, 7 or 8) "))

    simulation = None
    if (choose == 1):
        simulation_data = None
        while (not simulation_data):
            try:
                simulation_data = input_simulation_data()
            except:
                print("WRONG FORMAT")
                print("Please retype your simulation values: ")

        write_simulation_file(simulation_data)
        simulation = read_simulation_file("simulations/custom_simulation.txt")
    elif (choose == 2):
        simulation = read_simulation_file("simulations/exercise2_sim.txt")
    elif (choose == 3):
        simulation = read_simulation_file("simulations/exercise3_sim.txt")
    elif (choose == 4):
        simulation = read_simulation_file("simulations/exercise4_sim.txt")
    elif (choose == 5):
        simulation = read_simulation_file("simulations/RiMEA_TEST1.txt")
    elif (choose == 6):
        simulation = read_simulation_file("simulations/RiMEA_TEST2.txt")
    elif (choose == 7):
        simulation = read_simulation_file("simulations/RiMEA_TEST3.txt")
    elif (choose == 8):
        simulation = read_simulation_file("simulations/RiMEA_TEST4.txt")
    return simulation
        