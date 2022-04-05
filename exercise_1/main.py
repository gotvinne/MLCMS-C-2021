"""Main simulation routine
"""

from CLI import cli_menu
from GUI.grid import Grid

simulate = True
while (simulate):
    simulation_steps = 40


    simulation_dict = cli_menu()
    #simulation_dict = {"N":["4"],"P":["0,0"],"O":["1,1"],"T":["3,3"]}

    grid = Grid(simulation_dict,simulation_steps)

    ans = input("Terminate simulation? (y/n) ").upper()
    if (ans == 'Y' or ans == "YES" or ans == "y"):
        simulate = False 
    