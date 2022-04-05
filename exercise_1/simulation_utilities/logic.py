"""Logic.py 
    Collects the simulation logic (helper functions) for the simulation object Table
"""
from objects.movement import Movement

# Thresholds for distance error
ERR_THRESHOLD = [-0.6,0.6]

def accepted_err(err):
    return (err < ERR_THRESHOLD[1])

def valid_position(pos,n):
    return (0 <= pos[0] and pos[0] < n and 0 <= pos[1] and pos[1] < n)

def get_movement(err):
    """Returns correct movement based on error
    Args:
        err (int): Distance error
    """
    # If error positiv -> slow simulation
    if (ERR_THRESHOLD[0] < err):
        return Movement("DIAGONAL",True)
    # If error negativ -> Fast simulation
    elif (ERR_THRESHOLD[1] > err):
        return Movement("HALT",False)
    else: 
        return Movement("NORMAL",False)

def unique_target(cell_dict): 
    target_list = cell_dict.get("T")
    return (len(target_list)==1) 

def get_ped_info(ped_pos_strings):
    """Formates a list of pedestrians
    """
    ped_pos_lst = [] 
    for str in ped_pos_strings:
        num_lst = str.split(",")
        ped_pos_lst.append((int(num_lst[1]),int(num_lst[0])))
    return ped_pos_lst

def get_target_pos(target_pos_lst):
    """Formating the position of target from dictionary
    Args:
        target_pos_lst (list): List containing a string position
    Returns:
        list: Formated correctly for simulation
    """
    pos = target_pos_lst[0]
    pos_lst = [ int(num) for num in pos.split(",") ]
    return (pos_lst[1],pos_lst[0])

