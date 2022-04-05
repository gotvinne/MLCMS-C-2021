
from objects.transition import Transition
from objects.pedestrian import Pedestrian
from objects.cell import Cell
from objects.graph import Graph

from simulation_utilities.euclidian_dist import calculate_dist
from simulation_utilities.dijkstra import get_next_node
import simulation_utilities.logic as logic

import numpy as np 

SPEED_SIZE = 3
REACHED_TARGET = (-1,-1)

class DijkstraTable:  
    """ Structure representing the Cellular automaton and forcasting the simulation
    """
    def __init__(self,n,simulation_dict): 
        """ 
        Args:
            n (int): dimention
            simulation_dict (dictionary): Dictionary specifying distribution of pedestrian, object, target etc. 
        Raises:
            ValueError: Target not unique
        """
        self.n = n
        self.finished = np.empty(0)

        if (not logic.unique_target(simulation_dict)):
            raise ValueError("There is more than one target")
        self.target_pos = logic.get_target_pos(simulation_dict.get("T"))
        self.ped_pos_lst = logic.get_ped_info(simulation_dict.get("P")) 
        # Store simulations
        self.start_simulation = self.create_simulation(simulation_dict)
        self.current_simulation = self.start_simulation

    def create_simulation(self,simulation_dict): 
        """ Layout consist of a 2D numpy-matrix consisting of Cell-objects
        Raises:
            ValueError: Position does not fit grid
        Returns:
            [np.array[][]]: Matrix of Cell-objects 
        """
        simulation = np.array([[Cell(i,j,"E") for j in range(self.n)] for i in range(self.n)],dtype=object)
        for state, data in simulation_dict.items():
            for elem in data:
                data_lst = [float(num) for num in elem.split(",")]
                data_lst[0] = int(data_lst[0])
                data_lst[1] = int(data_lst[1])
                if (not logic.valid_position(data_lst,self.n)):
                    raise ValueError("Position is outside the grid")
                cell = simulation[data_lst[1]][data_lst[0]]
                if (state == "P"):
                    if (len(data_lst) == SPEED_SIZE):
                        ped = Pedestrian(data_lst[1],data_lst[0],data_lst[2])
                    else: 
                        ped = Pedestrian(data_lst[1],data_lst[0])
                    dist_to_target = calculate_dist((data_lst[1],data_lst[0]),self.target_pos)
                    ped.initialize_dist(dist_to_target)
                    cell.set_pedestrian(ped) 
                    continue
                cell.set_state(state)
        return simulation

    def update_simulation(self): 
        """ Updating the cells attributes in grid. Utilizes the distance error [err] to determine movement placement of a pedestrian 
        Returns:
            [Transitions]: Returns a list of Transitions object, determing the update function of a cell.
        """
        transitions = []  
        for index, pos in enumerate(self.ped_pos_lst):
            origin_pos = pos

            cell,ped,err = self.get_cell_info(pos)
            movement = logic.get_movement(err)
            if (movement.get_type() == "HALT"):
                self.forecast_real_dist(ped)
                continue
            
            pos = self.forecast_ped(index,pos,ped,cell,movement) 
            if (pos != REACHED_TARGET):
                self.forecast_real_dist(ped)
                # Extract new cell
                cell,ped,err = self.get_cell_info(pos)
            
            while ((not logic.accepted_err(err)) and pos != REACHED_TARGET):
                movement = logic.get_movement(err)
                pos = self.forecast_ped(index,pos,ped,cell,movement)
                if (pos != REACHED_TARGET):
                    cell,ped,err = self.get_cell_info(pos)
        
            transitions.append(Transition(origin_pos,pos)) 
        return transitions

    def forecast_ped(self,index,pos,ped,cell,movement):
        """ Inner update function for determing the placement of a pedestrian
        Args:
            index (int): Indexing a pedestrian in model
            pos (tuple): Position
            ped (Pedestrian): Current pedestrian object
            cell (Cell): Current cell object
            movement (Movement): Movement determined by distance error

        Returns:
            tuple : Optimal placement for a pedestrian for given environment
        """
        if (movement.get_type() == "HALT"): # With halt, we do not create transition object
            self.forecast_real_dist(ped)
            return pos
                
        adjacents = self.get_adjacents(pos[0],pos[1],movement.diagonal_movement())
        optimal_pos = self.optimal_cell(adjacents) #self.dijkstra(pos)

        if (optimal_pos != REACHED_TARGET):
            self.forecast_simulated_dist(ped,optimal_pos)
            self.current_simulation[optimal_pos[0]][optimal_pos[1]].set_pedestrian(ped)
            self.ped_pos_lst[index] = optimal_pos
        else: 
            self.finished = np.append(self.finished,ped)
            del self.ped_pos_lst[index]
        cell.empty()
        return optimal_pos
    
    def get_adjacents(self,pos_x,pos_y,diagonal):
        """
        Args:
            pos_x (int): x-position
            pos_y (int): y-position
            diagonal (bool): Allow diagonal step

        Returns:
            list: adjacents positions given input arguments
        """
        adjacents = []
        for i in range(-1,2):
            for j in range(-1,2):
                if (i == 0 and j == 0):
                    continue
                elif logic.valid_position((pos_x+i,pos_y+j),self.n):
                    if (self.current_simulation[pos_x+i][pos_y+j].get_state() != "O"):
                        if diagonal:
                            adjacents.append((pos_x+i,pos_y+j))
                        elif (i == 0 or j == 0):
                            adjacents.append((pos_x+i,pos_y+j))
        return adjacents

    def get_cell_info(self,pos):
        """ Interpreting the info stored in a cell
        Args:
            pos (tuple): position of cell
        Returns:
            Cell-object, Pedestian-object, and distance error
        """
        cell = self.current_simulation[pos[0]][pos[1]]
        ped = cell.get_pedestrian()
        err = ped.calculate_dist_err()
        return cell,ped,err

    def forecast_simulated_dist(self,ped,optimal_pos):
        dist_to_target = calculate_dist(optimal_pos,self.target_pos)
        ped.set_simulated_dist(dist_to_target)
    
    def forecast_real_dist(self,ped):
        ped.update_real_dist()
        ped.increment_step()
    
    def optimal_cell(self,adjacents):
        """ Finds the optimal adjacent in terms of distance to target
        Args:
            adjacents (list): Cell-objects
        Returns:
            tuple: Optimal step
        """
        cost, positions = ([] for _ in range(2))
        for adj in adjacents:
            cell = self.current_simulation[adj[0]][adj[1]]
            cell_pos = cell.get_position()
            
            positions.append(cell_pos)
            cost.append(calculate_dist(cell_pos,self.target_pos))

        if cost: 
            min_cost = min(cost)
            if (min_cost == 0):
                return REACHED_TARGET 
            else: 
                index = cost.index(min_cost)
                optimal_pos = positions[index]
                return optimal_pos
        else: # If cell is surronded
            return self.current_simulation[cell[0]][cell[0]].get_position()
    
    def get_dijkstra_transition(self,ped_pos):
        """Creates a graph from the current simulation of cells, and finds the optimal position using the Djikstra algorithm. 
        Args:
            ped_pos ([x,y]): position of the pedestrian
        Returns:
           ([x,y]): The optimal next position of the pedestrian
        """
        pos_to_node = [[i+self.n*j for i in range(self.n)] for j in range(self.n)]
        graph = Graph(self.n*self.n)
        for i in range(self.n):
            for j in range(self.n):
                current_cell = self.current_simulation[i][j]
                current_cell_pos = current_cell.get_position()
                current_node = pos_to_node[current_cell_pos[0]][current_cell_pos[1]]
                neighbours = self.get_adjacents(current_cell_pos[0],current_cell_pos[1],False)
                for neighbour in neighbours:
                    neighbour_node = pos_to_node[neighbour[0]][neighbour[1]]
                    if graph.edges[current_node][neighbour_node] == -1:
                        if self.current_simulation[neighbour[0]][neighbour[1]].get_state() == "O":
                            graph.add_edge(current_node,neighbour_node,50)
                        elif self.current_simulation[neighbour[0]][neighbour[1]].get_state() == "T":
                            graph.add_edge(current_node, neighbour_node,0)
                        elif self.current_simulation[neighbour[0]][neighbour[1]].get_state() == "P" and neighbour != ped_pos:
                            graph.add_edge(current_node, neighbour_node, 50)
                        else: 
                            cost = calculate_dist(neighbour, self.target_pos)
                            graph.add_edge(current_node,neighbour_node,cost)
  
        parent = graph.dijkstra(pos_to_node[ped_pos[0]][ped_pos[1]])
        next_node = get_next_node(parent, pos_to_node[self.target_pos[0]][self.target_pos[1]], pos_to_node[ped_pos[0]][ped_pos[1]])
        next_pos = [(index, row.index(next_node)) for index, row in enumerate(pos_to_node) if next_node in row]
        if next_pos[0] == self.target_pos:
            return REACHED_TARGET
        return next_pos[0]

    def get_finished(self):
        return self.finished

    def done_simulation(self):
        return (len(self.ped_pos_lst)==0)
