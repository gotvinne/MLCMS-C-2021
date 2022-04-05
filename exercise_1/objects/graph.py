
from queue import PriorityQueue
INITIAL_WEIGHT = -1

class Graph:
    """ Representation of a graph with nodes and edges required for dijkstra's algorithm
    """
    def __init__(self, n_nodes):
        self.nodes = n_nodes
        self.edges = [[INITIAL_WEIGHT for i in range(n_nodes)] for j in range(n_nodes)]
        self.visited = []

    def add_edge(self, u, v, weight):
        """Adds an edge with weight between node u and v

        Args:
            u (int): node u
            v (int): node v
            weight (int): weight

        Raises:
            ValueError: If there is already a weight between the nodes, updates will be ignored
        """
        if self.edges[u][v] == INITIAL_WEIGHT:
            self.edges[u][v] = weight
            self.edges[v][u] = weight
        else:
            raise ValueError("Already weight: "+str(weight)+" on node "+str(u)+" and "+str(v))

    def dijkstra(self, start_node):
        """Implements the general Dijkstra's algorithm (see 4,1 in reort)

        Args:
            start_node (int): Source node to calculate weights from 

        Returns:
            array: list of the predecessor node at each node i 
        """
        node_dict = {node:float('inf') for node in range(self.nodes)}
        node_dict[start_node] = 0
        parent=self.nodes*[INITIAL_WEIGHT]

        queue = PriorityQueue()
        queue.put((0, start_node))

        while not queue.empty():
            (_, current_node) = queue.get()
            self.visited.append(current_node)
            
            for neighbour in range(self.nodes):
                if self.edges[current_node][neighbour] != INITIAL_WEIGHT:
                    distance = self.edges[current_node][neighbour]
                    if neighbour not in self.visited:
                        old_cost = node_dict[neighbour]
                        new_cost = node_dict[current_node] + distance
                        if new_cost < old_cost:
                            queue.put((new_cost, neighbour))
                            parent[neighbour] = current_node
                            node_dict[neighbour] = new_cost
        return parent