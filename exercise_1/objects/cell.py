
                                    
STATES = ["E","P","O","T"] 

class Cell:
    """ Class for a single cell in the grid
    """
    def __init__(self,x,y,state):
        """
        Args:
            x (int): x-position in grid
            y (int): y-position in grid
            state (string): state of the cell, determined by STATES

        Raises:
            ValueError: Constructor called with invalid state
        """
        self.position = (x,y)
        self.pedestrian = None
        if not self.valid_state(state):
            raise ValueError("State not E, P, O or T")
        self.state = state
        
    def valid_state(self,state):
        for valid_state in STATES:
            if state == valid_state: 
                return True
        return False
    
    def get_state(self): 
        return self.state
    
    def get_position(self):
        return self.position

    def get_pedestrian(self):
        return self.pedestrian
    
    def set_state(self, state):
        self.valid_state(state)
        self.state = state
    
    def set_pedestrian(self,pedestrian):
        if (not self.pedestrian):
            self.state = "P"
            self.pedestrian = pedestrian
        else: 
            raise ValueError("This cell is not empty!")
    
    def empty(self):
        self.state = "E"
        self.pedestrian = None

    