
MOVEMENTS = ["NORMAL","DIAGONAL","HALT"]

class Movement: 
    """Describing the movement of a pedestrian in the model
    """
    def __init__(self,type,diagonal):
        """
        Args:
            type (str): Movement indicator
            diagonal (bool): Specifies if diagonal movement is possible
        Raises:
            ValueError: If constructor is called with invalid type
        """
        if (not self.valid_type(type)):
            raise ValueError("Invalid movement")
        self.type = type
        self.diagonal = diagonal
    
    def valid_type(self,type):
        for movement in MOVEMENTS:
            if (movement == type):
                return True
        return False
    
    def get_type(self):
        return self.type
    
    def diagonal_movement(self):
        return self.diagonal
