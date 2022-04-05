
VALID_SPEEDS = [0.5,2.5]

class Pedestrian:
    """Module representing a pedestrian
    """
    def __init__(self, x_start, y_start, speed=1.0):
        """
        Args:
            x_start (int): x start position in the cellular automaton
            y_start (int): y start position in the cellular automaton
            speed (float): Modelled speed. Defaults to 1.0.

        Raises:
            ValueError: If constructor is called with invalid speed
        """
        self.start_position = (x_start,y_start)
        if (not self.valid_speed(speed)):
            raise ValueError("Speeds out of range!")
        self.speed = speed 
        self.simulation_steps = 1

    def initialize_dist(self,dist):
        """
        Args:
            dist (float): Describe the initial distance to target
        """
        self.simulated_target_dist = dist
        self.real_target_dist = dist
        self.target_dist = dist
    
    def calculate_dist_err(self):
        """
        Returns:
            float: Simulation distance error, Err = simulated - real
        """
        return (self.simulated_target_dist-self.real_target_dist)

    def increment_step(self):
        self.simulation_steps += 1
    
    def valid_speed(self,speed):
        return (VALID_SPEEDS[0] <= speed and speed <= VALID_SPEEDS[1])

    def update_real_dist(self):
        self.real_target_dist -= self.speed
    
    def set_simulated_dist(self,dist):
        self.simulated_target_dist = dist
    
    def get_results(self):
        return [self.start_position,self.speed,self.target_dist,self.simulation_steps]

   

    

    

    



