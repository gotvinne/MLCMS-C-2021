
from dataclasses import dataclass

@dataclass
class Transition: 
    """Struct holding the position transition of a pedestrian
    """
    old: tuple
    new: tuple
