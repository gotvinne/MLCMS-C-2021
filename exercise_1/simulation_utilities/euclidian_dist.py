"""euclidian_dist.py
"""
from math import sqrt

def calculate_dist(cell_pos, target_pos):
        dx = cell_pos[0] - target_pos[0]
        dy = cell_pos[1] - target_pos[1]
        dist = sqrt(dx**2 + dy**2)
        return dist