"""dijkstra.py
    Helper function for implementing dijkstras algorithm 
"""

def get_next_node(parent, end, base):
    """Recurses through a parent list from end to find the base node, and then returning the child of the base 
    Args:
        parent (array): a list of parents to node i
        end (int): end node
        base (int): our wanted base node

    Returns:
        int: the child of the base node 
    """
    if parent[end]==base:
        return end
    else:
        return get_next_node(parent,parent[end],base)