class Edge:
    def __init__(self, adjacent_node, start_ph, l):
        self.adjacent_node = adjacent_node
        self.pheromone = start_ph
        self.len = l
        self.closeness = 1 / l
