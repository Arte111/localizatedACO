import numpy as np


class Node:
    index: int

    def __init__(self, index, coordinates):
        self.index = int(index)
        self.edges = {}
        self.cords = np.array(coordinates)

    def __str__(self):
        res = f"{self.index} {' '.join(map(str, self.cords))}: "
        for node, _ in self.edges.items():
            res += f"{node.index} "
        return res
