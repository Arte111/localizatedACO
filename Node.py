class Node:
    index: int

    def __init__(self, index, x=None, y=None, z=None):
        self.index = int(index)
        self.edges = {}
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        res = f"{self.index}: "
        for node, _ in self.edges.items():
            res += f"{node.index} "
        return res
