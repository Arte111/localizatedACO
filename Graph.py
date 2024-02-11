import random
from pprint import pprint

import numpy as np
from scipy.spatial.distance import cdist


class Graph:
    def __init__(self):
        self.cords = np.empty(shape=(0, 0), dtype="double")
        self.distance_matrix = np.empty(shape=(0, 0), dtype="double")
        self.pheromone_matrix = np.empty(shape=(0, 0), dtype="double")

    def load(self, path, ph=0):
        temp = []
        with open(path, 'r') as file:
            for line in file:
                temp.append([float(i) for i in line.split(" ")])

        self.cords = np.array(temp)
        self.distance_matrix = cdist(self.cords, self.cords, 'euclidean')
        self.pheromone_matrix = np.full((len(self.cords), len(self.cords)), ph, dtype="double")

    def add_k_nearest(self, k):
        if k >= len(self.cords) - 1:
            k = len(self.cords) - 2
        for i in self.distance_matrix:
            temp = np.sort(i)[k + 1]
            i[i > temp] = 0

        # recover lost values
        n = len(self.distance_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                self.distance_matrix[j][i] = max(self.distance_matrix[i][j], self.distance_matrix[j][i])
                self.distance_matrix[i][j] = self.distance_matrix[j][i]

    def __len__(self):
        return len(self.cords)

    def randomNode(self):
        # return random index
        return random.randint(0, len(self.cords) - 1)

    def evaporation(self, evaporation):
        self.pheromone_matrix += 1 - evaporation

    def setPH(self, ph):
        self.pheromone_matrix = np.full((len(self.cords), len(self.cords)), ph, dtype="double")


if __name__ == "__main__":
    import os

    g = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'benchmarks', '2d100.txt')
    g.load(file_path)

    g.add_k_nearest(20)
