import os
import time
import random

import numpy as np
# from Graph import Graph
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

    def evaporation(self, evaporation):
        self.pheromone_matrix += 1 - evaporation

    def setPH(self, ph):
        self.pheromone_matrix = np.full((len(self.cords), len(self.cords)), ph, dtype="double")


class ACO:
    def __init__(self, graph):
        self.graph = graph

    def step(self, ant_count, A, B, Q, evap):
        node_count = len(self.graph)
        better_path = []
        better_path_len = float("inf")
        for _ in range(ant_count):
            path = np.empty(node_count, dtype=int)
            path[0] = random.randint(0, node_count - 1)
            for i in range(1, node_count):
                # for i in range(1, node_count)
                enable = np.setdiff1d(np.nonzero(self.graph.distance_matrix[path[i - 1]]), path[:i])
                if len(enable) == 0:
                    break  # path not found

                probabilities = np.power(self.graph.distance_matrix[path[i - 1]][enable], A) * \
                                np.power(self.graph.pheromone_matrix[path[i - 1]][enable], B)
                probabilities /= probabilities.sum()

                path[i] = random.choices(enable, weights=probabilities)[0]

            if self.graph.distance_matrix[path[0]][path[-1]] == 0:
                continue  # because we didn't find valid path

            # Calculate path length
            path_len = np.sum(self.graph.distance_matrix[path[:-1], path[1:]]) + \
                       self.graph.distance_matrix[path[-1], path[0]]

            # Update best path
            if better_path_len > path_len:
                better_path = path
                better_path_len = path_len

        # костыль(.. ну а что поделать
        """if len(better_path) == 0:
            # print("no path")
            return float('inf'), None"""

        # Evaporation
        self.graph.evaporation(evap)

        # add ph by best_path
        ph = Q / better_path_len
        for j in range(len(better_path) - 1):
            self.graph.pheromone_matrix[better_path[j]][better_path[j + 1]] += ph
            self.graph.pheromone_matrix[better_path[j + 1]][better_path[j]] += ph
        try:
            self.graph.pheromone_matrix[better_path[-1]][better_path[0]] += ph
            self.graph.pheromone_matrix[better_path[0]][better_path[-1]] += ph
        except IndexError:
            pass

        return better_path_len, better_path

    def run_performance(self, ant_count, A, B, Q, evap, start_ph, worktime):
        performance = 0
        self.graph.setPH(start_ph)
        # best_path_len = self.lenRandomPath()
        best_path_len = float("inf")
        startTime = time.time()
        lastTime = startTime
        while time.time() - startTime < worktime:
            bpl, _ = self.step(ant_count=ant_count, A=A, B=B, Q=Q, evap=evap)
            if best_path_len > bpl:
                performance += (time.time() - lastTime) * bpl
                lastTime = time.time()
                best_path_len = bpl

        return performance

    def run_print(self, ant_count, A, B, Q, evap, start_ph, worktime):
        print("let's gooo")
        # TODO: написать отображение графика эффективности
        self.graph.setPH(start_ph)
        # best_path_len = self.graph.lenRandomPath()
        best_path_len = float("inf")
        startTime = time.time()
        lastTime = startTime
        while time.time() - startTime < worktime:
            bpl, _ = self.step(ant_count=ant_count, A=A, B=B, Q=Q, evap=evap)
            if best_path_len > bpl:
                print(f"{time.time() - startTime} {bpl}")
                lastTime = time.time()
                best_path_len = bpl


if __name__ == "__main__":
    graph = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'benchmarks', '2d100.txt')
    graph.load(file_path, ph=0.4)
    graph.add_k_nearest(99)

    aco = ACO(graph)
    start = time.time()
    aco.step(20, 1, 3, 100, 0.4)
    aco.step(20, 1, 3, 100, 0.4)
    aco.step(20, 1, 3, 100, 0.4)
    aco.step(20, 1, 3, 100, 0.4)
    aco.step(20, 1, 3, 100, 0.4)
    aco.step(20, 1, 3, 100, 0.4)
    aco.step(20, 1, 3, 100, 0.4)
    aco.step(20, 1, 3, 100, 0.4)
    aco.step(20, 1, 3, 100, 0.4)
    aco.step(20, 1, 3, 100, 0.4)
    finish = time.time()
    print(finish - start)
    """for _ in range(10):
        print(aco.run_performance(20, 1, 3, 100, 0.4, 0.4, 20))"""
