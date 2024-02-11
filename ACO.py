import os
import time
from pprint import pprint

import numpy as np
from Graph import Graph


class ACO:
    def __init__(self, graph):
        self.graph = graph

    def step(self, ant_count, A, B, Q, evap):
        print("step")
        node_count = len(self.graph)
        better_path = []
        better_path_len = float("inf")
        for _ in range(ant_count):
            path = np.array([self.graph.randomNode()])

            while len(path) < node_count:
                # TODO: заменить np.arange на поиск доступных вершин в графе
                enable = np.setdiff1d(np.arange(len(self.graph)), path)

                probabilities = np.array([pow(self.graph.distance_matrix[path[-1]][i], A) *
                                          pow(self.graph.pheromone_matrix[path[-1]][i], B)
                                          for i in enable])
                probabilities /= probabilities.sum()

                chosen_index = np.random.choice(enable, p=probabilities)

                path = np.append(path, chosen_index)

            # Calculate path length
            path_len = 0.0
            for j in range(len(path) - 1):
                path_len += self.graph.distance_matrix[path[j]][path[j + 1]]
            path_len += self.graph.distance_matrix[path[-1]][path[0]]

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
        self.graph.pheromone_matrix[better_path[-1]][better_path[0]] += ph
        self.graph.pheromone_matrix[better_path[0]][better_path[-1]] += ph

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
        # TODO: написать отображение графика эфективности
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
    graph.load(file_path)

    aco = ACO(graph)
    aco.run_print(100, 1, 3, 900, 0.4, 0.4, 20)

