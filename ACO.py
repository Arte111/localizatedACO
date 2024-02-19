import os
import time

import numpy as np

from Graph import Graph


class ACO:
    def __init__(self, graph):
        self.graph = graph

    def step(self, ant_count, A, B, Q, evap):
        # TODO: написать функцию на С\Python API (да, сложно, надо.)
        node_count = len(self.graph)
        better_path = []
        better_path_len = float("inf")
        for _ in range(ant_count):
            path = [self.graph.randomNode()]

            while len(path) < node_count:
                enable = {node: edge for node, edge in path[-1].edges.items() if node not in path}
                if not enable:
                    break

                probabilities = np.array([pow(edge.pheromone, A) * pow(edge.closeness, B) for edge in enable.values()])
                probabilities /= probabilities.sum()

                chosen_index = np.random.choice(len(enable), p=probabilities)
                path.append(list(enable.keys())[chosen_index])

            # Check valid path
            if len(path) != node_count or path[-1].edges.get(path[0]) is None:
                continue

            # Calculate path length
            path_len = 0.0
            for j in range(len(path)):
                path_len += path[j].edges[path[(j + 1) % len(path)]].len

            # Update best path
            if better_path_len > path_len:
                better_path = path
                better_path_len = path_len

        # костыль(.. ну а что поделать
        if len(better_path) == 0:
            # print("no path")
            return float('inf'), None

        # Evaporation
        self.graph.evaporation(evap)

        # add ph by best_path
        ph = Q / better_path_len
        for j in range(len(better_path)):
            self.graph.graph[better_path[j].index].edges[better_path[(j + 1) % len(better_path)]].pheromone += ph
            self.graph.graph[better_path[(j + 1) % len(better_path)].index].edges[better_path[j]].pheromone += ph

        return better_path_len, better_path

    def stepElite(self, best_path, best_path_len, Q, ant_count=1):
        ph = (Q / best_path_len) * ant_count
        for j in range(len(best_path)):
            self.graph.graph[best_path[j].index].edges[best_path[(j + 1) % len(best_path)]].pheromone += ph
            self.graph.graph[best_path[(j + 1) % len(best_path)].index].edges[best_path[j]].pheromone += ph

    def run(self, try_count, ant_count, A, B, Q, evap, start_ph):
        self.graph.setPH(start_ph)
        best_path = []
        best_path_len = float('inf')
        for _ in range(try_count):
            bpl, bp = self.step(ant_count=ant_count, A=A, B=B, Q=Q, evap=evap)
            if bpl < best_path_len:
                best_path_len = bpl
                best_path = bp
            if bp is not None:
                self.stepElite(bp, bpl, Q)

        return best_path_len, best_path

    def run_performance(self, ant_count, A, B, Q, evap, start_ph, worktime):
        performance = 0
        self.graph.setPH(start_ph)
        best_path_len = self.graph.lenRandomPath()
        # best_path_len = float("inf")
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
        best_path_len = self.graph.lenRandomPath()
        # best_path_len = float("inf")
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
    graph.add_k_nearest_edges(99)

    aco = ACO(graph)
    for _ in range(10):
        print(aco.run_performance(20, 1, 3, 100, 0.4, 0.4, 20))
