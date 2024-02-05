import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy import random
from scipy.spatial.distance import cdist
import matplotlib.animation as animation
import random

from Edge import Edge
from Node import Node


class Graph:
    graph: list[Node]

    def __init__(self):
        self.graph = []

    def load_from_file(self, file_path):
        self.graph.clear()
        with open(file_path, 'r') as file:
            for index, line in enumerate(file):
                coordinates = line.strip().split()
                cords = list(map(float, coordinates))
                node = Node(index, cords)
                self.graph.append(node)

    def upload(self, file_path):
        with open(file_path, 'w') as file:
            for node in self.graph:
                print(f"{' '.join(map(str, node.cords))}", file=file)

    def generate_random_3d_graph(self, node_count):
        self.graph.clear()
        for index in range(node_count):
            x, y, z = np.random.uniform(-100, 100, size=3)
            node = Node(index, [x, y, z])
            self.graph.append(node)

    def generate_random_graph(self, node_count, space_dimension=3, points_per_cluster=0, n_clusters=1):
        self.graph.clear()
        if not points_per_cluster:
            for index in range(node_count):
                cords = np.random.uniform(-100, 100, size=space_dimension)
                node = Node(index, cords)
                self.graph.append(node)
        else:
            total_points = points_per_cluster * n_clusters
            if node_count >= total_points:
                for cluster in range(n_clusters):
                    cluster_center = np.random.uniform(-100, 100, size=space_dimension)
                    for point_index in range(points_per_cluster):
                        offset = np.random.normal(0, 10, size=space_dimension)
                        coords = cluster_center + offset
                        node = Node(len(self.graph), coords)
                        self.graph.append(node)

                # Распределение равномерно оставшихся точек
                remaining_points = node_count - total_points
                for index in range(remaining_points):
                    coords = np.random.uniform(-100, 100, size=space_dimension)
                    node = Node(len(self.graph), coords)
                    self.graph.append(node)
            else:
                print("Error: node_count should be greater than or equal to points_per_cluster * n_clusters")

    def add_k_nearest_edges(self, k):
        for node in self.graph:
            node_coordinates = node.cords
            all_nodes = [(n, n.cords) for n in self.graph if n != node]

            distances = cdist([node_coordinates], [n[1] for n in all_nodes])[0]
            nearest_indices = np.argsort(distances)[:k]
            for index in nearest_indices:
                neighbor, _ = all_nodes[index]
                edge_len = np.linalg.norm(node_coordinates - np.array(neighbor.cords))
                node.edges[neighbor] = Edge(neighbor, start_ph=1.0, l=edge_len)

        # check for lost edges and add them
        for node in self.graph:
            for adj_node, edge in node.edges.items():
                if adj_node.edges.get(node) is None:
                    adj_node.edges[node] = Edge(node, start_ph=edge.pheromone, l=edge.len)

    @staticmethod
    def distance(node1, node2):
        return cdist([node1.cords], [node2.cords])[0][0]

    def setPH(self, start_ph):
        for node in self.graph:
            for _, edge in node.edges.items():
                edge.pheromone = start_ph

    def visualize_graph(self, i=0):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        lines = []
        colors = []
        for node in self.graph:
            x_start, y_start, z_start = node.cords[0], node.cords[1], node.cords[2]
            for _, edge in node.edges.items():
                x_end, y_end, z_end = edge.adjacent_node.cords[0], edge.adjacent_node.cords[1], \
                                      edge.adjacent_node.cords[2]
                colors.append((1, 0, 0, edge.pheromone))
                lines.append([(x_start, y_start, z_start), (x_end, y_end, z_end)])

        # Create the Line3DCollection
        col = Line3DCollection(lines, edgecolors=colors)
        ax.add_collection(col)

        xs = [n.cords[0] for n in self.graph]
        ys = [n.cords[1] for n in self.graph]
        zs = [n.cords[2] for n in self.graph]
        ax.scatter(xs, ys, zs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # plt.savefig(f"C:/Users/USER/Desktop/cd1/graph{i}.png")
        plt.show()

    def visualize_best_path(self, best_path):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        lines = []
        for i in range(len(best_path)):
            x_start, y_start, z_start = best_path[i].cords
            x_end, y_end, z_end = best_path[(i + 1) % len(best_path)].cords
            lines.append([(x_start, y_start, z_start), (x_end, y_end, z_end)])

        # Create the Line3DCollection
        col = Line3DCollection(lines, edgecolors='g')
        ax.add_collection(col)

        xs = [n.cords[0] for n in self.graph]
        ys = [n.cords[1] for n in self.graph]
        zs = [n.cords[2] for n in self.graph]
        ax.scatter(xs, ys, zs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def visualize_best_path_2d(self, best_path):
        fig = plt.figure()
        ax = fig.add_subplot()

        lines = []
        for i in range(len(best_path)):
            x_start, y_start = best_path[i].cords
            x_end, y_end = best_path[(i + 1) % len(best_path)].cords
            lines.append([(x_start, y_start), (x_end, y_end)])

        # Create the Line3DCollection
        from matplotlib.collections import LineCollection
        col = LineCollection(lines, edgecolors='g')
        ax.add_collection(col)

        xs = [n.cords[0] for n in self.graph]
        ys = [n.cords[1] for n in self.graph]
        ax.scatter(xs, ys)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

    def lenRandomPath(self):
        lence = 0
        temp = list(self.graph)
        random.shuffle(temp)
        for x in range(len(temp) - 1):
            lence += Graph.distance(temp[x], temp[x + 1])
        lence += Graph.distance(temp[0], temp[-1])
        return lence

    def stepACO(self, ant_count, A, B, Q, evap):
        node_count = len(self.graph)
        better_path = []
        better_path_len = float("inf")
        for _ in range(ant_count):
            path = [random.choice(self.graph)]

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

        # функция на С должна возвращать лучший путь, его длинну
        # принимает ant_count, 2 таблицы -- феромоны и расстояния (или же продолжить ебаться с ооп), А, В

        # костыль(.. ну а что поделать
        if len(better_path) == 0:
            # print("no path")
            return float('inf'), None

        # Evaporation
        for node in self.graph:
            for edge in node.edges.values():
                edge.pheromone = (1 - evap) * edge.pheromone

        # add ph by best_path
        ph = Q / better_path_len
        for j in range(len(better_path)):
            self.graph[better_path[j].index].edges[better_path[(j + 1) % len(better_path)]].pheromone += ph
            self.graph[better_path[(j + 1) % len(better_path)].index].edges[better_path[j]].pheromone += ph

        return better_path_len, better_path

    def stepElite(self, best_path, best_path_len, Q, ant_count=1):
        ph = (Q / best_path_len) * ant_count
        for j in range(len(best_path)):
            self.graph[best_path[j].index].edges[best_path[(j + 1) % len(best_path)]].pheromone += ph
            self.graph[best_path[(j + 1) % len(best_path)].index].edges[best_path[j]].pheromone += ph

    def ACO(self, try_count, ant_count, A, B, Q, evap, start_ph):
        self.setPH(start_ph)
        best_path = []
        best_path_len = float('inf')
        for _ in range(try_count):
            bpl, bp = self.stepACO(ant_count=ant_count, A=A, B=B, Q=Q, evap=evap)
            if bpl < best_path_len:
                best_path_len = bpl
                best_path = bp
            if bp is not None:
                self.stepElite(bp, bpl, Q)

        return best_path_len, best_path

    def ACOperfomance(self, ant_count, A, B, Q, evap, start_ph, worktime):
        performance = 0
        self.setPH(start_ph)
        # best_path_len = self.lenRandomPath()
        best_path_len = float("inf")
        starttime = time.time()
        lasttime = time.time()
        while time.time() - starttime < worktime:
            bpl, _ = self.stepACO(ant_count=ant_count, A=A, B=B, Q=Q, evap=evap)
            if best_path_len > bpl:
                performance += (time.time() - lasttime) * bpl
                lasttime = time.time()
                best_path_len = bpl

        return performance


if __name__ == "__main__":
    print("Okay, let's go")
    graph = Graph()
    graph.load_from_file("3d300-1.txt")
    graph.add_k_nearest_edges(200)


    print([graph.ACOperfomance(260, 1, 2, 9420, 0.35, 0.3, 60) for _ in range(5)])

    # generate files with graph
    """for d in range(2, 11):
        for i in range(1, 5):
            graph = Graph()
            graph.generate_random_graph(300, d)
            graph.upload(f"{d}d300-{i}.txt")"""

    # testing Q param
    """for d in range(2, 11, 2):
        for i in range(1, 5):
            graph = Graph()
            graph.load_from_file(f"{d}d300-{i}.txt")
            graph.add_k_nearest_edges(120)
            for k in range(100, 20_000, 1_000):
                res = []
                for _ in range(10):
                    res.append(graph.ACOperfomance(ant_count=300, A=1, B=2, Q=k, evap=0.2, start_ph=0.2, worktime=60))
                print(f"{d}d300-{i} {k} {res}")"""

    # testing K nearist
    """for d, q in zip([4, 5, 7, 9], [5150, 6150, 7250, 8350]):
        for i in range(1, 5):
            for k in range(95, 14, -10):
                res = []
                for _ in range(10):
                    graph = Graph()
                    graph.load_from_file(f"{d}d100-{i}.txt")
                    graph.add_k_nearest_edges(k)
                    res.append(graph.ACOperfomance(ant_count=100, A=1, B=2, Q=q, evap=0.2, start_ph=0.2, worktime=60))
                print(f"{d}d100-{i} {k} {res}")"""

    """for i in range(1, 5):
        graph = Graph()
        graph.load_from_file(f"10d300-{i}.txt")
        graph.add_k_nearest_edges(120)
        for k in range(10_000, 100_001, 10_000):
            res = []
            for _ in range(10):
                res.append(graph.ACOperfomance(ant_count=100, A=1, B=2, Q=k, evap=0.2, start_ph=0.2, worktime=60))
            print(f"10d300-{i} {k} {res}")



    for i in range(1, 5):
        graph = Graph()
        graph.load_from_file(f"9d300-{i}.txt")
        graph.add_k_nearest_edges(120)
        for k in range(10_000, 100_001, 10_000):
            res = []
            for _ in range(10):
                res.append(graph.ACOperfomance(ant_count=100, A=1, B=2, Q=k, evap=0.2, start_ph=0.2, worktime=60))
            print(f"9d300-{i} {k} {res}")
    """
