import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy import random
from scipy.spatial.distance import cdist
import random

from Edge import Edge
from Node import Node


class Graph:
    graph: list[Node]

    def __init__(self):
        self.graph = []

    def load(self, file_path):
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

    def __len__(self):
        return len(self.graph)

    def evaporation(self, evaporation):
        for node in self.graph:
            for edge in node.edges.values():
                edge.pheromone *= (1 - evaporation)

    def randomNode(self):
        return np.random.choice(self.graph)


if __name__ == '__main__':
    g = Graph()
    g.generate_random_graph(300, 4)
    g.upload("temp.txt")
