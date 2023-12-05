import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy import random
from scipy.spatial.distance import cdist
import matplotlib.animation as animation

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

                if len(coordinates) != 3:
                    print(f"Error in line {index + 1}: Expected 3 coordinates, found {len(coordinates)}")
                    continue

                x, y, z = map(float, coordinates)
                node = Node(index + 1, x, y, z)
                self.graph.append(node)

    def generate_random_3d_graph(self, node_count):
        self.graph.clear()
        for index in range(node_count):
            x, y, z = np.random.uniform(-10, 10, size=3)
            node = Node(index, x, y, z)
            self.graph.append(node)

    def add_k_nearest_edges(self, k):
        for node in self.graph:
            node_coordinates = np.array([node.x, node.y, node.z])
            all_nodes = [(n, np.array([n.x, n.y, n.z])) for n in self.graph if n != node]

            distances = cdist([node_coordinates], [n[1] for n in all_nodes])[0]
            nearest_indices = np.argsort(distances)[:k]
            for index in nearest_indices:
                neighbor, _ = all_nodes[index]
                edge_len = np.linalg.norm(node_coordinates - np.array([neighbor.x, neighbor.y, neighbor.z]))
                node.edges[neighbor] = Edge(neighbor, start_ph=1.0, l=edge_len)

        # check for lost edges and add them
        for node in self.graph:
            for adj_node, edge in node.edges.items():
                if adj_node.edges.get(node) is None:
                    adj_node.edges[node] = Edge(node, start_ph=edge.pheromone, l=edge.len)

    @staticmethod
    def distance(node1, node2):
        return cdist(node1, node2)

    def setPH(self, start_ph):
        for node in self.graph:
            for _, edge in node.edges.items():
                edge.pheromone = start_ph

    def visualize_graph(self, i):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        lines = []
        colors = []
        for node in self.graph:
            x_start, y_start, z_start = node.x, node.y, node.z
            for _, edge in node.edges.items():
                x_end, y_end, z_end = edge.adjacent_node.x, edge.adjacent_node.y, edge.adjacent_node.z
                colors.append((1, 0, 0, edge.pheromone))
                lines.append([(x_start, y_start, z_start), (x_end, y_end, z_end)])

        # Create the Line3DCollection
        col = Line3DCollection(lines, edgecolors=colors)
        ax.add_collection(col)

        xs = [node.x for node in self.graph]
        ys = [node.y for node in self.graph]
        zs = [node.z for node in self.graph]
        ax.scatter(xs, ys, zs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(f"C:/Users/USER/Desktop/cd1/graph{i}.png")
        # plt.show()

    def stepACO(self, i, ant_count, A, B, Q, evap):
        node_count = len(self.graph)
        best_path = []
        best_path_len = float("inf")
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
            for j in range(len(path) - 1):
                path_len += path[j].edges[path[j + 1]].len
            path_len += path[-1].edges[path[0]].len

            # Update best path
            if best_path_len > path_len:
                best_path = path
                best_path_len = path_len

        # костыль(.. ну а что поделать
        if len(best_path) == 0:
            print("no path")
            return

        # Evaporation
        for node in self.graph:
            for edge in node.edges.values():
                edge.pheromone = (1 - evap) * edge.pheromone

        # add ph by best_path
        ph = Q / best_path_len
        for j in range(len(best_path) - 1):
            self.graph[best_path[j].index].edges[best_path[j + 1]].pheromone += ph
            self.graph[best_path[j + 1].index].edges[best_path[j]].pheromone += ph
        self.graph[best_path[-1].index].edges[best_path[0]].pheromone += ph
        self.graph[best_path[0].index].edges[best_path[-1]].pheromone += ph

        self.visualize_graph(i)


if __name__ == "__main__":
    print("Okay, let's go")
    graph = Graph()
    graph.generate_random_3d_graph(10)
    graph.add_k_nearest_edges(9)
    graph.setPH(0.2)
    for i in range(30):
        graph.stepACO(i=i, ant_count=10, A=1, B=2, Q=350, evap=0.2)
    # graph.visualize_graph()
