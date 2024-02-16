import os
import time
import random
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')

_dll = ctypes.CDLL('D:/projects/testsyka/x64/Debug/testsyka.dll')

_step = _dll.step
_step.argtypes = [_doublepp, _doublepp, ctypes.c_size_t, ctypes.c_size_t,
                  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]

_step.restype = ctypes.POINTER(ctypes.c_size_t)


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

    def step(self, ant_count, A, B, Q, E):

        dmpp = (self.graph.distance_matrix.__array_interface__['data'][0] +
                np.arange(self.graph.distance_matrix.shape[0]) * self.graph.distance_matrix.strides[0]).astype(np.uintp)
        pmpp = (self.graph.pheromone_matrix.__array_interface__['data'][0] +
                np.arange(self.graph.pheromone_matrix.shape[0]) * self.graph.pheromone_matrix.strides[0]).astype(np.uintp)
        node_count = ctypes.c_size_t(self.graph.pheromone_matrix.shape[0])
        ant_count = ctypes.c_size_t(ant_count)
        A = ctypes.c_double(A)
        B = ctypes.c_double(B)
        Q = ctypes.c_double(Q)
        E = ctypes.c_double(E)
        result_ptr = _step(dmpp, pmpp, node_count, ant_count, A, B, Q, E)

        # Преобразование указателя в массив
        better_path = np.ctypeslib.as_array(result_ptr, shape=(1, node_count.value))[0]
        #return better_path_len, better_path
        return 0, better_path

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
    aco.step(1, 1, 3, 100, 0.4)
    aco.step(1, 1, 3, 100, 0.4)
    aco.step(1, 1, 3, 100, 0.4)
    aco.step(1, 1, 3, 100, 0.4)
    aco.step(1, 1, 3, 100, 0.4)
    aco.step(1, 1, 3, 100, 0.4)
    aco.step(1, 1, 3, 100, 0.4)
    aco.step(1, 1, 3, 100, 0.4)
    aco.step(1, 1, 3, 100, 0.4)
    aco.step(1, 1, 3, 100, 0.4)


    finish = time.time()
    print(finish - start)
    """for _ in range(10):
        print(aco.run_performance(20, 1, 3, 100, 0.4, 0.4, 20))"""
