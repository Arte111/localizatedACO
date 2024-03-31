import os
import random
import time

import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from Graph import Graph

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')

_dll = ctypes.CDLL('testsyka.dll')

_step = _dll.step
_step.argtypes = [_doublepp, _doublepp, ctypes.c_size_t, ctypes.c_size_t,
                  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]

_step.restype = ctypes.POINTER(ctypes.c_size_t)

_init_rand = _dll.init_rand
_init_rand.argtypes = [ctypes.c_size_t]
_init_rand.restype = None


class ACO:
    def __init__(self, graph):
        self.graph = graph

    def step(self, ant_count, A, B, Q, E):
        dmpp = (self.graph.closeness_matrix.__array_interface__['data'][0] + np.arange(
            self.graph.closeness_matrix.shape[0]) * self.graph.closeness_matrix.strides[0]).astype(np.uintp)
        pmpp = (self.graph.pheromone_matrix.__array_interface__['data'][0] + np.arange(
            self.graph.pheromone_matrix.shape[0]) * self.graph.pheromone_matrix.strides[0]).astype(np.uintp)
        node_count = ctypes.c_size_t(self.graph.pheromone_matrix.shape[0])
        ant_count = ctypes.c_size_t(ant_count)
        A = ctypes.c_double(A)
        B = ctypes.c_double(B)
        Q = ctypes.c_double(Q)
        E = ctypes.c_double(E)
        bpl = ctypes.c_double()
        try:
            result_ptr = _step(dmpp, pmpp, node_count, ant_count, A, B, Q, E, ctypes.byref(bpl))
        except:
            print("Error")
            return float("inf"), []
        # Преобразование указателя в массив
        better_path = np.ctypeslib.as_array(result_ptr, shape=(1, node_count.value))[0]
        return bpl.value, better_path

    def run_performance(self, ant_count, A, B, Q, evap, start_ph, worktime, fine):
        performances = []
        self.graph.setPH(ph=start_ph)
        # best_path_len = self.graph.lenRandomPath()
        # best_path_len = float("inf")
        best_path_len = fine
        startTime = time.time()
        while time.time() - startTime < worktime:
            bpl, _ = self.step(ant_count=ant_count, A=A, B=B, Q=Q, E=evap)
            if best_path_len > bpl:
                performances.append((time.time() - startTime, bpl))
                best_path_len = bpl

        performance = 0
        for i in range(1, len(performances)):
            performance += (performances[i][0] - performances[i - 1][0]) * performances[i][1]

        if performances[-1][0] > worktime:
            performance -= (performances[-1][0] - worktime) * performances[-1][1]
        else:
            performance += (worktime - performances[-1][0]) * performances[-1][1]

        performance += best_path_len * 3 * worktime

        return performance

    def run_print(self, ant_count, A, B, Q, evap, start_ph, worktime):
        print("let's go")
        # TODO: написать отображение графика эффективности
        self.graph.setPH(start_ph)
        # best_path_len = self.graph.lenRandomPath()
        best_path_len = float("inf")
        startTime = time.time()
        while time.time() - startTime < worktime:
            bpl, _ = self.step(ant_count=ant_count, A=A, B=B, Q=Q, E=evap)
            if best_path_len > bpl:
                print(f"{time.time() - startTime} {bpl}")
                best_path_len = bpl

    def run(self, ant_count, A, B, Q, evap, start_ph, worktime):
        self.graph.setPH(start_ph)
        best_path = []
        # best_path_len = self.graph.lenRandomPath()
        best_path_len = float("inf")
        startTime = time.time()
        while time.time() - startTime < worktime:
            bpl, bp = self.step(ant_count=ant_count, A=A, B=B, Q=Q, E=evap)
            if best_path_len > bpl:
                best_path_len = bpl
                best_path = bp
                print(f"{time.time() - startTime} {bpl}")

        print(best_path_len)
        return best_path


if __name__ == "__main__":
    _init_rand(random.randint(0, 4294967295))
    with open("logs.txt", "w") as file:
        for k in range(300, 50, -10):
            graph = Graph()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, 'benchmarks', f'6d300.txt')
            graph.load(file_path, ph=0.5)
            graph.add_k_nearest_edges(k)

            aco = ACO(graph)
            res = []
            for _ in range(10):
                res.append(aco.run_performance(300, 3, 9, 7200, 0.30, 0.50, 20, 48_000))

            print(f"{k} {res}")
            print(f"{k} {res}", file=file)

    """start = time.time()
    for _ in range(1):
        aco.step(100_000, 1, 2, 200, 0.2)
    finish = time.time()
    print(finish - start)"""
    # graph.visualize_best_path_2d(bp)
    """for _ in range(10):
        print(aco.run_performance(500, 3, 10, 650, 0.4, 0.75, 3))"""
    # graph.visualize_best_path_2d(bp)
    # bp = aco.run(100, 2.85, 8.39, 400, 0.20, 0.50, 60)
    # print(aco.run_performance(100, 2.85, 8.39, 400, 0.20, 0.50, 10))
    # 462, 3.69, 7.05, 7147, 0.23, 0.63, 3
    # 483, 0.95, 6.16, 4990, 0.54, 0.38, 3
    # 474, 2.69, 6.05, 738.1, 0.2, 0.47, 3
    # 469, 2.53, 7.27, 8045, 0.43, 0.71, 3
    # 100, 1.48, 8.18, 1078, 0.20, 0.50, 3
