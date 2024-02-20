import os
import random
import time
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from Graph import Graph

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')

_dll = ctypes.CDLL('D:/projects/testsyka/x64/Debug/testsyka.dll')

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
        dmpp = (self.graph.distance_matrix.__array_interface__['data'][0] + np.arange(
            self.graph.distance_matrix.shape[0]) * self.graph.distance_matrix.strides[0]).astype(np.uintp)
        pmpp = (self.graph.pheromone_matrix.__array_interface__['data'][0] + np.arange(
            self.graph.pheromone_matrix.shape[0]) * self.graph.pheromone_matrix.strides[0]).astype(np.uintp)
        node_count = ctypes.c_size_t(self.graph.pheromone_matrix.shape[0])
        ant_count = ctypes.c_size_t(ant_count)
        A = ctypes.c_double(A)
        B = ctypes.c_double(B)
        Q = ctypes.c_double(Q)
        E = ctypes.c_double(E)
        bpl = ctypes.c_double()
        result_ptr = _step(dmpp, pmpp, node_count, ant_count, A, B, Q, E, ctypes.byref(bpl))

        # Преобразование указателя в массив
        better_path = np.ctypeslib.as_array(result_ptr, shape=(1, node_count.value))[0]
        return bpl.value, better_path

    def run_performance(self, ant_count, A, B, Q, evap, start_ph, worktime):
        performance = 0
        self.graph.setPH(ph=start_ph)
        best_path_len = self.graph.lenRandomPath()
        # best_path_len = float("inf")
        startTime = time.time()
        lastTime = startTime
        while time.time() - startTime < worktime:
            bpl, _ = self.step(ant_count=ant_count, A=A, B=B, Q=Q, E=evap)
            if best_path_len > bpl:
                performance += (time.time() - lastTime) * bpl
                lastTime = time.time()
                best_path_len = bpl

        return performance

    def run_print(self, ant_count, A, B, Q, evap, start_ph, worktime):
        print("let's go")
        # TODO: написать отображение графика эффективности
        self.graph.setPH(start_ph)
        # best_path_len = self.graph.lenRandomPath()
        best_path_len = float("inf")
        startTime = time.time()
        lastTime = startTime
        while time.time() - startTime < worktime:
            bpl, _ = self.step(ant_count=ant_count, A=A, B=B, Q=Q, E=evap)
            if best_path_len > bpl:
                print(f"{time.time() - startTime} {bpl}")
                lastTime = time.time()
                best_path_len = bpl

    def run(self, ant_count, A, B, Q, evap, start_ph, worktime):
        print("let's go")
        # TODO: написать отображение графика эффективности
        self.graph.setPH(start_ph)
        best_path = []
        # best_path_len = self.graph.lenRandomPath()
        best_path_len = float("inf")
        startTime = time.time()
        lastTime = startTime
        while time.time() - startTime < worktime:
            bpl, bp = self.step(ant_count=ant_count, A=A, B=B, Q=Q, E=evap)
            if best_path_len > bpl:
                lastTime = time.time()
                best_path_len = bpl
                best_path = bp

        print(best_path_len)
        return best_path


if __name__ == "__main__":
    _init_rand(random.randint(0, 4294967295))
    graph = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'benchmarks', '2d300.txt')
    graph.load(file_path, ph=0.5)
    graph.add_k_nearest(100)

    aco = ACO(graph)
    """start = time.time()
    print(aco.step(20, 1, 3, 100, 0.4))
    finish = time.time()
    print(finish - start)"""
    """for _ in range(10):
        print(aco.run_performance(20, 1, 3, 100, 0.4, 0.4, 20))"""
    """bp = aco.run(300, 1, 3, 12500, 0.3, 1, 180)
    graph.visualize_best_path_2d(bp)"""
    aco.run_print(300, 1, 3, 12_500, 0.3, 1, 5*60)
