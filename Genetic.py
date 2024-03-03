import os
import random
import statistics

from ACO import ACO, Graph, _init_rand

if __name__ == "__main__":
    _init_rand(random.randint(0, 4294967295))
    aco_worktime = 2
    graph = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'benchmarks', '2d100.txt')
    graph.load(file_path)
    graph.add_k_nearest_edges(99)
    aco = ACO(graph)
    parameters = {
        "ant_count": range(100, 500, 10),
        "A": [0.25, 0.5, 1, 1.5, 2, 3, 4],
        "B": [3, 4, 5, 6, 7],
        "Q": range(500, 1000, 50),
        "evap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "start_ph": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    population_size = 50
    population = []

    for i in range(population_size):
        individual = {}
        for key in parameters:
            individual[key] = random.choice(parameters[key])
        population.append(individual)

    for l in range(20):
        print(f"let's go {l}")
        for individual in population:
            all_perfomance = [
                aco.run_performance(ant_count=int(individual["ant_count"]), A=individual["A"], B=individual["B"],
                                    Q=individual["Q"], evap=individual["evap"], start_ph=individual["start_ph"],
                                    worktime=aco_worktime) for _ in range(5)]
            individual["performance"] = statistics.mean(all_perfomance)

        b = min(population, key=lambda x: x["performance"])
        print(b)

        # selection
        temp = []
        for _ in population:
            temp.append(min(random.choices(population, k=4), key=lambda x: x["performance"]))
        population = temp.copy()


        def crossover(mommy, daddy):
            child = {}
            for j in ["ant_count", "A", "B", "Q", "evap", "start_ph"]:
                minj = min(mommy[j], daddy[j])
                maxj = max(mommy[j], daddy[j])
                dmin = minj - 0.5 * (maxj - minj)
                dmax = minj + 0.5 * (maxj - minj)
                child[j] = random.uniform(dmin, dmax)

            return child


        def mutate(child):
            for i in ["ant_count", "A", "B", "Q", "evap", "start_ph"]:
                if random.random() < 0.05:
                    child[i] = random.triangular(min(parameters[i]), max(parameters[i]),
                                                 random.gauss(child[i], child[i] * 0.2))

            for key in child:
                child[key] = abs(child[key])
            if child["evap"] > 0.99: child["evap"] = 0.99
            return child


        # crossing and mutate
        offspring = []

        for j in range(0, len(population), 2):
            if random.random() < 0.95:
                offspring.append(crossover(population[j], population[j + 1]))
                offspring.append(crossover(population[j], population[j + 1]))

        map(mutate, offspring)
        population = offspring.copy()
