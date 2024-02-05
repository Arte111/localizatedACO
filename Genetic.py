import random
import statistics

from Graph import Graph

if __name__ == "__main__":
    aco_worktime = 30
    graph = Graph()
    graph.load("2d100.txt")
    graph.add_k_nearest_edges(60)
    parameters = {
        "ant_count": range(20, 150, 10),
        "A": [0.5, 1, 1.5, 2, 3, 4],
        "B": [0.5, 1, 1.5, 2, 3, 4],
        "Q": range(100, 5000, 200),
        "evap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "start_ph": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    population_size = 40
    population = []

    for i in range(population_size):
        individual = {}
        for key in parameters:
            individual[key] = random.choice(parameters[key])
        population.append(individual)

    for l in range(9):
        print(f"let's go {l}")
        # оценка каждой особи 5 раз
        for individual in population:
            all_perfomance = [graph.ACOperfomance(individual["ant_count"], individual["A"], individual["B"],
                                                  individual["Q"], individual["evap"], individual["start_ph"],
                                                  aco_worktime) for _ in range(5)]
            individual["performance"] = statistics.mean(all_perfomance)

        # selection
        elite_size = int(population_size * 0.25)
        elite = sorted(population, key=lambda x: x["performance"])[:elite_size]

        # debug print
        for i in elite:
            print(f"{i['ant_count']} {i['A']} {i['B']} {i['Q']} {i['evap']} {i['start_ph']} ")


        def crossover(parent1, parent2):
            child = {
                "ant_count": random.choice([parent1["ant_count"], parent2["ant_count"]]),
                "A": random.choice([parent1["A"], parent2["A"]]),
                "B": random.choice([parent1["B"], parent2["B"]]),
                "Q": random.choice([parent1["Q"], parent2["Q"]]),
                "evap": random.choice([parent1["evap"], parent2["evap"]]),
                "start_ph": random.choice([parent1["start_ph"], parent2["start_ph"]]),
            }
            return child


        def mutate(child):
            child["ant_count"] += random.randint(-5, 5)
            child["A"] += random.uniform(-0.3, 0.3)
            child["B"] += random.uniform(-0.3, 0.3)
            child["Q"] += random.randint(-100, 100)
            child["evap"] += random.uniform(-0.3, 0.3)
            child["start_ph"] += random.uniform(-0.3, 0.3)
            for key in child:
                child[key] = abs(child[key])
            return child


        # crossing and mutate
        offspring = []
        for i in range(population_size - elite_size):
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = crossover(parent1, parent2)
            child = mutate(child)
            offspring.append(child)

        population = elite + offspring
