import numpy as np
import random
from models.fitness_function import fitness_function

class Population:
    def __init__(self, size, num_features, X, y, elite_size=2):
        self.size = size
        self.X = X
        self.y = y
        self.elite_size = elite_size
        # 
        self.individuals = [np.random.uniform(-1, 1, num_features) for _ in range(size)]

    def evaluate(self):
        scores = [fitness_function(ind, self.X, self.y) for ind in self.individuals]
        return np.array(scores)

    def select_parents(self, scores):
        total_score = np.sum(scores)
        if total_score == 0 or np.isnan(total_score):
            probs = np.ones(len(scores)) / len(scores)
        else:
            probs = scores / total_score
        parents = random.choices(self.individuals, weights=probs, k=2)
        return parents

    def crossover(self, parent1, parent2):
        alpha = np.random.uniform(0, 1)
        child = alpha * parent1 + (1 - alpha) * parent2
        return child

    def mutate(self, individual, mutation_rate):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] += np.random.normal(0, 10.0)  # 
        return individual

    def evolve(self, generation, max_generations):
        scores = self.evaluate()
        elite_indices = scores.argsort()[::-1][:self.elite_size]
        elites = [self.individuals[i].copy() for i in elite_indices]

        progress_ratio = generation / max_generations
        mutation_rate = 0.4 * (1 - progress_ratio) + 0.01

        new_generation = []
        new_generation.extend(elites)

        while len(new_generation) < self.size:
            p1, p2 = self.select_parents(scores)
            child = self.crossover(p1, p2)
            child = self.mutate(child, mutation_rate=mutation_rate)
            new_generation.append(child)

        self.individuals = new_generation
