import numpy as np
import pandas as pd
from models.population import Population

def genetic_algorithm(X, y,
                      population_size=100,
                      generations=2000,
                      elite_size=2,
                      save_path="best_weights.csv"):
    """
    Main Genetic Algorithm function to optimize linear model weights.

    Parameters
    ----------
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training target.
    population_size : int
        The number of individuals in the population.
    generations : int
        The total number of evolution iterations.
    elite_size : int
        Number of top individuals to preserve each generation.
    save_path : str
        File path to save the best weights.

    Returns
    -------
    best_weights : np.ndarray
        The best weight vector found after 'generations' evolution.
    best_fitness_per_gen : list
        A list of the best fitness scores over generations.
    """
    pop = Population(size=population_size,
                     num_features=X.shape[1],
                     X=X, y=y,
                     elite_size=elite_size)

    best_fitness_per_gen = []

    for gen in range(1, generations + 1):
        pop.evolve(generation=gen, max_generations=generations)
        scores = pop.evaluate()
        best_score = scores.max()
        best_fitness_per_gen.append(best_score)

        # Print intermediate results every 100 generations (optional)
        if gen % 100 == 0:
            print(f"Generation {gen}/{generations} - Best Fitness: {best_score:.6f}")

    # After finishing all generations, find the best individual
    final_scores = pop.evaluate()
    best_idx = np.argmax(final_scores)
    best_weights = pop.individuals[best_idx]

    # Save the best weights to CSV
    weights_df = pd.DataFrame(best_weights, columns=["Weight"])
    weights_df.index.name = "Feature_Index"
    weights_df.to_csv(save_path)
    print(f"[DONE] Best weights saved to {save_path}")

    return best_weights, best_fitness_per_gen
