import numpy as np

def fitness_function(individual, X, y):
    predictions = X @ individual
    mse = np.mean((predictions - y) ** 2)
    return -mse