from sklearn.tree import DecisionTreeRegressor

class SklearnTreeWrapper:
    def __init__(self, **kwargs):
        self.model = DecisionTreeRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X.numpy(), y.numpy().flatten())

    def predict(self, X):
        import numpy as np
        return self.model.predict(X.numpy()).reshape(-1, 1)
