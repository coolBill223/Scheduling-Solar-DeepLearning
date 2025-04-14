from sklearn.linear_model import LinearRegression

class SklearnLRWrapper:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X.numpy(), y.numpy().flatten())

    def predict(self, X):
        return self.model.predict(X.numpy()).reshape(-1, 1)
