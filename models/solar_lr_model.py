from sklearn.linear_model import Ridge
import numpy as np

class SklearnLRWrapper:
     def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.feature_names = None
        self.coef_ = None

     def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.model.fit(X.numpy(), y.numpy().flatten())
        self.coef_ = self.model.coef_

     def predict(self, X):
        return self.model.predict(X.numpy()).reshape(-1, 1)

     def get_coefficients(self):
        if self.coef_ is None:
            return None
        return sorted(zip(self.feature_names, self.coef_), key=lambda x: abs(x[1]), reverse=True)