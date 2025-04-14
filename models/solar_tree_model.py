from sklearn.ensemble import RandomForestRegressor
import numpy as np

class SklearnTreeWrapper:
    def __init__(self, n_estimators=100, max_depth=6, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.feature_importances_ = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.model.fit(X.numpy(), y.numpy().flatten())
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X):
        return self.model.predict(X.numpy()).reshape(-1, 1)

    def get_feature_importance(self):
        if self.feature_importances_ is None:
            return None
        return sorted(zip(self.feature_names, self.feature_importances_), key=lambda x: -x[1])