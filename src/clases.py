import numpy as np
from sklearn.metrics import r2_score

class MediaPredictor:
    def fit(self, X, y):
        media = np.mean(y)
        self.media = media

    def predict(self, X):
        return np.full(X.shape[0], self.media)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

class T2Predictor:
    def __init__(self, column_names, column="T2"):
        """
        column_names: lista de nombres de las columnas en X (orden original)
        column: nombre de la columna que se quiere usar (por defecto T2)
        """
        if column not in column_names:
            raise ValueError(f"La columna '{column}' no se encuentra en el array de columnas.")
        self.interaction_index = column_names.index(column)

    def fit(self, X, y):
        x_vals = X[:, self.interaction_index].reshape(-1, 1)
        X_aug = np.hstack([x_vals, np.ones_like(x_vals)])
        beta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
        self.coef_ = beta[0]
        self.intercept_ = beta[1]

    def predict(self, X):
        x_vals = X[:, self.interaction_index]
        return self.coef_ * x_vals + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))
