
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .mnist_interface import MnistClassifierInterface

class RandomForestMnistClassifier(MnistClassifierInterface):
    """RandomForest on flattened MNIST pixels."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42, n_jobs: int = -1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs
        )

    def train(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> None:
        X_train = self._ensure_np(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X: Any) -> Any:
        X = self._ensure_np(X)
        return self.model.predict(X)

    @staticmethod
    def _ensure_np(X):
        if isinstance(X, np.ndarray):
            return X
        # torch tensors -> numpy
        try:
            import torch
            if isinstance(X, torch.Tensor):
                return X.detach().cpu().numpy()
        except Exception:
            pass
        # fallback: attempt to convert
        return np.asarray(X)
