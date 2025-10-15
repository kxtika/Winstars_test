
from typing import Any, Literal
import torch
from .rf_model import RandomForestMnistClassifier
from .nn_model import FeedForwardMnistClassifier
from .cnn_model import ConvolutionalMnistClassifier
from .mnist_interface import MnistClassifierInterface

Algo = Literal['rf','nn','cnn']

class MnistClassifier:
    """Wrapper that hides model-specific details and exposes a common API."""
    def __init__(self, algorithm: Algo = 'cnn'):
        self.algorithm = algorithm
        self.model: MnistClassifierInterface
        if algorithm == 'rf':
            self.model = RandomForestMnistClassifier()
        elif algorithm == 'nn':
            self.model = FeedForwardMnistClassifier()
        elif algorithm == 'cnn':
            self.model = ConvolutionalMnistClassifier()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> None:
        self.model.train(X_train, y_train, X_val, y_val)

    def predict(self, X: Any):
        preds = self.model.predict(X)
        # Always return labels as a 1D tensor/array
        if isinstance(preds, torch.Tensor):
            return preds.cpu().view(-1)
        try:
            import numpy as np
            if isinstance(preds, np.ndarray):
                return preds.reshape(-1)
        except Exception:
            pass
        return preds
