
from abc import ABC, abstractmethod
from typing import Any

class MnistClassifierInterface(ABC):
    """Abstract interface that all MNIST classifiers must implement."""

    @abstractmethod
    def train(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> None:
        """Train the model in-place."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Return class logits or probabilities or labels, depending on implementation.
        This project wraps models with MnistClassifier to normalize outputs to labels.
        """
        raise NotImplementedError
