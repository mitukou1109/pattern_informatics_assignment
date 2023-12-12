import numpy as np


class Classifier2D:
    def train(self, patterns: np.ndarray, labels: np.ndarray, epochs: int) -> None:
        raise NotImplementedError

    def predict(self, patterns: np.ndarray) -> np.ndarray:
        raise NotImplementedError
