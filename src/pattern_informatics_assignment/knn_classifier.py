import numpy as np

from .classifier_2d import Classifier2D


class KNNClassifier(Classifier2D):
    def __init__(self, k: int):
        self.k = k
        self.patterns = None
        self.labels = None

    def train(
        self, patterns: np.ndarray, labels: np.ndarray, epochs: int = None
    ) -> None:
        self.patterns = patterns
        self.labels = labels

    def predict(self, patterns: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.patterns[:, np.newaxis, :] - patterns, axis=2)
        # np.argpartition is more efficient than np.argsort
        k_nearest_indices = np.argpartition(distances, self.k, axis=0)[: self.k]
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=self.labels[k_nearest_indices],
        )
