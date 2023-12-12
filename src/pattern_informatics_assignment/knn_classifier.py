import numpy as np

from .classifier_2d import Classifier2D
from .confusion_matrix import ConfusionMatrix


class KNNClassifier(Classifier2D):
    def __init__(self, k: int, n_classes: int):
        self.k = k
        self.patterns = None
        self.labels = None
        self.confusion_matrix = ConfusionMatrix(n_classes)

    def cross_validation(self, patterns: np.ndarray, labels: np.ndarray, n_groups: int):
        n_data_per_group = patterns.shape[0] // n_groups
        accuracy = 0
        for i in range(n_groups):
            test_indices = np.arange(i * n_data_per_group, (i + 1) * n_data_per_group)
            train_indices = np.delete(np.arange(patterns.shape[0]), test_indices)
            self.train(patterns[train_indices], labels[train_indices])
            self.confusion_matrix.evaluate(
                self.predict(patterns[test_indices]), labels[test_indices]
            )
            accuracy += self.confusion_matrix.accuracy
        return accuracy / n_groups

    def train(
        self, patterns: np.ndarray, labels: np.ndarray, epochs: int = None
    ) -> None:
        self.patterns = patterns
        self.labels = labels

    def test(self, patterns: np.ndarray, labels: np.ndarray) -> None:
        self.confusion_matrix.evaluate(self.predict(patterns), labels)
        with np.printoptions(precision=3):
            print(f"Recall: {self.confusion_matrix.recall}")
            print(f"Precision: {self.confusion_matrix.precision}")
            print(f"F1: {self.confusion_matrix.f1}")
            print(f"Accuracy: {self.confusion_matrix.accuracy:.3f}")

    def predict(self, patterns: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.patterns[:, np.newaxis, :] - patterns, axis=2)
        # np.argpartition is more efficient than np.argsort
        k_nearest_indices = np.argpartition(distances, self.k, axis=0)[: self.k]
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=self.labels[k_nearest_indices],
        )
