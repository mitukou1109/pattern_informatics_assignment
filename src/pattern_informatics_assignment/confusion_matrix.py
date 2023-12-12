import numpy as np


class ConfusionMatrix:
    def __init__(self, n_classes: int) -> None:
        self.matrix = np.zeros((n_classes, n_classes))
        self.recall = 0.0
        self.precision = 0.0
        self.f1 = 0.0
        self.accuracy = 0.0

    def evaluate(self, pred: np.ndarray, labels: np.ndarray) -> None:
        assert pred.shape == labels.shape
        for i in range(pred.shape[0]):
            self.matrix[pred[i], labels[i]] += 1
        self.recall = np.diag(self.matrix) / np.sum(self.matrix, axis=0)
        self.precision = np.diag(self.matrix) / np.sum(self.matrix, axis=1)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        self.accuracy = np.sum(np.diag(self.matrix)) / np.sum(self.matrix)
