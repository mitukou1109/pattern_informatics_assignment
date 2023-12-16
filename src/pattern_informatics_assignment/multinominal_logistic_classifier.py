import numpy as np

from .confusion_matrix import ConfusionMatrix


class MultinominalLogisticClassifier:
    def __init__(self, n_features: int, n_classes: int, learning_rate: float) -> None:
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weights: np.ndarray = np.random.rand(n_features, n_classes)
        self.confusion_matrix = ConfusionMatrix(n_classes)

    def train(self, patterns: np.ndarray, labels: np.ndarray, epochs: int) -> None:
        teacher = np.zeros((labels.shape[0], self.n_classes))
        teacher[np.arange(labels.shape[0]), labels] = 1
        print("Train")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            pred = self.predict(patterns)
            self.weights -= self.learning_rate * patterns.T @ (pred - teacher)
            self.confusion_matrix.evaluate(np.argmax(pred, axis=1), labels)
            print(f"Accuracy: {self.confusion_matrix.accuracy}")

    def test(self, patterns: np.ndarray, labels: np.ndarray) -> float:
        print("Test")
        self.confusion_matrix.evaluate(
            np.argmax(self.predict(patterns), axis=1), labels
        )
        with np.printoptions(precision=3):
            print(f"Recall: {self.confusion_matrix.recall}")
            print(f"Precision: {self.confusion_matrix.precision}")
            print(f"F1: {self.confusion_matrix.f1}")
            print(f"Accuracy: {self.confusion_matrix.accuracy:.3f}")
        return self.confusion_matrix.accuracy

    def predict(self, patterns: np.ndarray) -> np.ndarray:
        return np.exp(self.weights.T @ patterns.T).T / np.sum(
            np.exp(self.weights.T @ patterns.T).T, axis=1, keepdims=True
        )
