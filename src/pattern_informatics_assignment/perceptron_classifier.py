import numpy as np

from .classifier_2d import Classifier2D


class PerceptronClassifier(Classifier2D):
    def __init__(
        self, n_features: int, learning_rate: float, initial_weights: np.ndarray = None
    ) -> None:
        self.learning_rate = learning_rate
        if initial_weights is not None:
            assert initial_weights.shape == (n_features,)
            self.weights = initial_weights
        else:
            self.weights: np.ndarray = np.random.rand(n_features)

    def make_features(self, patterns: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def train(self, patterns: np.ndarray, labels: np.ndarray, epochs: int) -> None:
        print("Train")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            correct = 0
            for pattern, label in zip(patterns, labels):
                pred = self.predict(pattern)[0][0]
                error = pred - label
                correct += pred == label
                self.weights -= (
                    self.learning_rate * error * self.make_features(pattern).flatten()
                )
            print(f"Accuracy: {correct / patterns.shape[0]}")
            if correct == patterns.shape[0]:
                break

    def predict(self, patterns: np.ndarray) -> np.ndarray:
        features = self.make_features(patterns)
        return np.where(
            features @ self.weights.reshape(-1, 1) > 0.5,
            1,
            0,
        )


class LinearPerceptronClassifier(PerceptronClassifier):
    def __init__(
        self, learning_rate: float, initial_weights: np.ndarray = None
    ) -> None:
        super().__init__(3, learning_rate, initial_weights)

    def make_features(self, patterns: np.ndarray) -> np.ndarray:
        patterns = patterns.reshape(-1, 2)
        return np.hstack((np.ones((patterns.shape[0], 1)), patterns))


class QuadraticPerceptronClassifier(PerceptronClassifier):
    def __init__(
        self, learning_rate: float, initial_weights: np.ndarray = None
    ) -> None:
        super().__init__(6, learning_rate, initial_weights)

    def make_features(self, patterns: np.ndarray) -> np.ndarray:
        patterns = patterns.reshape(-1, 2)
        return np.hstack(
            (
                np.ones((patterns.shape[0], 1)),
                patterns,
                patterns[:, 0:1] * patterns[:, 1:2],
                patterns**2,
            )
        )


class CubicPerceptronClassifier(PerceptronClassifier):
    def __init__(
        self, learning_rate: float, initial_weights: np.ndarray = None
    ) -> None:
        super().__init__(10, learning_rate, initial_weights)

    def make_features(self, patterns: np.ndarray) -> np.ndarray:
        patterns = patterns.reshape(-1, 2)
        return np.hstack(
            (
                np.ones((patterns.shape[0], 1)),
                patterns,
                patterns[:, 0:1] * patterns[:, 1:2],
                patterns**2,
                patterns[:, 0:1] ** 2 * patterns[:, 1:2],
                patterns[:, 0:1] * patterns[:, 1:2] ** 2,
                patterns**3,
            )
        )
