from sklearn.datasets import make_circles, make_classification, make_moons

from . import util
from .perceptron_classifier import (
    CubicPerceptronClassifier,
    LinearPerceptronClassifier,
    QuadraticPerceptronClassifier,
)

LINEAR = 0
QUADRATIC = 1
CUBIC = 2

target = CUBIC

n_samples = 200
n_features = 2

if target == LINEAR:
    learning_rate = 0.001
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=15,
    )
    classifier = LinearPerceptronClassifier(learning_rate)
    classifier.train(X, y, 100)
elif target == QUADRATIC:
    learning_rate = 0.001
    X, y = make_circles(n_samples=n_samples, noise=0.1, random_state=42)
    classifier = QuadraticPerceptronClassifier(learning_rate)
    classifier.train(X, y, 100)
elif target == CUBIC:
    learning_rate = 0.002
    X, y = make_moons(n_samples=n_samples, noise=0.3)
    classifier = CubicPerceptronClassifier(learning_rate)
    classifier.train(X, y, 100)

util.plot_decision_boundary(classifier, X, y)
