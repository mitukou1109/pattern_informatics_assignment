import sys

import numpy as np
from sklearn.datasets import load_digits, make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split

from . import util
from .knn_classifier import KNNClassifier
from .multinominal_logistic_classifier import MultinominalLogisticClassifier
from .perceptron_classifier import (
    CubicPerceptronClassifier,
    LinearPerceptronClassifier,
    QuadraticPerceptronClassifier,
)

PERCEPTRON_LINEAR = 0
PERCEPTRON_QUADRATIC = 1
PERCEPTRON_CUBIC = 2
MULTINOMINAL_LOGISTIC = 3
KNN_CLASSIFICATION = 4
KNN_CIRCLES = 5
KNN_MOONS = 6
KNN_MULTICLASS = 7

np.random.seed(13)

if len(sys.argv) > 1:
    target = int(sys.argv[1])
else:
    target = int(
        input(
            "0: Perceptron - Linear, 1: Perceptron - Quadratic, 2: Perceptron - Cubic, 3: Multinominal Logistic, \
             \n4: KNN - make_classification, 5: KNN - make_circles, 6: KNN - make_moons 7: KNN - Multiclass >> "
        )
    )

if target <= PERCEPTRON_CUBIC:
    if target == PERCEPTRON_LINEAR:
        X, y = make_classification(
            n_samples=200,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=15,
        )
        classifier = LinearPerceptronClassifier(learning_rate=0.001)
        epochs = 100
    elif target == PERCEPTRON_QUADRATIC:
        X, y = make_circles(n_samples=200, noise=0.1, random_state=42)
        classifier = QuadraticPerceptronClassifier(learning_rate=0.001)
        epochs = 100
    elif target == PERCEPTRON_CUBIC:
        X, y = make_moons(n_samples=200, noise=0.3)
        classifier = CubicPerceptronClassifier(learning_rate=0.2)
        epochs = 100
    classifier.train(X, y, epochs)
    util.plot_decision_boundary(classifier, X, y)

elif target == MULTINOMINAL_LOGISTIC:
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data / 16, digits.target, test_size=0.2, random_state=0
    )
    classifier = MultinominalLogisticClassifier(
        n_features=digits.data.shape[1],
        n_classes=len(digits.target_names),
        learning_rate=0.001,
    )
    classifier.train(X_train, y_train, 1000)
    classifier.test(X_test, y_test)

elif target <= KNN_MOONS:
    if target == KNN_CLASSIFICATION:
        X, y = make_classification(
            n_samples=200,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=15,
        )
    elif target == KNN_CIRCLES:
        X, y = make_circles(n_samples=200, noise=0.1, random_state=42)
    elif target == KNN_MOONS:
        X, y = make_moons(n_samples=200, noise=0.3)
    classifier = KNNClassifier(k=16, n_classes=2)
    classifier.train(X, y)
    util.plot_decision_boundary(classifier, X, y)

elif target == KNN_MULTICLASS:
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data / 16, digits.target, test_size=0.5, random_state=0
    )
    k_max = 32
    ks = np.arange(k_max) + 1
    accuracy = np.zeros(k_max)
    for k in ks:
        print(f"k = {k}")
        classifier = KNNClassifier(k=k, n_classes=len(digits.target_names))
        accuracy[k - 1] = classifier.cross_validation(X_train, y_train, 100)
        classifier.train(X_train, y_train)
        classifier.test(X_test, y_test)
    util.plot_line(ks, accuracy, xlabel="k", ylabel="accuracy")

else:
    raise ValueError("Invalid target")
