import matplotlib.pyplot as plt
import numpy as np

from .classifier_2d import Classifier2D


def plot_decision_boundary(
    model: Classifier2D,
    patterns: np.ndarray,
    labels: np.ndarray,
    offset: int = 1,
    width: float = 0.01,
):
    min1, max1 = patterns[:, 0].min() - offset, patterns[:, 0].max() + offset
    min2, max2 = patterns[:, 1].min() - offset, patterns[:, 1].max() + offset

    x1grid = np.arange(min1, max1, width)
    x2grid = np.arange(min2, max2, width)

    xx, yy = np.meshgrid(x1grid, x2grid)

    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    grid = np.hstack((r1, r2))

    y_pred = model.predict(grid)

    zz = y_pred.reshape(xx.shape)

    plt.contourf(xx, yy, zz, cmap="Paired")
    plt.scatter(patterns[:, 0], patterns[:, 1], c=labels, cmap="cividis")
    plt.xlabel("x_1", fontsize=16)
    plt.ylabel("x_2", fontsize=16)
    plt.show()


def plot_line(x: np.ndarray, y: np.ndarray, xlabel: str = "", ylabel: str = ""):
    plt.plot(x, y)
    plt.grid(axis="x")
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()
