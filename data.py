import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles

mpl.style.use("default")

def get_isolated_data():
    X, y = make_blobs(n_samples=100, n_features=2, centers=np.array([[-2, 2], [2, -2]]), cluster_std=np.array([0.5, 0.8]))
    return X, y


def get_overlapping_data():
    X, y = make_blobs(n_samples=100, n_features=2, centers=np.array([[-2, 2], [2, -2]]), cluster_std=np.array([2.5, 2.0]))
    return X, y


def get_moon_data():
    X, y = make_moons(n_samples=100, noise=0.1)
    return X, y


def get_circle_data():
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.1)
    return X, y


def get_xor_data():
    X1, _ = make_blobs(n_samples=50, n_features=2, centers=np.array([[-2, 2], [2, -2]]), cluster_std=np.array([0.45, 0.55]))
    y1 = np.zeros(X1.shape[0])
    X2, _ = make_blobs(n_samples=50, n_features=2, centers=np.array([[-2, -2], [2, 2]]), cluster_std=np.array([0.65, 0.33]))
    y2 = np.ones(X2.shape[0])
    return np.r_[X1, X2], np.r_[y1, y2]


for f in [get_isolated_data, get_overlapping_data, get_moon_data, get_circle_data, get_xor_data]:
    X, y = f()
    plt.figure()
    plt.plot(X[y==0,0], X[y==0,1], marker="o", linestyle="None")
    plt.plot(X[y==1,0], X[y==1,1], marker="o", linestyle="None")
    plt.show()

