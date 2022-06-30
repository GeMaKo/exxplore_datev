from lib2to3.pgen2.pgen import DFAState
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles

import streamlit as st


def get_isolated_data(noise: float):
    X, y = make_blobs(n_samples=100, n_features=2, centers=np.array([[-2, 2], [2, -2]]), cluster_std=np.array([noise*1.5, noise*1.3]))
    return X, y


def get_xor_data(noise: float):
    X1, _ = make_blobs(n_samples=50, n_features=2, centers=np.array([[-2, 2], [2, -2]]), cluster_std=np.array([noise*1.5, noise*1.3]))
    y1 = np.zeros(X1.shape[0])
    X2, _ = make_blobs(n_samples=50, n_features=2, centers=np.array([[-2, -2], [2, 2]]), cluster_std=np.array([noise*1.4, noise*1.2]))
    y2 = np.ones(X2.shape[0])
    return np.r_[X1, X2], np.r_[y1, y2]


def get_moon_data(noise: float):
    X, y = make_moons(n_samples=100, noise=noise/5)
    return X, y


def get_circle_data(noise: float):
    X, y = make_circles(n_samples=100, noise=noise/5, factor=0.1)
    return X, y


DATASETS = {
    "Isolated dataset": get_isolated_data,
    "XOR dataset": get_xor_data,
    "Moon dataset": get_moon_data,
    "Circle dataset": get_circle_data,
}

def create_figure(X, y, ax):
    ax.plot(X[y==0,0], X[y==0,1], marker="o", markersize=10, linestyle="None")
    ax.plot(X[y==1,0], X[y==1,1], marker="o", markersize=10, linestyle="None")




