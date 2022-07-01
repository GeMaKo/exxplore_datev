from lib2to3.pgen2.pgen import DFAState

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split


@st.cache
def get_isolated_data(noise: float) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=100, n_features=2, centers=np.array([[-2, 2], [2, -2]]), cluster_std=np.array([noise*5, noise*1.3]))
    return X, y

@st.cache
def get_xor_data(noise: float) -> tuple[np.ndarray, np.ndarray]:
    X1, _ = make_blobs(n_samples=50, n_features=2, centers=np.array([[-2, 2], [2, -2]]), cluster_std=np.array([noise*5, noise*1.3]))
    y1 = np.zeros(X1.shape[0])
    X2, _ = make_blobs(n_samples=50, n_features=2, centers=np.array([[-2, -2], [2, 2]]), cluster_std=np.array([noise*5, noise*1.2]))
    y2 = np.ones(X2.shape[0])
    return np.r_[X1, X2], np.r_[y1, y2]

@st.cache
def get_moon_data(noise: float) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=100, noise=noise/1.5)
    return X, y

@st.cache
def get_circle_data(noise: float) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_circles(n_samples=100, noise=noise/1.5, factor=0.1)
    return X, y

@st.cache
def get_train_test_data(X: np.ndarray, y: np.ndarray, data_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=data_ratio)
    return X_train, X_test, y_train, y_test


DATASETS = {
    "Isolated dataset": get_isolated_data,
    "XOR dataset": get_xor_data,
    "Moon dataset": get_moon_data,
    "Circle dataset": get_circle_data,
}
