import numpy as np
import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split


@st.cache_data
def get_isolated_data(noise: float, balance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of isolated clusters
    """
    n_samples = [int(100 * balance), int(100 * (1 - balance))]
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=np.array([[-2, 2], [2, -2]]),
        cluster_std=np.array([noise * 5, noise * 5]),
    )
    return X, y


@st.cache_data
def get_xor_data(noise: float, balance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an XOR dataset
    """
    n_samples = [int(100 * balance), int(100 * (1 - balance))]
    X1, _ = make_blobs(
        n_samples=n_samples[0],
        n_features=2,
        centers=np.array([[-2, 2], [2, -2]]),
        cluster_std=np.array([noise * 5, noise * 5]),
    )
    y1 = np.zeros(X1.shape[0])
    X2, _ = make_blobs(
        n_samples=n_samples[1],
        n_features=2,
        centers=np.array([[-2, -2], [2, 2]]),
        cluster_std=np.array([noise * 5, noise * 5]),
    )
    y2 = np.ones(X2.shape[0])
    return np.r_[X1, X2], np.r_[y1, y2]


@st.cache_data
def get_moon_data(noise: float, balance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the moon dataset
    """
    n_samples = [int(100 * balance), int(100 * (1 - balance))]
    X, y = make_moons(n_samples=n_samples, noise=noise / 1.5)
    return X, y


@st.cache_data
def get_circle_data(noise: float, balance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the circle dataset
    """
    n_samples = [int(100 * balance), int(100 * (1 - balance))]
    X, y = make_circles(n_samples=n_samples, noise=noise / 1.5, factor=0.1)
    return X, y


@st.cache_data
def get_train_test_data(
    X: np.ndarray, y: np.ndarray, data_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform train / test split
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=data_ratio)
    return X_train, X_test, y_train, y_test


DATASETS = {
    "Isolated dataset": get_isolated_data,
    "XOR dataset": get_xor_data,
    "Moon dataset": get_moon_data,
    "Circle dataset": get_circle_data,
}


def get_scores_styler(styler: Styler, columns: list[str]):
    styler.set_caption("Performance measures based on CV on training data.")
    # styler.format(formatter={("fit_time", "score_time"): "{:.4f}", ("test_f1", "train_f1"): "{:.2f}"})
    for col in columns:
        styler.background_gradient(axis=None, subset=[col], cmap="Greens")
    return styler


def cv_scores2df(cv_scores: dict) -> Styler:
    df_cv_results = pd.DataFrame.from_dict(cv_scores)
    df_cv_results = df_cv_results.drop(columns=["fit_time", "score_time"])
    df_cv_results.index = [f"Fold {i}" for i in range(len(cv_scores["fit_time"]))]
    styler = df_cv_results.style.pipe(get_scores_styler, df_cv_results.columns)

    return styler
