import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.inspection import DecisionBoundaryDisplay


def create_dataset_figure(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    fig = plt.figure()
    
    lines = plt.plot(X_train[y_train==0, 0], X_train[y_train==0, 1], marker="o", markersize=10, linestyle="None", label="train")
    first_color = lines[0].get_color()
    lines = plt.plot(X_train[y_train==1, 0], X_train[y_train==1, 1], marker="o", markersize=10, linestyle="None")
    second_color = lines[0].get_color()
    
    plt.plot(X_test[y_test==0, 0], X_test[y_test==0, 1], color=first_color, marker="x", markersize=15, linestyle="None", label="test")
    plt.plot(X_test[y_test==1, 0], X_test[y_test==1, 1], color=second_color, marker="x", markersize=15, linewidth=2, linestyle="None")
    
    return fig


def create_decision_plot(estimator: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray):
    fig = plt.figure()
    
    disp = DecisionBoundaryDisplay.from_estimator(estimator, X_train, alpha=0.5)
    
    disp.ax_.scatter(X_train, )
    
    return fig