import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.inspection import DecisionBoundaryDisplay
import streamlit as st


def create_dataset_figure(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, ax = None):
    if ax is None:
        fig = plt.figure()
        
        lines = plt.plot(X_train[y_train==0, 0], X_train[y_train==0, 1], marker="o", markersize=10, linestyle="None", label="train")
        first_color = lines[0].get_color()
        lines = plt.plot(X_train[y_train==1, 0], X_train[y_train==1, 1], marker="o", markersize=10, linestyle="None")
        second_color = lines[0].get_color()
        
        plt.plot(X_test[y_test==0, 0], X_test[y_test==0, 1], color=first_color, marker="x", markersize=15, linestyle="None", label="test")
        plt.plot(X_test[y_test==1, 0], X_test[y_test==1, 1], color=second_color, marker="x", markersize=15, linewidth=2, linestyle="None")
        
        return fig
    else:
        lines = ax.plot(X_train[y_train==0, 0], X_train[y_train==0, 1], marker="o", markersize=10, linestyle="None", label="train")
        first_color = lines[0].get_color()
        lines = ax.plot(X_train[y_train==1, 0], X_train[y_train==1, 1], marker="o", markersize=10, linestyle="None")
        second_color = lines[0].get_color()
        
        ax.plot(X_test[y_test==0, 0], X_test[y_test==0, 1], color=first_color, marker="x", markersize=15, linestyle="None", label="test")
        ax.plot(X_test[y_test==1, 0], X_test[y_test==1, 1], color=second_color, marker="x", markersize=15, linewidth=2, linestyle="None")
        
        return ax


def create_decision_plot(estimator: BaseEstimator, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    disp = DecisionBoundaryDisplay.from_estimator(estimator, np.r_[X_train, X_test], alpha=0.5, response_method="auto")
    
    create_dataset_figure(X_train, X_test, y_train, y_test, ax=disp.ax_)
    
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    disp.figure_.colorbar(mpl.cm.ScalarMappable(norm=norm), ax=disp.ax_)
    
    return disp.figure_
