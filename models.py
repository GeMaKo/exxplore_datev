import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import streamlit as st
import time


# Ridge_Classifier
ridge_classification_defintion =  {
    "estimator": RidgeClassifier,
    "parameters": {    
        "alpha": {
            "type": "select_slider",
            "values": np.logspace(-5, 0, num=6),
        },
        "fit_intercept": {
            "type": "checkbox",
            "values": [True, False],
        },
        #"solver": {
        #    "type": "selection",
        #    "values": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"],
        #}  
    }
}

decision_tree_classification_definition = {
    "estimator": DecisionTreeClassifier,
    "parameters": {    
        #"criterion": {
        #    "type": "selection",
        #    "values": ["gini", "entropy", "log_loss"],
        #},
        "max_depth": {
            "type": "select_slider",
            "values": np.arange(1, 5),
        },
        "min_samples_split": {
            "type": "select_slider",
            "values": np.arange(2, 5),
        },
        "min_samples_leaf": {
            "type": "select_slider",
            "values": np.arange(1, 5),
        }   
    }
}

random_forest_classification_definition = {
    "estimator": RandomForestClassifier,
    "parameters": {
        "n_estimators": {
            "type": "select_slider",
            "values": np.arange(10, 101, 10),
        },
        "max_depth": {
            "type": "select_slider",
            "values": np.arange(1, 5),
        },
        "min_samples_split": {
            "type": "select_slider",
            "values": np.arange(2, 5),
        },
        "min_samples_leaf": {
            "type": "select_slider",
            "values": np.arange(1, 5),
        },
        "bootstrap": {
            "type": "checkbox",
            "values": [True, False],
        },
    }  

}

models = {
    "Ridge Classifier": ridge_classification_defintion,
    "Decision Tree Classifier": decision_tree_classification_definition,
    "Random Forest Classifier": random_forest_classification_definition,
}


@st.cache(allow_output_mutation=True)
def fit_estimator(estimator, X_train, y_train) -> tuple[sklearn.base.BaseEstimator, float]:
    start_time = time.perf_counter()
    estimator.fit(X_train, y_train)
    end_time = time.perf_counter()
    return estimator, end_time - start_time


