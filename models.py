import numpy as np

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Ridge_Classifier
ridge_classification_defintion =  {
    "estimator": RidgeClassifier,
    "parameters": {    
        #"alpha": {
        #    "type": "select_slider",
        #    "values": np.logspace(-5, 0, num=6),
        #},
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
            "values": np.arange(1, 14),
        },
        #"min_samples_split": {
        #    "type": "select_slider",
        #    "values": np.arange(2, 5),
        #},
        #"min_samples_leaf": {
        #    "type": "select_slider",
        #    "values": np.arange(1, 5),
        #}   
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
            "values": np.arange(1, 11),
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

svm_classification_definition = {
    "estimator": SVC,
    "parameters": {
        "kernel": {
            "type": "selection",
            "values": ["linear", "poly", "rbf"],
        },
        #"C": {
        #    "type": "select_slider",
        #    "values": np.logspace(-5, 0, num=6),
        #},
        "degree": {
            "type": "select_slider",
            "values": np.arange(1, 7),
        },
        "shrinking": {
            "type": "checkbox",
            "values": [True, False],
        },
    }  
}


models = {
    "Ridge Classifier": ridge_classification_defintion,
    "SVM Classifier": svm_classification_definition,
    "Decision Tree Classifier": decision_tree_classification_definition,
    "Random Forest Classifier": random_forest_classification_definition,
    
}


st.cache(allow_output_mutation=True)
def init_model(estimator, params: dict):
    return estimator(**params)


@st.cache()
def fit_estimator_with_cv(estimator, X_train: np.ndarray, y_train: np.ndarray, scoring_classification_methods: list):
    cv_scores = cross_validate(estimator, X_train, y_train,
                scoring=scoring_classification_methods,
                return_train_score=True)
    
    return cv_scores


@st.cache
def get_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, output_dict=False)


@st.cache(allow_output_mutation=True)
def fit_estimator(estimator, X_train: np.ndarray, y_train: np.ndarray):
    return estimator.fit(X_train, y_train)


@st.cache
def get_predictions(estimator, X_train, y_train) -> np.ndarray:
    return cross_val_predict(estimator, X_train, y_train, cv=3)
