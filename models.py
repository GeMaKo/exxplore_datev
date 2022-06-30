from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# Ridge_Classifier
ridge_classification_defintion =  {
    "estimator": RidgeClassifier,
    "parameters": {    
        "alpha": {
            "type": "slider",
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
            "type": "slider",
            "values": np.arange(1, 5),
        },
        "min_samples_split": {
            "type": "slider",
            "values": np.arange(2, 5),
        },
        "min_samples_leaf": {
            "type": "slider",
            "values": np.arange(1, 5),
        }   
    }
}

models = {
    "ridge_classification": ridge_classification_defintion,
    "decision_tree_classification": decision_tree_classification_definition,
}





