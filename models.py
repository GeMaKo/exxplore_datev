from sklearn.linear_model import RidgeClassifier
import numpy as np


# Ridge_Classifier
model_dict = {
    "ridge_classification": {
        "estimator": RidgeClassifier,
        "parameters": {
            "slider": {
                "alpha":np.logspace(-5, 0, num=6),
            },
            "checkbox": {
                "fit_intercept":[True, False],
            },
            "selection": {
                "solver":["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"],
            }  
        }
    }
}





