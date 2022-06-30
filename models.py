from sklearn.linear_model import RidgeClassifier
import numpy as np

# Ridge_Classifier
ridge_classifier_parameters = {
    "slider": {
        "alpha":np.logspace(-5, 0, num=6),
    },
    "checkbox": {
        "fit_intercept":[True, False],
    },
    "selection": {
        "solver":["auto", "svd", "cholesky", "sqr", "sparse_cg", "sag", "saga", "lbfgs"],
    }   
}
ridge_classifier = RidgeClassifier





