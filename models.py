from sklearn.linear_model import RidgeClassifier
import numpy as np

# Ridge_Classifier
ridge_classifier_parameters = {
    "alpha":np.logspace(0.00001, 1.0, num=6),
    "fit_intercept":[True, False],
    "solver":["auto", "svd", "cholesky", "sqr", "sparse_cg", "sag", "saga", "lbfgs"],
}

for parameter, values in ridge_classifier_parameters.items():
    st.select_slider(
        'Solver',
        options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])

ridge_classifier_model = RidgeClassifier(alpha=alpha, fit_intercept=fit_intercept, solver=solver)
ridge_classifier_model.fit(X, y)





