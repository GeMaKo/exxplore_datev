import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from plot import create_dataset_figure, create_decision_plot
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler


import pandas as pd
from data import DATASETS, get_train_test_data
from models import fit_estimator, models
from plot import create_dataset_figure, create_decision_plot

mpl.style.use("default")


st.set_page_config(layout="wide")
st.title("Exxplore - Machine Learning Visualized")

with st.sidebar:
    st.subheader("Choose your dataset")
    
    data_noise = st.slider("Select noise", min_value=0.05, max_value=1.0, step=0.05, value=0.3)
    data_ratio = st.slider("Select train/test ratio", min_value=0.1, max_value=0.9, step=0.05, value=0.8)
    
    dataset_name = st.selectbox("Select dataset", options=DATASETS.keys())
    data_func = DATASETS[dataset_name]
    fig, ax = plt.subplots(1, 1)
    
    X, y = data_func(data_noise)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = get_train_test_data(X, y, data_ratio)
    
    data_fig = create_dataset_figure(X_train, X_test, y_train, y_test)
    
    st.pyplot(data_fig, clear_figure=True)


st.subheader("Classifier")

widget_key = 0
for model_name, model_dict in models.items():
    with st.container():
        st.write(model_name)
        paramter_values = {}
        left, right = st.columns([60,40])
        with left:
            for parameter_name, properties in model_dict['parameters'].items():
                if properties["type"] == "select_slider":
                    paramter_values[parameter_name] = st.select_slider(
                        parameter_name,
                        options=properties["values"],
                        key=widget_key)
                elif properties["type"] == "checkbox":
                    paramter_values[parameter_name] = st.checkbox(
                        parameter_name,
                        key=widget_key)
                elif properties["type"] == "selection":
                    paramter_values[parameter_name] = st.selectbox(
                        parameter_name,
                        options=properties["values"],
                        key=widget_key)
                else:
                    pass
                widget_key += 1
            
            
            fitted_estimator = fit_estimator(models[model_name]["estimator"](**paramter_values), X_train, y_train)
            scoring_classification_methods = ["f1", "accuracy"]

            cv_scores = cross_validate(fitted_estimator, X_train, y_train, cv=3,
                scoring=scoring_classification_methods,
                return_train_score=True)
            df_cv_results = pd.DataFrame.from_dict(cv_scores)
            df_cv_results = df_cv_results.rename({0: "CV 1", 1: "CV 2", 2: "CV 3"}, axis='columns')
            st.dataframe(df_cv_results)
            models[model_name]["fitted_estimator"] = fitted_estimator
            

        with right:
            discretize = st.checkbox("Discretize prediction", key=f"discretize_{widget_key}")
            fig = create_decision_plot(fitted_estimator, X_train, X_test, y_train, y_test, discretize)
            st.pyplot(fig)
            test_score = fitted_estimator.score(X_test, y_test)
            st.write(f"Test scores is: {test_score}")
            widget_key += 1
            
        st.markdown("---")
