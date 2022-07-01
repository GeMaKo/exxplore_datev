import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from plot import create_dataset_figure, create_decision_plot
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

from data import DATASETS, get_train_test_data
from models import models, fit_estimator

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
    X_train, X_test, y_train, y_test = get_train_test_data(X, y, data_ratio)
    
    data_fig = create_dataset_figure(X_train, X_test, y_train, y_test)
    
    st.pyplot(data_fig, clear_figure=True)




st.subheader("Classifier")

widget_key = 0
for model_name, model_dict in models.items():
    with st.container():
        st.write(model_name)
        paramter_values = {}
        left, right = st.columns([50,50])
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
            models[model_name]["fitted_estimator"] = fitted_estimator
            score = fitted_estimator.score(X_test, y_test)
            st.write(f"score is: {score}")

        with right:
            fig = create_decision_plot(fitted_estimator, X_train, X_test, y_train, y_test)
            st.pyplot(fig)
            
        st.markdown("---")
