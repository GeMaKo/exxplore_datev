import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

from data import (DATASETS, create_figure, get_circle_data, get_isolated_data,
                  get_moon_data, get_xor_data)
from models import ridge_classifier_parameters, ridge_classifier


mpl.style.use("default")

st.title("Exxplore - Machine Learning Visualized")

left, center, right = st.columns(3)

with left:
    
    data_noise = st.slider("noise", min_value=0.0, max_value=1.0, step=0.05, value=0.3)
    
    dataset_name = st.selectbox("Select dataset", options=DATASETS.keys())
    data_func = DATASETS[dataset_name]
    fig, ax = plt.subplots(1, 1)
    
    X, y = data_func(data_noise)
    create_figure(X, y, ax)
    plt.axis('tight')
    plt.xticks([])
    plt.yticks([])
    plt.grid(visible=True, axis="both")
    st.pyplot(fig, clear_figure=True)


with center:
    paramter_values = {}
    for parameter, values in ridge_classifier_parameters['slider'].items():
        paramter_values[parameter] = st.select_slider(
            parameter,
            options=values)
    for parameter, values in ridge_classifier_parameters['checkbox'].items():
        paramter_values[parameter] = st.checkbox(
            parameter)
    for parameter, values in ridge_classifier_parameters['selection'].items():
        paramter_values[parameter] = st.selectbox(
            parameter,
            options=values)
    estimator = ridge_classifier(**paramter_values)
    clf = estimator.fit(X, y)
    score = clf.score(X, y)
    st.write(f"score is: {score}")
    

with right:
    ...

    
    
    