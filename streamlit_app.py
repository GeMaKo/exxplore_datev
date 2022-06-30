import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data import (DATASETS, create_figure, get_circle_data, get_isolated_data,
                  get_moon_data, get_xor_data)

mpl.style.use("default")


st.set_page_config(layout="wide")
st.title("Exxplore - Machine Learning Visualized")

left, center, right = st.columns([25,50,25])

with left:
    st.write("Choose your dataset")
    
    data_noise = st.slider("Select noise", min_value=0.0, max_value=1.0, step=0.05, value=0.3)
    data_ratio = st.slider("Select train/test ratio", min_value=0.0, max_value=1.0, step=0.05, value=0.8)
    
    dataset_name = st.selectbox("Select dataset", options=DATASETS.keys())
    data_func = DATASETS[dataset_name]
    fig, ax = plt.subplots(1, 1)
    
    X, y = data_func(data_noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=data_ratio)
    create_figure(X_train, X_test, y_train, y_test, ax)
    plt.axis('tight')
    plt.xticks([])
    plt.yticks([])
    plt.grid(visible=True, axis="both")
    plt.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


with center:
    ...
    

with right:
    model = LogisticRegression()
    

    
    
    